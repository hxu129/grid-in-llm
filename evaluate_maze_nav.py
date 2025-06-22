"""
Maze Navigation Model Evaluation

Comprehensive evaluation module for GPT models trained on maze navigation tasks.
Provides multiple evaluation metrics including path generation accuracy, 
next-step prediction accuracy, and path validity assessment.
"""

import os
import time
import pickle
import random
from typing import Dict, List, Tuple, Optional, Set
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

class MazeNavEvaluator:
    """Evaluator for maze navigation GPT models."""
    
    def __init__(self, model, data_dir: str, device: str, ctx=None):
        """
        Initialize evaluator.
        
        Args:
            model: The GPT model to evaluate
            data_dir: Directory containing validation data and meta.pkl
            device: Device to run evaluation on
            ctx: Context manager for mixed precision
        """
        self.model = model
        self.data_dir = data_dir
        self.device = device
        self.ctx = ctx if ctx is not None else nullcontext()
        
        # Load metadata
        meta_path = os.path.join(data_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            raise ValueError(f"meta.pkl not found in {data_dir}")
            
        with open(meta_path, 'rb') as f:
            self.meta = pickle.load(f)
            
        self.vocab_size = self.meta['vocab_size']
        self.eos_token_id = None
        self.padding_token_id = None
        
        # Determine EOS token ID
        if 'stoi' in self.meta and '\n' in self.meta['stoi']:
            self.eos_token_id = self.meta['stoi']['\n']
        elif 'vocab_size' in self.meta:
            self.eos_token_id = self.meta['vocab_size'] - 1
            
        # Determine padding token ID
        if 'padding_token_id' in self.meta:
            self.padding_token_id = self.meta['padding_token_id']
        elif 'stoi' in self.meta and '<PAD>' in self.meta['stoi']:
            self.padding_token_id = self.meta['stoi']['<PAD>']
        else:
            self.padding_token_id = self.meta['vocab_size'] - 2
            
        print(f"Evaluator initialized with vocab_size={self.vocab_size}, "
              f"eos_token_id={self.eos_token_id}, padding_token_id={self.padding_token_id}")

    def parse_validation_sequences(self, split='val') -> List[List[int]]:
        """
        Parse validation data into individual sequences.
        
        Args:
            split: 'val' or 'train'
            
        Returns:
            List of sequences (each sequence is a list of token IDs)
        """
        data_file = f'{split}.bin'
        data_path = os.path.join(self.data_dir, data_file)
        
        if not os.path.exists(data_path):
            raise ValueError(f"Data file {data_path} not found")
            
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
        # Split by EOS token
        sequences = []
        start_idx = 0
        
        for i, token_id in enumerate(data):
            if token_id == self.eos_token_id:
                if i > start_idx:  # Non-empty sequence
                    seq = data[start_idx:i].tolist()
                    sequences.append(seq)
                start_idx = i + 1
                
        # Handle last sequence if file doesn't end with EOS
        if start_idx < len(data):
            seq = data[start_idx:].tolist()
            if seq:
                sequences.append(seq)
                
        return sequences

    def validate_sequence_format(self, sequence: List[int]) -> bool:
        """
        Validate that a sequence follows the expected format:
        [start_node, end_node, start_node, direction1, node1, direction2, node2, ...]
        
        Args:
            sequence: List of token IDs
            
        Returns:
            True if sequence format is valid
        """
        if len(sequence) < 5:  # Minimum: start, end, start, direction, node
            return False
            
        # Check if first and third tokens are the same (start node)
        if sequence[0] != sequence[2]:
            return False
            
        # Check if remaining tokens come in direction-node pairs
        remaining = sequence[3:]
        if len(remaining) % 2 != 0:
            return False
            
        return True

    def extract_path_from_sequence(self, sequence: List[int]) -> Tuple[int, int, List[int]]:
        """
        Extract start, end, and path nodes from a sequence.
        
        Args:
            sequence: Token sequence [start, end, start, dir1, node1, dir2, node2, ...]
            
        Returns:
            Tuple of (start_node, end_node, path_nodes)
        """
        if len(sequence) < 5:
            return None, None, []
            
        start_node = sequence[0]
        end_node = sequence[1]
        
        # Extract path: start with start_node, then every second token from position 4
        path = [start_node]
        for i in range(4, len(sequence), 2):  # Skip directions, take nodes
            if i < len(sequence):
                path.append(sequence[i])
                
        return start_node, end_node, path

    @torch.no_grad()
    def evaluate_next_step_prediction(self, max_sequences: int = 1000) -> Dict[str, float]:
        """
        Evaluate next-step prediction accuracy.
        
        Args:
            max_sequences: Maximum number of sequences to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating next-step prediction accuracy...")
        self.model.eval()
        
        sequences = self.parse_validation_sequences('val')
        if max_sequences:
            sequences = sequences[:max_sequences]
            
        total_predictions = 0
        correct_predictions = 0
        
        for seq in sequences:
            if not self.validate_sequence_format(seq):
                continue
                
            # Convert to tensor
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device)[None, ...]
            
            # Evaluate prediction for each position (starting from position 3, since first 2 are skipped)
            for i in range(2, len(seq) - 1):  # Skip first 2, don't predict past end
                input_ids = seq_tensor[:, :i+1]
                target = seq[i+1]
                
                with self.ctx:
                    logits, _ = self.model(input_ids=input_ids)
                    
                # Get prediction for next token
                next_token_logits = logits[0, -1, :]
                predicted_token = torch.argmax(next_token_logits).item()
                
                total_predictions += 1
                if predicted_token == target:
                    correct_predictions += 1
                    
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'next_step_accuracy': accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions
        }

    @torch.no_grad()
    def evaluate_path_generation(self, max_sequences: int = 100, max_new_tokens: int = 200) -> Dict[str, float]:
        """
        Evaluate complete path generation from [start, end, start] prompts.
        
        Args:
            max_sequences: Maximum number of sequences to evaluate
            max_new_tokens: Maximum tokens to generate per sequence
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating complete path generation...")
        self.model.eval()
        
        sequences = self.parse_validation_sequences('val')
        if max_sequences:
            sequences = sequences[:max_sequences]
            
        valid_completions = 0
        exact_matches = 0
        path_validity_matches = 0
        total_attempts = 0
        
        for seq in sequences:
            if not self.validate_sequence_format(seq) or len(seq) < 5:
                continue
                
            # Use first 3 tokens as prompt [start, end, start]
            prompt = seq[:3]
            ground_truth_completion = seq[3:]  # Only the part after prompt
            original_sequence = seq  # Complete original sequence for comparison
            
            prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=self.device)[None, ...]
            
            try:
                # Generate completion
                with self.ctx:
                    generated = self._generate_sequence(
                        prompt_tensor, 
                        max_new_tokens=max_new_tokens,
                        temperature=0.0  # Greedy decoding
                    )
                
                generated_tokens = generated[0][len(prompt):].tolist()
                
                # Remove EOS token if present
                if generated_tokens and generated_tokens[-1] == self.eos_token_id:
                    generated_tokens = generated_tokens[:-1]
                
                total_attempts += 1
                
                # Create complete generated sequence
                complete_generated_sequence = prompt + generated_tokens
                
                # Debug information for first few sequences
                if total_attempts <= 3:
                    print(f"\nDebug - Sequence {total_attempts}:")
                    print(f"  Original sequence: {original_sequence}")
                    print(f"  Prompt: {prompt}")
                    print(f"  Generated tokens: {generated_tokens}")
                    print(f"  Complete generated: {complete_generated_sequence}")
                    print(f"  Match: {complete_generated_sequence == original_sequence}")
                
                # Check exact match with original sequence
                if complete_generated_sequence == original_sequence:
                    exact_matches += 1
                    valid_completions += 1
                    path_validity_matches += 1
                else:
                    # Check if generated path is valid (reaches the target)
                    if self._is_valid_path_completion(complete_generated_sequence):
                        valid_completions += 1
                        
                    # Check if the path reaches the correct end node
                    if self._reaches_target_node(complete_generated_sequence):
                        path_validity_matches += 1
                        
            except Exception as e:
                print(f"Error generating sequence: {e}")
                total_attempts += 1
                continue
                
        # Calculate metrics
        exact_match_rate = exact_matches / total_attempts if total_attempts > 0 else 0.0
        valid_completion_rate = valid_completions / total_attempts if total_attempts > 0 else 0.0
        path_validity_rate = path_validity_matches / total_attempts if total_attempts > 0 else 0.0
        
        return {
            'exact_match_rate': exact_match_rate,
            'valid_completion_rate': valid_completion_rate,
            'path_validity_rate': path_validity_rate,
            'total_attempts': total_attempts,
            'exact_matches': exact_matches,
            'valid_completions': valid_completions,
            'path_validity_matches': path_validity_matches
        }

    def _generate_sequence(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate sequence using the model's generate method or manual generation.
        
        Args:
            input_ids: Input token tensor
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated sequence tensor
        """
        try:
            # Try using model's generate method
            return self.model.generate(
                input_ids=input_ids,
                max_length=input_ids.size(1) + max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                top_k=1 if temperature == 0.0 else None,
                eos_token_id=self.eos_token_id
            )
        except Exception:
            # Fallback to manual generation
            return self._manual_generate(input_ids, max_new_tokens, temperature)

    def _manual_generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float) -> torch.Tensor:
        """Manual generation fallback."""
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            with self.ctx:
                logits, _ = self.model(input_ids=generated)
                
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-8)
            
            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits).item()
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
            # Stop if EOS token is generated
            if next_token == self.eos_token_id:
                break
                
            # Append token
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            generated = torch.cat([generated, next_token_tensor], dim=1)
            
        return generated

    def _is_valid_path_completion(self, sequence: List[int]) -> bool:
        """Check if a sequence represents a valid path (basic format validation)."""
        if not self.validate_sequence_format(sequence):
            return False
            
        # Additional validation could include checking if moves are valid in the maze
        # For now, just check format
        return True

    def _reaches_target_node(self, sequence: List[int]) -> bool:
        """Check if the generated path reaches the target end node."""
        if len(sequence) < 5:
            return False
            
        target_node = sequence[1]  # End node from prompt
        
        # Check if the last node in the path equals the target
        # Path nodes are at positions [0, 4, 6, 8, ...]
        path_positions = [0] + list(range(4, len(sequence), 2))
        
        if path_positions:
            last_node = sequence[path_positions[-1]]
            return last_node == target_node
            
        return False

    def run_comprehensive_evaluation(self, max_sequences: int = 1000) -> Dict[str, float]:
        """
        Run comprehensive evaluation including multiple metrics.
        
        Args:
            max_sequences: Maximum sequences to evaluate
            
        Returns:
            Combined evaluation results
        """
        print("Running comprehensive maze navigation evaluation...")
        
        # Next-step prediction
        next_step_results = self.evaluate_next_step_prediction(max_sequences)
        
        # Path generation (use fewer sequences for generation as it's more expensive)
        generation_sequences = min(100, max_sequences // 10)
        path_gen_results = self.evaluate_path_generation(generation_sequences)
        
        # Combine results
        results = {
            **next_step_results,
            **path_gen_results,
            'evaluation_timestamp': time.time(),
            'max_sequences_evaluated': max_sequences
        }
        
        # Print summary
        print("\n" + "="*50)
        print("MAZE NAVIGATION EVALUATION RESULTS")
        print("="*50)
        print(f"Next-step prediction accuracy: {results['next_step_accuracy']:.4f} "
              f"({results['correct_predictions']}/{results['total_predictions']})")
        print(f"Exact path match rate: {results['exact_match_rate']:.4f} "
              f"({results['exact_matches']}/{results['total_attempts']})")
        print(f"Valid completion rate: {results['valid_completion_rate']:.4f} "
              f"({results['valid_completions']}/{results['total_attempts']})")
        print(f"Path validity rate: {results['path_validity_rate']:.4f} "
              f"({results['path_validity_matches']}/{results['total_attempts']})")
        print("="*50)
        
        return results


def evaluate_maze_model(model, data_dir: str, device: str, ctx=None, max_sequences: int = 1000) -> Dict[str, float]:
    """
    Convenience function to evaluate a maze navigation model.
    
    Args:
        model: GPT model to evaluate
        data_dir: Directory containing validation data
        device: Device to run evaluation on
        ctx: Context manager for mixed precision
        max_sequences: Maximum sequences to evaluate
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = MazeNavEvaluator(model, data_dir, device, ctx)
    return evaluator.run_comprehensive_evaluation(max_sequences)


if __name__ == "__main__":
    # Example usage - this would need to be adapted based on your model loading
    print("Maze Navigation Evaluator")
    print("This module provides comprehensive evaluation for maze navigation GPT models.")
    print("Import and use evaluate_maze_model() function to evaluate your trained model.") 