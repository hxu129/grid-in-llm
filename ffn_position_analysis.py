import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from contextlib import nullcontext
from model import GPTConfig, GPT
import ecco
from typing import Dict, List, Tuple, Optional
import re

class FFNActivationCollector:
    """
    Collects FFN activations from maze navigation model when generating position tokens.
    Focuses on intermediate activations (after GELU, before second linear layer).
    """
    
    def __init__(self, model_path: str = 'out-maze-nav', grid_size: int = 8):
        """
        Initialize the activation collector.
        
        Args:
            model_path: Path to the trained model directory
            grid_size: Size of the maze grid (8 for 8x8)
        """
        self.grid_size = grid_size
        self.n_positions = grid_size * grid_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model and setup
        self.model, self.tokenizer = self._load_model(model_path)
        self.n_layers = self.model.config.n_layer
        self.n_embd = self.model.config.n_embd
        self.ffn_size = 4 * self.n_embd  # Size of FFN intermediate layer
        
        # Storage for activations
        self.position_activations = defaultdict(lambda: defaultdict(list))  # {layer: {position: [activations]}}
        self.hooks = {}
        
        print(f"Model loaded: {self.n_layers} layers, {self.n_embd} embedding dim, {self.ffn_size} FFN size")
        print(f"Grid size: {grid_size}x{grid_size} = {self.n_positions} positions")
    
    def _load_model(self, model_path: str) -> Tuple[GPT, object]:
        """Load the trained maze navigation model and tokenizer."""
        # Load model (following logit_lens.ipynb pattern)
        ckpt_path = os.path.join(model_path, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        
        # Clean up state dict
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        
        # Load tokenizer metadata
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        stoi, itos = meta['stoi'], meta['itos']
        
        def encode(s):
            if isinstance(s, str):
                tokens = s.split()
                return [stoi[token] if token in stoi else int(token) if token.isdigit() else stoi.get(token, 0) for token in tokens]
            return s
        
        def decode(l):
            result = []
            for i in l:
                token = itos.get(i, str(i))
                result.append(token)
            return ' '.join(result)
        
        # Create tokenizer-like object
        class MazeTokenizer:
            def __init__(self, encode_fn, decode_fn):
                self.encode = encode_fn
                self.decode = decode_fn
                
            def convert_ids_to_tokens(self, ids):
                if not isinstance(ids, list):
                    ids = [ids]
                return [itos.get(i, str(i)) for i in ids]
        
        tokenizer = MazeTokenizer(encode, decode)
        
        return model, tokenizer
    
    def _load_validation_data(self, val_path: str = "data/maze/maze_nav_data/val.bin", 
                            max_samples: int = 100) -> List[List[int]]:
        """Load validation data following logit_lens.ipynb pattern."""
        val = np.memmap(val_path, dtype=np.uint16, mode="r")
        
        # Parse validation data into individual paths
        data = []
        temp = []
        for d in val:
            if d == 8 ** 2 + 4:  # End of sequence marker
                if temp:
                    data.append(temp)
                temp = []
            else:
                temp.append(int(d))
        
        # Shuffle and limit samples
        data = np.array(data, dtype=object)
        np.random.shuffle(data)
        return data[:max_samples].tolist()
    
    def _register_ffn_hooks(self):
        """Register hooks to capture FFN intermediate activations."""
        self._remove_hooks()  # Clear any existing hooks
        
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                # This hook captures activations after GELU but before c_proj
                # The input to this hook is the output of GELU
                if hasattr(self, '_current_generated_position') and self._current_generated_position is not None:
                    # Store activation for the current position being generated
                    activation = input[0].detach().float().cpu().numpy()  # Convert to float32 first
                    # Take the last token position (the one being generated)
                    last_token_activation = activation[0, -1, :]  # (ffn_size,)
                    
                    layer_key = f"layer_{layer_idx}"
                    position = self._current_generated_position
                    self.position_activations[layer_key][position].append(last_token_activation.copy())
            
            return hook_fn
        
        # Register hooks for each layer's MLP c_proj (captures input after GELU)
        for layer_idx in range(self.n_layers):
            layer_name = f"transformer.h.{layer_idx}.mlp.c_proj"
            layer_module = dict(self.model.named_modules())[layer_name]
            hook = layer_module.register_forward_hook(create_hook(layer_idx))
            self.hooks[layer_name] = hook
            
        print(f"Registered hooks for {len(self.hooks)} FFN layers")
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks = {}
    
    def _is_position_token(self, token_id: int) -> bool:
        """Check if a token represents a grid position (0-63 for 8x8 grid)."""
        token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        try:
            pos = int(token_str)
            return 0 <= pos < self.n_positions
        except (ValueError, TypeError):
            return False
    
    def _extract_path_completion_samples(self, paths: List[List[int]], 
                                       completion_length: int = 10) -> List[Tuple[List[int], List[int]]]:
        """
        Extract (prefix, target) pairs for path completion.
        
        Args:
            paths: List of complete paths
            completion_length: Number of tokens to predict
            
        Returns:
            List of (prefix, target_positions) tuples
        """
        samples = []
        
        for path in paths:
            if len(path) < completion_length + 3:  # Need at least start, end, first_pos + completion
                continue
                
            # Take first 3 tokens as prefix (start, end, first_pos)
            prefix = path[:3]
            
            # Extract position tokens from the remainder (skip direction tokens)
            remaining_tokens = path[3:]
            target_positions = []
            
            for token in remaining_tokens:
                if self._is_position_token(token) and len(target_positions) < completion_length:
                    target_positions.append(token)
                if len(target_positions) >= completion_length:
                    break
            
            if len(target_positions) >= 3:  # Need at least a few positions to predict
                samples.append((prefix, target_positions))
        
        return samples
    
    def collect_activations(self, max_samples: int = 100, completion_length: int = 8):
        """
        Main method to collect FFN activations during position generation.
        
        Args:
            max_samples: Maximum number of validation samples to process
            completion_length: Number of tokens to generate per sample
        """
        print("Loading validation data...")
        validation_paths = self._load_validation_data(max_samples=max_samples * 2)  # Load extra in case some are filtered
        
        print("Extracting path completion samples...")
        completion_samples = self._extract_path_completion_samples(
            validation_paths, completion_length=completion_length
        )[:max_samples]
        
        print(f"Processing {len(completion_samples)} completion samples...")
        
        # Register hooks
        self._register_ffn_hooks()
        
        processed_samples = 0
        
        try:
            with torch.no_grad():
                for sample_idx, (prefix, target_positions) in enumerate(completion_samples):
                    if sample_idx % 10 == 0:
                        print(f"Processing sample {sample_idx}/{len(completion_samples)}")
                    
                    try:
                        # Convert prefix to tensor
                        input_ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
                        
                        # Generate tokens one by one
                        current_input = input_ids.clone()
                        
                        for target_pos in target_positions[:completion_length]:
                            # Set the current position we're trying to generate
                            self._current_generated_position = target_pos
                            
                            try:
                                # Forward pass to generate next token
                                # Use autocast only if supported and beneficial
                                if self.device == 'cuda' and torch.cuda.is_bf16_supported():
                                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                                        outputs = self.model(input_ids=current_input)
                                        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
                                else:
                                    # No autocast for CPU or unsupported GPU
                                    outputs = self.model(input_ids=current_input)
                                    logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
                                
                                # Get the predicted token (we know it should be target_pos)
                                next_token_logits = logits[:, -1, :]
                                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                                
                                # Append the predicted token and continue
                                current_input = torch.cat([current_input, next_token], dim=1)
                                
                                # Verify we're generating position tokens
                                if self._is_position_token(next_token.item()):
                                    # Activation was captured in the hook
                                    pass
                                
                                # Prevent sequences from getting too long
                                if current_input.size(1) > 100:
                                    break
                                    
                            except Exception as e:
                                print(f"Warning: Error processing position {target_pos} in sample {sample_idx}: {e}")
                                break  # Skip rest of this sample
                        
                        processed_samples += 1
                        
                    except Exception as e:
                        print(f"Warning: Error processing sample {sample_idx}: {e}")
                        continue  # Skip this sample entirely
        
        finally:
            # Clean up
            self._current_generated_position = None
            self._remove_hooks()
        
        print(f"Processed {processed_samples} samples")
        self._print_collection_summary()
    
    def _print_collection_summary(self):
        """Print summary of collected activations."""
        print("\nActivation Collection Summary:")
        print("=" * 50)
        
        for layer_key in sorted(self.position_activations.keys()):
            layer_data = self.position_activations[layer_key]
            total_activations = sum(len(acts) for acts in layer_data.values())
            unique_positions = len(layer_data)
            
            print(f"{layer_key}: {total_activations} activations across {unique_positions} unique positions")
            
            # Show position coverage
            if layer_data:
                positions_with_data = sorted(layer_data.keys())
                print(f"  Positions: {positions_with_data[:10]}{'...' if len(positions_with_data) > 10 else ''}")
                
                # Show activation counts per position
                pos_counts = {pos: len(acts) for pos, acts in layer_data.items()}
                avg_count = np.mean(list(pos_counts.values()))
                print(f"  Avg activations per position: {avg_count:.1f}")
    
    def normalize_and_create_matrices(self, normalization: str = 'z_score') -> Dict[str, np.ndarray]:
        """
        Normalize activations and create position-neuron matrices.
        
        Args:
            normalization: 'z_score', 'min_max', or 'none'
            
        Returns:
            Dictionary mapping layer names to (n_positions, n_neurons) matrices
        """
        print(f"\nCreating position-neuron matrices with {normalization} normalization...")
        
        matrices = {}
        
        for layer_key in sorted(self.position_activations.keys()):
            layer_data = self.position_activations[layer_key]
            
            if not layer_data:
                print(f"Warning: No data for {layer_key}")
                continue
            
            # Initialize matrix: (n_positions, n_neurons)
            matrix = np.full((self.n_positions, self.ffn_size), np.nan)
            
            # Fill matrix with averaged activations
            for position, activations_list in layer_data.items():
                if activations_list:
                    # Average multiple activations for the same position
                    avg_activation = np.mean(activations_list, axis=0)
                    matrix[position, :] = avg_activation
            
            # Apply normalization
            if normalization == 'z_score':
                # Z-score normalization per neuron (across positions)
                for neuron_idx in range(self.ffn_size):
                    neuron_data = matrix[:, neuron_idx]
                    valid_data = neuron_data[~np.isnan(neuron_data)]
                    if len(valid_data) > 1:
                        mean_val = np.mean(valid_data)
                        std_val = np.std(valid_data)
                        if std_val > 0:
                            matrix[:, neuron_idx] = (neuron_data - mean_val) / std_val
            
            elif normalization == 'min_max':
                # Min-max normalization per neuron
                for neuron_idx in range(self.ffn_size):
                    neuron_data = matrix[:, neuron_idx]
                    valid_data = neuron_data[~np.isnan(neuron_data)]
                    if len(valid_data) > 1:
                        min_val = np.min(valid_data)
                        max_val = np.max(valid_data)
                        if max_val > min_val:
                            matrix[:, neuron_idx] = (neuron_data - min_val) / (max_val - min_val)
            
            # Store matrix
            matrices[layer_key] = matrix
            
            # Print statistics
            valid_entries = ~np.isnan(matrix)
            coverage = np.mean(valid_entries) * 100
            print(f"{layer_key}: {matrix.shape} matrix, {coverage:.1f}% coverage")
        
        return matrices
    
    def save_results(self, matrices: Dict[str, np.ndarray], 
                    save_path: str = "ffn_position_analysis_results.npz"):
        """Save the analysis results."""
        print(f"\nSaving results to {save_path}...")
        
        # Prepare data for saving
        save_data = {
            'grid_size': self.grid_size,
            'n_positions': self.n_positions,
            'n_layers': self.n_layers,
            'ffn_size': self.ffn_size,
            **matrices  # Unpack all layer matrices
        }
        
        # Save metadata
        metadata = {
            'layer_names': list(matrices.keys()),
            'matrix_shape': f"({self.n_positions}, {self.ffn_size})",
            'description': "FFN neuron activations when generating specific grid positions"
        }
        save_data['metadata'] = metadata
        
        np.savez_compressed(save_path, **save_data)
        print(f"Results saved! Matrices for {len(matrices)} layers")
        
        return save_path

def main():
    """Main execution function."""
    print("Starting FFN Position Analysis...")
    print("=" * 60)
    
    # Initialize collector
    collector = FFNActivationCollector(
        model_path='out-maze-nav',
        grid_size=8
    )
    
    # Collect activations
    collector.collect_activations(
        max_samples=50,  # Start with fewer samples for testing
        completion_length=8
    )
    
    # Create normalized matrices
    matrices = collector.normalize_and_create_matrices(normalization='z_score')
    
    # Save results
    save_path = collector.save_results(matrices)
    
    print(f"\nAnalysis complete! Results saved to {save_path}")
    print("\nTo load results later:")
    print(f"data = np.load('{save_path}')")
    print("layer_0_matrix = data['layer_0']")
    
    return matrices, save_path

if __name__ == "__main__":
    matrices, save_path = main() 