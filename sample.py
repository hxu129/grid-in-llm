"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "0 5 0" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# Configuration for training GPT on maze navigation data
# Optimized for path learning task

# I/O
out_dir = 'out-maze-nav'
eval_interval = 250
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'resume'

# wandb logging
wandb_log = False
wandb_project = 'maze-nav'
wandb_run_name = 'maze-nav-gpt'

# data
dataset = 'maze/maze_nav_data'  # This should match your maze data directory
gradient_accumulation_steps = 1  # Reduced for smaller sequences
batch_size = 32  # Reasonable batch size for path learning
max_seq_len = 512  # Maximum sequence length for any path (no artificial limit)

# model - smaller model suitable for maze navigation
n_layer = 6   # Fewer layers for simpler task
n_head = 6    # Fewer attention heads
n_embd = 192  # Smaller embedding dimension
dropout = 0.1 # Some dropout for regularization
bias = False  # Cleaner model

# adamw optimizer
learning_rate = 3e-4  # Slightly higher learning rate
max_iters = 5000      # Fewer iterations needed
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 100    # Quick warmup
lr_decay_iters = 5000 # Match max_iters
min_lr = 3e-5

# DDP settings
backend = 'nccl'

# system
device = 'cuda'
# Note: dtype check is done in train.py
dtype = 'bfloat16'  # will be validated in train.py
compile = True 


# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # Handle both string and integer tokens for maze navigation
    stoi, itos = meta['stoi'], meta['itos']
    
    def encode(s):
        if isinstance(s, str):
            if s.isdigit():
                # Handle numeric node IDs
                return [stoi[s]] if s in stoi else [int(s)]
            else:
                # Handle direction tokens or other strings
                tokens = s.split()
                return [stoi[token] if token in stoi else int(token) if token.isdigit() else stoi.get(token, 0) for token in tokens]
        elif isinstance(s, list):
            # Handle list of tokens
            return [stoi[str(token)] if str(token) in stoi else (int(token) if isinstance(token, str) and token.isdigit() else token) for token in s]
        else:
            return [stoi[str(s)] if str(s) in stoi else s]
    
    def decode(l):
        if 'direction_tokens' in meta:
            # Maze navigation specific decoding
            result = []
            for i in l:
                token = itos.get(i, str(i))
                result.append(token)
            return ' '.join(result)
        else:
            # Standard string concatenation
            return ''.join([itos.get(i, str(i)) for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

print(f"Starting with: {start}")
start_ids = encode(start)
print(f"Encoded as: {start_ids}")
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            generated_sequence = y[0].tolist()
            decoded_output = decode(generated_sequence)
            print(f"Sample {k+1}:")
            print(f"Raw tokens: {generated_sequence}")
            print(f"Decoded: {decoded_output}")
            print('---------------')
