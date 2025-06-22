"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

TRAINING MODIFICATION: The model is trained with a modified loss function that skips
predicting the 2nd and 3rd tokens. In inference, the first 3 tokens will be provided
as prompt, and the model only needs to predict from the 4th token onwards.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
max_seq_len = 1024  # Maximum sequence length for sinusoidal embeddings
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# -----------------------------------------------------------------------------
# exec configurator
exec(open('configurator.py').read()) # overrides from command line or config file

config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Check if this is path data by looking for meta.pkl
    meta_path = os.path.join(data_dir, 'meta.pkl')
    is_path_data = os.path.exists(meta_path)
    
    if is_path_data:
        # For path data: sample complete sequences instead of fixed blocks
        # This requires parsing the sequences from the flattened token stream
        return get_path_batch(data, split)
    else:
        # Original behavior for continuous text data
        ix = torch.randint(len(data) - max_seq_len, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+max_seq_len]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+max_seq_len]).astype(np.int64)) for i in ix])
        
        # Modify targets: don't compute loss for first 2 positions (predicting tokens 2 and 3)
        # Set first 2 positions to -1 which will be ignored by cross_entropy
        y[:, :2] = -1
        
        # For continuous text, create attention mask with all 1s (no padding)
        attention_mask = torch.ones_like(x)
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
            attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
            attention_mask = attention_mask.to(device)
        return x, y, attention_mask

def get_path_batch(data, split):
    """Get batch for path learning with variable-length sequences."""
    # For maze path data, we reconstruct sequences from the flattened data
    # by splitting on the EOS token.
    
    if eos_token_id is None:
        raise ValueError("get_path_batch is called but eos_token_id is not set. Is meta.pkl present?")
    
    if padding_token_id is None:
        raise ValueError("get_path_batch is called but padding_token_id is not set. Is meta.pkl present?")

    # Find all indices of the newline token
    # The data is a numpy memmap, so we can use numpy operations on it
    newline_indices = np.where(data == eos_token_id)[0]
    
    # Create sequences by splitting the data array based on the newline token
    sequences = []
    start_idx = 0
    for end_idx in newline_indices:
        # Extract the sequence, excluding the newline token itself
        seq = data[start_idx:end_idx].tolist()
        if seq:  # Avoid adding empty sequences
            sequences.append(seq)
        start_idx = end_idx + 1
    
    # Add the last sequence if the file doesn't end with a newline token
    if start_idx < len(data):
        seq = data[start_idx:].tolist()
        if seq:
            sequences.append(seq)

    # Sample batch_size sequences
    if not sequences:
        # This can happen if the newline_token is wrong or data is malformed.
        # Provide a helpful error message.
        print(f"Error: No sequences found after splitting by newline_token_id={eos_token_id}.")
        print("Please ensure your .bin files are correctly formatted with this separator token.")
        # Fallback to returning empty tensors to avoid crashing the training loop immediately.
        # The user can then inspect the logs and fix the data.
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    if len(sequences) < batch_size:
        # If we don't have enough sequences, repeat some
        sequences = sequences * ((batch_size // len(sequences)) + 1)
    
    sampled_sequences = random.sample(sequences, min(batch_size, len(sequences)))
    
    # Pad sequences to the same length for batching
    if not sampled_sequences:
        # Fallback to original behavior if no sequences found
        ix = torch.randint(len(data) - max_seq_len, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+max_seq_len]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+max_seq_len]).astype(np.int64)) for i in ix])
        
        # Modify targets: don't compute loss for first 2 positions (predicting tokens 2 and 3)
        # Set first 2 positions to -1 which will be ignored by cross_entropy
        y[:, :2] = -1
        
        # Create attention mask (all 1s for continuous text)
        attention_mask = torch.ones_like(x)
    else:
        max_len = min(max([len(seq) for seq in sampled_sequences]), max_seq_len)
        
        x_batch = []
        y_batch = []
        attention_mask_batch = []
        
        for seq in sampled_sequences:
            # Truncate sequence if too long
            if len(seq) > max_len:
                seq = seq[:max_len]
            
            # Create input (x) and target (y) sequences
            x_seq = seq[:-1] if len(seq) > 1 else seq + [padding_token_id]  # Input is all but last token
            y_seq = seq[1:] if len(seq) > 1 else [padding_token_id]        # Target is shifted by 1
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask_seq = [1] * len(x_seq)
            
            # Pad to max_len using the proper padding token
            while len(x_seq) < max_len:
                x_seq.append(padding_token_id)  # Use proper padding token
                attention_mask_seq.append(0)    # Mark as padding in attention mask
            while len(y_seq) < max_len:
                y_seq.append(-1)  # Pad targets with -1 (ignored in loss)
            
            # Modify targets: don't compute loss for first 2 positions (predicting tokens 2 and 3)
            # Set first 2 positions to -1 which will be ignored by cross_entropy
            if len(y_seq) > 0:
                y_seq[0] = -1
            if len(y_seq) > 1:
                y_seq[1] = -1
            
            x_batch.append(x_seq[:max_len])
            y_batch.append(y_seq[:max_len])
            attention_mask_batch.append(attention_mask_seq[:max_len])
        
        x = torch.tensor(x_batch, dtype=torch.long)
        y = torch.tensor(y_batch, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_batch, dtype=torch.long)
    
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
        attention_mask = attention_mask.to(device)
    
    return x, y, attention_mask

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
eos_token_id = None
padding_token_id = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    # For path data, determine the End-of-Sequence (EOS) token ID
    if 'stoi' in meta and '\n' in meta['stoi']:
        eos_token_id = meta['stoi']['\n']
        print(f"found EOS token ID = {eos_token_id} (from meta['stoi']['\\n'])")
    elif 'vocab_size' in meta:
        # Per user instruction, assume the EOS token is the last token in the vocabulary.
        eos_token_id = meta['vocab_size'] - 1
        print(f"using EOS token ID = {eos_token_id} (last token in vocab)")
    
    # Get padding token ID
    if 'padding_token_id' in meta:
        padding_token_id = meta['padding_token_id']
        print(f"found padding token ID = {padding_token_id} (from meta['padding_token_id'])")
    elif 'stoi' in meta and '<PAD>' in meta['stoi']:
        padding_token_id = meta['stoi']['<PAD>']
        print(f"found padding token ID = {padding_token_id} (from meta['stoi']['<PAD>'])")
    else:
        # Fallback: assume padding token is the second-to-last token (before EOS)
        padding_token_id = meta['vocab_size'] - 2
        print(f"using padding token ID = {padding_token_id} (fallback: vocab_size - 2)")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, max_seq_len=max_seq_len,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'max_seq_len', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'max_seq_len', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# No need to crop model since we use sinusoidal embeddings
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, attention_mask = get_batch(split)
            with ctx:
                logits, loss = model(input_ids=X, targets=Y, attention_mask=attention_mask)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Import the new evaluation module
try:
    from evaluate_maze_nav import evaluate_maze_model
    MAZE_EVAL_AVAILABLE = True
except ImportError:
    MAZE_EVAL_AVAILABLE = False
    print("Warning: Maze evaluation module not available. Skipping path generation evaluation.")

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, attention_mask = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(input_ids=X, targets=Y, attention_mask=attention_mask)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, attention_mask = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# Run comprehensive maze navigation evaluation at the end of training
if master_process and MAZE_EVAL_AVAILABLE and os.path.exists(meta_path):
    print("\n" + "="*60)
    print("TRAINING COMPLETED - Running Final Maze Navigation Evaluation")
    print("="*60)
    try:
        final_maze_results = evaluate_maze_model(
            model=raw_model, 
            data_dir=data_dir, 
            device=device, 
            ctx=ctx,
            max_sequences=2000  # Use more sequences for final comprehensive evaluation
        )
        
        # Log final results to wandb if enabled
        if wandb_log:
            final_log_dict = {
                "final_eval/next_step_accuracy": final_maze_results.get('next_step_accuracy', 0.0),
                "final_eval/exact_match_rate": final_maze_results.get('exact_match_rate', 0.0),
                "final_eval/valid_completion_rate": final_maze_results.get('valid_completion_rate', 0.0),
                "final_eval/path_validity_rate": final_maze_results.get('path_validity_rate', 0.0),
                "final_eval/total_attempts": final_maze_results.get('total_attempts', 0),
                "final_eval/iter_num": iter_num
            }
            wandb.log(final_log_dict)
            
        print("\nFinal maze navigation evaluation completed!")
        print("="*60)
        
    except Exception as e:
        print(f"Error in final maze evaluation: {e}")

if ddp:
    destroy_process_group()
