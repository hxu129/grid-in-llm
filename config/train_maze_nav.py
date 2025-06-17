# Configuration for training GPT on maze navigation data
# Optimized for path learning task

# I/O
out_dir = 'out-maze-nav'
eval_interval = 250
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = True

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