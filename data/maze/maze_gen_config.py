maze_size = 30
max_pairs = 10000000
train_ratio = 0.8
seed = 2025
output_dir = "maze_nav_data"
algorithm = "serpentine" # "wilson", "dfs", "kruskal", or "serpentine"

# Free world mode configuration
free_world_mode = False  # Set to True to generate paths in free world (no walls)
num_random_paths = (maze_size ** 2) ** 2 * 10000  # Number of random paths to generate in free world mode
free_world_output_dir = "path_int_data"  # Output directory for free world paths