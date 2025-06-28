# Grid cells in LLM

This project aims to train Large Language Models (LLMs) to perform spatial navigation tasks and to investigate their internal representations of position, drawing inspiration from the concept of grid cells in neuroscience.

**Not that this project has been archived for special reasons since June, 2025.**

## Repository Structure

Here is an overview of the key files and directories in this project:

- `train.py`: The main script for training the GPT-like model.
- `evaluate_maze_nav.py`: Script for evaluating a trained model on maze navigation tasks.
- `model.py`: Contains the GPT model definition.
- `config/`: Directory for training and evaluation configurations.
- `data/`: Intended to store datasets. It includes scripts for maze data generation under `data/maze/`.
- `ffn_analysis/`: Contains scripts and outputs for analyzing the model's feed-forward networks to understand its spatial representation.
- `out-maze-nav/`: Default output directory for trained models.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/grid-in-llm.git
    cd grid-in-llm
    ```

2.  **Create a virtual environment and install dependencies:**

    It is recommended to use a virtual environment (e.g., venv or conda).
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
    Install the required packages. A `requirements.txt` file is not provided, but the core dependencies are:
    ```bash
    pip install torch numpy matplotlib
    ```
    You might need to install other packages depending on your specific setup.

## Usage

### 1. Data Preparation

The training script expects data to be in the `data/{dataset_name}` directory, preprocessed into binary files (`.bin`).

For maze navigation tasks, you can generate your own dataset using the scripts in `data/maze/`. For example, `data/maze/maze_generator.py` can be used to create maze datasets. After generation, ensure the data is converted to the `.bin` format expected by the trainer. A `meta.pkl` file containing vocabulary information is also required in the dataset directory.

### 2. Training

The model is trained using `train.py`. Configurations are handled by `configurator.py` and can be specified in files within the `config/` directory or as command-line arguments.

For example, to train a model for maze navigation, you can use a configuration like `config/train_maze_nav.py`.

**To run on a single GPU:**
```bash
python train.py --config=config/train_maze_nav.py
```

**To run with Distributed Data Parallel (DDP) on 4 GPUs on a single node:**
```bash
torchrun --standalone --nproc_per_node=4 train.py --config=config/train_maze_nav.py
```
The training script provides more examples for multi-node training in its docstring. Checkpoints and logs will be saved to the directory specified by the `out_dir` variable in the configuration (e.g., `out-maze-nav`).

### 3. Evaluation

To evaluate a trained model on the maze navigation task, use the `evaluate_maze_nav.py` script. It computes various metrics such as next-step prediction accuracy and path generation validity.

```bash
python evaluate_maze_nav.py --model_path=out-maze-nav --data_dir=data/maze_10x10
```
Replace `out-maze-nav` with the path to your trained model and `data/maze_10x10` with the path to your validation dataset.

### 4. FFN Analysis

To analyze the internal representations of the model, you can run the FFN analysis pipeline. This will collect FFN activations, generate heatmaps, and find representative neurons for each position.

```bash
python ffn_analysis/run_ffn_analysis.py --model-path=out-maze-nav --grid-size=10 --task=maze
```
The analysis results, including visualizations, will be saved in a directory named `grid_{grid_size}x{grid_size}_{task}` (e.g., `grid_10x10_maze`).

## Visualizations

The FFN analysis pipeline generates several types of visualizations that help in understanding the model's behavior:

-   **Layer Heatmaps**: These show the activation patterns of all neurons in a layer across all grid positions. They are saved in the `neuron_heatmaps` subdirectory of the analysis output folder.
-   **Representative Neuron Heatmaps**: For each position in the grid, a heatmap of the most active neuron is generated. This shows which neuron is responsible for representing that specific position. These are saved in the `representative_neurons` subdirectory.

These visualizations provide insights into how the model builds a map of the space it is trained on.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. 