import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Union
import math

def plot_neuron_heatmaps(
    activations: np.ndarray,
    neuron_indices: Optional[Union[int, list]] = None,
    grid_size: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cols: int = 4,
    cmap: str = 'viridis',
    save_path: Optional[str] = None,
    title_prefix: str = "Neuron",
    show_colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> plt.Figure:
    """
    Plot heatmaps for neuron activations on a grid world.
    
    Args:
        activations (np.ndarray): Matrix of shape (n**2, y) where n is grid size and y is number of neurons
        neuron_indices (int or list, optional): Which neurons to plot. If None, plots all neurons.
        grid_size (int, optional): Size of the grid (n). If None, inferred from activations.shape[0]
        figsize (tuple, optional): Figure size. If None, automatically determined.
        cols (int): Number of columns in the subplot grid
        cmap (str): Colormap to use for heatmaps
        save_path (str, optional): Path to save the figure
        title_prefix (str): Prefix for subplot titles
        show_colorbar (bool): Whether to show colorbar for each heatmap
        vmin, vmax (float, optional): Min and max values for color scale. If None, uses data range.
    
    Returns:
        plt.Figure: The matplotlib figure object
    """
    
    # Validate input
    if not isinstance(activations, np.ndarray) or len(activations.shape) != 2:
        raise ValueError("activations must be a 2D numpy array of shape (n**2, y)")
    
    n_positions, n_neurons = activations.shape
    
    # Infer grid size if not provided
    if grid_size is None:
        grid_size = int(np.sqrt(n_positions))
        if grid_size * grid_size != n_positions:
            raise ValueError(f"Cannot infer grid size: {n_positions} is not a perfect square")
    else:
        if grid_size * grid_size != n_positions:
            raise ValueError(f"Grid size {grid_size} doesn't match activations shape: {grid_size}**2 != {n_positions}")
    
    # Determine which neurons to plot
    if neuron_indices is None:
        neuron_indices = list(range(n_neurons))
    elif isinstance(neuron_indices, int):
        neuron_indices = [neuron_indices]
    
    # Validate neuron indices
    for idx in neuron_indices:
        if idx < 0 or idx >= n_neurons:
            raise ValueError(f"Neuron index {idx} is out of range [0, {n_neurons-1}]")
    
    n_plots = len(neuron_indices)
    
    # Calculate subplot layout
    rows = math.ceil(n_plots / cols)
    
    # Set figure size if not provided
    if figsize is None:
        figsize = (cols * 4, rows * 3)
    
    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle different subplot configurations
    if rows == 1 and cols == 1:
        # Single subplot
        axes = [axes]
    elif rows == 1 or cols == 1:
        # Single row or single column - ensure it's a list
        if not hasattr(axes, '__len__'):
            axes = [axes]
        elif not isinstance(axes, list):
            axes = list(axes)
    else:
        # Multiple rows and columns - flatten the array
        axes = axes.flatten()
    
    # Determine global color scale if not provided
    if vmin is None or vmax is None:
        selected_activations = activations[:, neuron_indices]
        if vmin is None:
            vmin = selected_activations.min()
        if vmax is None:
            vmax = selected_activations.max()
    
    # Plot each neuron
    for i, neuron_idx in enumerate(neuron_indices):
        ax = axes[i]
        
        # Reshape activations to grid
        neuron_activations = activations[:, neuron_idx].reshape(grid_size, grid_size)
        
        # Create heatmap
        im = ax.imshow(
            neuron_activations,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin='upper'
        )
        
        # Add colorbar if requested
        if show_colorbar:
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Set title and labels
        ax.set_title(f"{title_prefix} {neuron_idx}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        
        # Add grid lines for better visualization
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which="minor", size=0)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmaps saved to {save_path}")
    
    return fig

def plot_single_neuron_heatmap(
    activations: np.ndarray,
    neuron_idx: int,
    grid_size: Optional[int] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'viridis',
    title: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a heatmap for a single neuron's activations.
    
    Args:
        activations (np.ndarray): Matrix of shape (n**2, y)
        neuron_idx (int): Index of the neuron to plot
        grid_size (int, optional): Size of the grid
        figsize (tuple): Figure size
        cmap (str): Colormap to use
        title (str, optional): Custom title for the plot
        save_path (str, optional): Path to save the figure
    
    Returns:
        plt.Figure: The matplotlib figure object
    """
    
    n_positions, n_neurons = activations.shape
    
    if neuron_idx < 0 or neuron_idx >= n_neurons:
        raise ValueError(f"Neuron index {neuron_idx} is out of range [0, {n_neurons-1}]")
    
    # Infer grid size if not provided
    if grid_size is None:
        grid_size = int(np.sqrt(n_positions))
        if grid_size * grid_size != n_positions:
            raise ValueError(f"Cannot infer grid size: {n_positions} is not a perfect square")
    
    # Reshape activations to grid
    neuron_activations = activations[:, neuron_idx].reshape(grid_size, grid_size)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(
        neuron_activations,
        cmap=cmap,
        origin='upper'
    )
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set title and labels
    if title is None:
        title = f"Neuron {neuron_idx} Activation Heatmap"
    ax.set_title(title)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", size=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    return fig

# Example usage function
def example_usage():
    """
    Example of how to use the neuron heatmap visualization functions.
    """
    # Create example data: 8x8 grid (64 positions) with 10 neurons
    grid_size = 8
    n_positions = grid_size * grid_size
    n_neurons = 10
    
    # Generate some example activation data
    np.random.seed(42)
    activations = np.random.rand(n_positions, n_neurons)
    
    # Add some structured patterns to make it more interesting
    for i in range(n_neurons):
        # Create different patterns for different neurons
        pattern = np.zeros((grid_size, grid_size))
        
        if i % 3 == 0:
            # Diagonal pattern
            np.fill_diagonal(pattern, 1.0)
        elif i % 3 == 1:
            # Center hotspot
            center = grid_size // 2
            pattern[center-1:center+2, center-1:center+2] = 1.0
        else:
            # Edge pattern
            pattern[0, :] = 1.0
            pattern[-1, :] = 1.0
            pattern[:, 0] = 1.0
            pattern[:, -1] = 1.0
        
        # Add to random baseline
        activations[:, i] += pattern.flatten() * 0.5
    
    # Plot all neurons
    fig1 = plot_neuron_heatmaps(
        activations,
        cols=3,
        title_prefix="Example Neuron",
        cmap='plasma'
    )
    plt.show()
    
    # Plot a single neuron
    fig2 = plot_single_neuron_heatmap(
        activations,
        neuron_idx=0,
        title="Example Single Neuron Heatmap"
    )
    plt.show()

if __name__ == "__main__":
    example_usage() 