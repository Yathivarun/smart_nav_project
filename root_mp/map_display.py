import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

def visualize_grid(grid: np.ndarray):
    """
    Visualize the grid with different colors for each element type without showing cell values.
    
    Args:
        grid: 2D NumPy array representing the map with:
              - 0: Path
              - 1: Wall
              - 2: BLOCK_A
              - 3: BLOCK_B
              - 4: TRAFFIC_LIGHT_RED
              - 5: TRAFFIC_LIGHT_GREEN
              - 6: TRAFFIC_LOW
              - 7: TRAFFIC_MED
              - 8: TRAFFIC_HIGH
    """
    # Create a color map
    cmap = mcolors.ListedColormap([
        'white',       # 0: Path (white)
        'black',       # 1: Wall (black)
        'orange',      # 2: BLOCK_A (orange)
        'red',         # 3: BLOCK_B (red)
        'darkred',     # 4: TRAFFIC_LIGHT_RED (dark red)
        'limegreen',   # 5: TRAFFIC_LIGHT_GREEN (green)
        'lightblue',   # 6: TRAFFIC_LOW (light blue)
        'dodgerblue',  # 7: TRAFFIC_MED (medium blue)
        'darkblue'     # 8: TRAFFIC_HIGH (dark blue)
    ])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the grid
    img = ax.imshow(grid, cmap=cmap, vmin=0, vmax=8)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    ax.tick_params(which="both", bottom=False, left=False, 
                   labelbottom=False, labelleft=False)
    
    # Create legend
    legend_elements = [
        Patch(facecolor='white', edgecolor='gray', label='Path'),
        Patch(facecolor='black', edgecolor='gray', label='Wall'),
        Patch(facecolor='orange', edgecolor='gray', label='Block A'),
        Patch(facecolor='red', edgecolor='gray', label='Block B'),
        Patch(facecolor='darkred', edgecolor='gray', label='Traffic Light (Red)'),
        Patch(facecolor='limegreen', edgecolor='gray', label='Traffic Light (Green)'),
        Patch(facecolor='lightblue', edgecolor='gray', label='Traffic Low'),
        Patch(facecolor='dodgerblue', edgecolor='gray', label='Traffic Medium'),
        Patch(facecolor='darkblue', edgecolor='gray', label='Traffic High')
    ]
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
              loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example grid with all elements
    example_grid = np.array([
        [0, 1, 0, 2, 0],
        [3, 0, 1, 0, 4],
        [0, 1, 1, 0, 5],
        [6, 0, 0, 7, 1],
        [1, 8, 0, 1, 0]
    ])
    
    visualize_grid(example_grid)