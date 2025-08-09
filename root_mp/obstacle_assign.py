import random
import numpy as np
from typing import Union, List

def create_dynamic_grid(static_grid: Union[List[List[int]], np.ndarray]) -> np.ndarray:
    """
    Convert a static grid (0=path, 1=wall) to dynamic by adding obstacles.
    Works with both Python lists and NumPy arrays.
    
    Args:
        static_grid: 2D array/list representing the map (0=path, 1=wall)
    
    Returns:
        Dynamic grid with obstacles as a NumPy array
    """
    # Convert to NumPy array if it isn't already
    grid = np.array(static_grid, copy=True)
    
    # Find all path cells (0s) where we can place obstacles
    path_cells = list(zip(*np.where(grid == 0)))
    
    # Randomly shuffle the available positions
    random.shuffle(path_cells)
    
    # Define obstacle types and counts
    obstacles = [
        (2, 1),  # 1 BLOCK_A (value 2)
        (3, 1),  # 1 BLOCK_B (value 3)
        (4, 1),  # 1 TRAFFIC_LIGHT_RED (value 4)
        (5, 1),  # 1 TRAFFIC_LIGHT_GREEN (value 5)
        (6, 1),  # 1 TRAFFIC_LOW (value 6)
        (7, 1),  # 1 TRAFFIC_MED (value 7)
        (8, 1)   # 1 TRAFFIC_HIGH (value 8)
    ]
    
    # Place obstacles if there's enough space
    for obstacle_value, count in obstacles:
        for _ in range(count):
            if not path_cells:
                print("Warning: Not enough space for all obstacles")
                break
            i, j = path_cells.pop()
            grid[i, j] = obstacle_value
    
    return grid

# Example usage with both list and NumPy array inputs
if __name__ == "__main__":
    # As Python list
    list_grid = [
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0]
    ]
    
    # As NumPy array
    numpy_grid = np.array([
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0]
    ])
    
    print("List input:")
    print(create_dynamic_grid(list_grid))
    
    print("\nNumPy array input:")
    print(create_dynamic_grid(numpy_grid))