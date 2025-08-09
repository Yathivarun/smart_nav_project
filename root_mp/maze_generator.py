import numpy as np
import matplotlib.pyplot as plt
import random

class MazeGenerator:
    def __init__(self, size, start_pos=(0, 1), end_pos=None):
        """
        Initialize the maze generator.
        
        Args:
            size: Size of the maze (will create a maze of size*size paths)
            start_pos: Tuple (row, col) for maze entrance (default (0,1))
            end_pos: Tuple (row, col) for maze exit (default bottom-right)
        """
        self.size = size
        # Maze is initialized as a grid of walls (1)
        # Actual maze will be 2*size+1 to include walls
        self.maze_size = 2 * size + 1
        self.maze = np.ones((self.maze_size, self.maze_size), dtype=int)
        self.start_pos = start_pos
        self.end_pos = end_pos if end_pos is not None else (self.maze_size-1, self.maze_size-2)
        
        # Validate positions
        self._validate_positions()

    def _validate_positions(self):
        """Validate that start and end positions are on maze borders and valid"""
        for pos, name in [(self.start_pos, "start"), (self.end_pos, "end")]:
            x, y = pos
            if not (0 <= x < self.maze_size and 0 <= y < self.maze_size):
                raise ValueError(f"{name} position {pos} is out of bounds")
            if not (x == 0 or x == self.maze_size-1 or y == 0 or y == self.maze_size-1):
                raise ValueError(f"{name} position {pos} must be on maze border")

    def generate_maze(self):
        """Generate a random maze using depth-first search algorithm"""
        # Start with the top-left cell (1,1 in the expanded grid)
        stack = [(1, 1)]
        self.maze[1, 1] = 0  # Mark as path
        
        # Directions: up, right, down, left
        directions = [(-2, 0), (0, 2), (2, 0), (0, -2)]
        
        while stack:
            current = stack[-1]
            x, y = current
            
            # Find all unvisited neighbors
            neighbors = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (1 <= nx < self.maze_size - 1 and 
                    1 <= ny < self.maze_size - 1 and 
                    self.maze[nx, ny] == 1):
                    neighbors.append((nx, ny, (x + dx//2, y + dy//2)))  # Also store the wall to break
            
            if neighbors:
                # Choose a random neighbor to visit
                nx, ny, (wx, wy) = random.choice(neighbors)
                self.maze[nx, ny] = 0  # Mark new cell as path
                self.maze[wx, wy] = 0   # Break the wall
                stack.append((nx, ny))
            else:
                # Backtrack if no unvisited neighbors
                stack.pop()
        
        # Set entrance and exit
        self.maze[self.start_pos] = 0
        self.maze[self.end_pos] = 0
        
        # Ensure (0,0) and (-1,-1) are always 0 (path)
        self.maze[0, 0] = 0
        self.maze[-1, -1] = 0
    
    def plot_maze(self):
        """Plot the maze using matplotlib"""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.maze, cmap='binary', interpolation='none')
        
        # Mark start and end positions
        plt.scatter(self.start_pos[1], self.start_pos[0], color='green', s=100, marker='o')
        plt.scatter(self.end_pos[1], self.end_pos[0], color='red', s=100, marker='o')
        
        plt.xticks([]), plt.yticks([])
        plt.title(f'Maze {self.size}x{self.size} (Paths: 0, Walls: 1)')
        plt.show()
    
    def get_binary_maze(self):
        """Return the binary maze representation as numpy array"""
        # Ensure (0,0) and (-1,-1) are 0 before returning
        maze_copy = self.maze.copy()
        maze_copy[0, 0] = 0
        maze_copy[-1, -1] = 0
        return maze_copy

    def print_maze(self):
        """Print the maze array with proper formatting"""
        print("\nBinary Maze Representation (0=path, 1=wall):")
        print("[")
        for row in self.maze:
            print(" [" + ", ".join(map(str, row)) + "],")
        print("]")


# Example usage function
def generate_maze(size, start_pos=(0, 1), end_pos=None, plot=True):
    """
    Generate and optionally plot a maze.
    
    Args:
        size: Size of the maze (n x n paths)
        start_pos: Starting position (default (0,1))
        end_pos: Exit position (default opposite corner)
        plot: Whether to display the maze plot
        
    Returns:
        numpy array representing the binary maze with (0,0) and (-1,-1) as 0
    """
    generator = MazeGenerator(size, start_pos, end_pos)
    generator.generate_maze()
    
    if plot:
        generator.plot_maze()
    
    return generator.get_binary_maze()


if __name__ == "__main__":
    # Example usage
    maze = generate_maze(10)  # 10x10 maze with default positions
    print(f"Maze shape: {maze.shape}")
    print(f"Value at (0,0): {maze[0,0]}")  # Should be 0
    print(f"Value at (-1,-1): {maze[-1,-1]}")  # Should be 0