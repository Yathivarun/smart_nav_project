import pygame
import numpy as np

class MazeEditor:
    def __init__(self, grid_size=50, cell_size=10):
        """
        Initialize the maze editor.
        
        Args:
            grid_size: Number of cells in each dimension (default 50)
            cell_size: Pixel size of each cell (default 10)
        """
        # Initialize pygame
        pygame.init()
        
        # Constants
        self.GRID_SIZE = grid_size
        self.CELL_SIZE = cell_size
        self.WINDOW_SIZE = self.GRID_SIZE * self.CELL_SIZE
        self.COLORS = {
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'GRAY': (200, 200, 200)
        }
        
        # Create the grid (0 = path, 1 = wall)
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        # Create the window
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption("Maze Editor - Click cells to create walls")
    
    def draw_grid(self):
        """Draw the current state of the grid"""
        self.screen.fill(self.COLORS['WHITE'])
        
        # Draw cells
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color = self.COLORS['BLACK'] if self.grid[y][x] == 1 else self.COLORS['WHITE']
                pygame.draw.rect(
                    self.screen, color,
                    (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                )
        # Draw grid lines
        for x in range(0, self.WINDOW_SIZE, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLORS['GRAY'], (x, 0), (x, self.WINDOW_SIZE))
        for y in range(0, self.WINDOW_SIZE, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLORS['GRAY'], (0, y), (self.WINDOW_SIZE, y))
        
        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events and return whether to continue running"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get mouse position and convert to grid coordinates
                x, y = pygame.mouse.get_pos()
                grid_x, grid_y = x // self.CELL_SIZE, y // self.CELL_SIZE
                
                # Toggle the cell state if within bounds
                if 0 <= grid_x < self.GRID_SIZE and 0 <= grid_y < self.GRID_SIZE:
                    self.grid[grid_y][grid_x] = 1 - self.grid[grid_y][grid_x]
        
        return True
    
    def run(self):
        """Run the editor until the window is closed"""
        running = True
        while running:
            running = self.handle_events()
            self.draw_grid()
        
        pygame.quit()
        return self.grid.copy()
    
    def print_grid(self):
        """Print the grid in a readable format"""
        print("Binary representation of your maze:")
        print("[")
        for row in self.grid:
            print("    " + str(row.tolist()) + ",")
        print("]")


def draw_maze(grid_size=50, cell_size=10):
    """
    Launch an interactive maze editor and return the created maze.
    
    Args:
        grid_size: Number of cells in each dimension (default 50)
        cell_size: Pixel size of each cell (default 10)
    
    Returns:
        numpy array representing the maze (0=path, 1=wall)
    """
    editor = MazeEditor(grid_size, cell_size)
    maze = editor.run()
    editor.print_grid()
    return maze


if __name__ == "__main__":
    # Example usage
    created_maze = draw_maze(grid_size=30, cell_size=15)
    print(f"\nMaze shape: {created_maze.shape}")

"""
# Basic usage (returns numpy array)
maze = draw_maze()  # Default 50x50 grid with 10px cells

# Custom size
maze = draw_maze(grid_size=30, cell_size=20)  # 30x30 grid with 20px cells

# Just get the array without printing
editor = MazeEditor(40, 15)
maze = editor.run()  # Returns the numpy array directly
"""