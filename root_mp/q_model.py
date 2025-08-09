import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class QLearningPathfinder:
    def __init__(self, grid, start_pos=(0, 0), goal_pos=None):
        """
        Initialize Q-learning pathfinder with environment grid.
        
        Args:
            grid: 2D numpy array representing the environment
            start_pos: Tuple (row, col) for starting position
            goal_pos: Tuple (row, col) for goal position (defaults to bottom-right)
        """
        self.grid = np.array(grid)
        self.ROWS, self.COLS = self.grid.shape
        
        # Obstacle codes (customize as needed)
        self.WALL = 1
        self.BLOCK_A = 2
        self.BLOCK_B = 3
        self.TRAFFIC_LIGHT_RED = 4
        self.TRAFFIC_LIGHT_GREEN = 5
        self.TRAFFIC_LOW = 6
        self.TRAFFIC_MED = 7
        self.TRAFFIC_HIGH = 8

        # Set positions
        self.START = start_pos
        self.GOAL = goal_pos if goal_pos is not None else (self.ROWS-1, self.COLS-1)
        
        # Validate environment
        self._validate_environment()
        
        # Initialize Q-learning parameters
        self.actions = [0, 1, 2, 3]  # up, down, left, right
        self.Q = np.zeros((self.ROWS, self.COLS, len(self.actions)))
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def _validate_environment(self):
        """Validate that start and goal positions are valid"""
        if not self.is_valid(*self.START):
            raise ValueError("Start position is invalid (wall or out of bounds)")
        if not self.is_valid(*self.GOAL):
            raise ValueError("Goal position is invalid (wall or out of bounds)")
        if not self.is_goal_reachable():
            print("Warning: Goal may not be reachable from start position")

    def is_valid(self, x, y):
        """Check if position is valid (within bounds and not a wall)"""
        return 0 <= x < self.ROWS and 0 <= y < self.COLS and self.grid[x, y] != self.WALL

    def is_goal_reachable(self):
        """Check if goal is reachable using flood fill algorithm"""
        visited = set()
        queue = [self.START]
        
        while queue:
            current = queue.pop(0)
            if current == self.GOAL:
                return True
            if current in visited:
                continue
                
            visited.add(current)
            x, y = current
            
            # Check all neighboring cells
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if self.is_valid(nx, ny):
                    queue.append((nx, ny))
        
        return False

    def get_reward(self, pos):
        """Calculate reward for moving to a position"""
        cell_value = self.grid[pos]
        
        if pos == self.GOAL:
            return 100  # Large reward for reaching goal
        elif cell_value == self.WALL:
            return -100
        elif cell_value in [self.BLOCK_A, self.BLOCK_B]:
            return -50
        elif cell_value == self.TRAFFIC_LIGHT_RED:
            return -20
        elif cell_value == self.TRAFFIC_HIGH:
            return -5
        elif cell_value == self.TRAFFIC_MED:
            return -2
        elif cell_value == self.TRAFFIC_LOW:
            return -1
        else:
            return -0.1  # Small penalty for movement

    def move(self, pos, action):
        """Move agent considering obstacles and traffic rules"""
        x, y = pos
        new_x, new_y = x, y
        
        if action == 0: new_x = x-1  # Up
        elif action == 1: new_x = x+1  # Down
        elif action == 2: new_y = y-1  # Left
        elif action == 3: new_y = y+1  # Right
        
        if self.is_valid(new_x, new_y):
            # Traffic light behavior
            if self.grid[new_x, new_y] == self.TRAFFIC_LIGHT_RED:
                if random.random() < 0.7:  # 70% chance to stop
                    return (x, y)
            return (new_x, new_y)
        return (x, y)

    def train(self, episodes=2000, max_steps_per_episode=1000, verbose=True):
        """
        Train the Q-learning model.
        
        Returns:
            tuple: (success_rate, episode_rewards)
        """
        success_count = 0
        episode_rewards = []
        
        for episode in range(episodes):
            state = self.START
            total_reward = 0
            visited = set()
            reached_goal = False
            
            for step in range(max_steps_per_episode):
                if state == self.GOAL:
                    reached_goal = True
                    success_count += 1
                    break
                    
                # Epsilon-greedy action selection
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(self.actions)
                else:
                    action = np.argmax(self.Q[state[0], state[1]])
                
                next_state = self.move(state, action)
                reward = self.get_reward(next_state)
                
                # Penalize revisits
                if next_state in visited:
                    reward -= 2
                
                # Q-learning update
                x, y = state
                nx, ny = next_state
                self.Q[x, y, action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[nx, ny]) - self.Q[x, y, action]
                )
                
                state = next_state
                visited.add(state)
                total_reward += reward
            
            episode_rewards.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            
            if verbose and episode % 100 == 0:
                status = "✓" if reached_goal else "✗"
                print(f"Episode {episode:4d} {status} | Reward: {total_reward:7.1f} | ε: {self.epsilon:.3f}")
        
        success_rate = success_count / episodes
        if verbose:
            print(f"\nTraining complete! Success rate: {success_rate:.1%}")
            if success_rate == 0:
                print("Warning: Never reached goal - check maze connectivity")
        
        return success_rate, episode_rewards

    def get_best_path(self, max_steps=1000):
        """
        Extract the best path from Q-table.
        
        Returns:
            tuple: (path, success)
        """
        path = [self.START]
        state = self.START
        
        for _ in range(max_steps):
            if state == self.GOAL:
                return path, True
                
            x, y = state
            action = np.argmax(self.Q[x, y])
            next_state = self.move(state, action)
            
            # Loop detection
            if next_state in path:
                break
                
            path.append(next_state)
            state = next_state
        
        return path, False

    def visualize_path(self, path=None, show_plot=True):
        """Visualize the grid and path with obstacle legend"""
        if path is None:
            path, success = self.get_best_path()
        else:
            success = path[-1] == self.GOAL if len(path) > 0 else False
        
        plt.figure(figsize=(12, 8))
        cmap = plt.cm.get_cmap('viridis', 9)
        
        # Display grid
        plt.imshow(self.grid, cmap=cmap, vmin=0, vmax=8)
        
        # Plot path
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        plt.plot(path_x, path_y, 'r-', linewidth=2, marker='o', markersize=4)
        
        # Mark start and goal
        plt.scatter(self.START[1], self.START[0], color='green', s=200, marker='*', edgecolor='black')
        goal_color = 'yellow' if success else 'red'
        goal_marker = '*' if success else 'X'
        plt.scatter(self.GOAL[1], self.GOAL[0], color=goal_color, s=200, marker=goal_marker, edgecolor='black')
        
        # Create legend
        legend_elements = [
            Patch(facecolor=cmap(0), label='Path'),
            Patch(facecolor=cmap(1), label='Wall'),
            Patch(facecolor=cmap(2), label='Block A'),
            Patch(facecolor=cmap(3), label='Block B'),
            Patch(facecolor=cmap(4), label='Red Light'),
            Patch(facecolor=cmap(5), label='Green Light'),
            Patch(facecolor=cmap(6), label='Low Traffic'),
            Patch(facecolor=cmap(7), label='Med Traffic'),
            Patch(facecolor=cmap(8), label='High Traffic')
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        title = "Q-Learning Path - " + ("Success!" if success else "No Path Found")
        plt.title(title)
        plt.grid(color='white', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        else:
            return plt.gcf()

    def plot_training_progress(self, episode_rewards, show_plot=True):
        """Plot the training rewards over episodes"""
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards)
        plt.title("Training Progress")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        
        if show_plot:
            plt.show()
        else:
            return plt.gcf()


# Example usage
if __name__ == "__main__":
    # Create a sample grid (0 = path, 1 = wall)
    grid = np.array([
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0]
    ])
    
    # Initialize and train
    pathfinder = QLearningPathfinder(grid)
    success_rate, rewards = pathfinder.train(episodes=500)
    
    # Get and visualize path
    path, success = pathfinder.get_best_path()
    print(f"\nPath found: {'Yes' if success else 'No'}")
    print(f"Path length: {len(path)} steps")
    
    pathfinder.visualize_path()
    pathfinder.plot_training_progress(rewards)