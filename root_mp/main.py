import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QSpinBox, QTabWidget, QMessageBox, QLineEdit,
                             QFormLayout, QGroupBox, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pygame
import random

# Import your original modules
import map_display as md
import city_maps as cm
import q_model as qm
import user_maze as um
import obstacle_assign as oa
import maze_generator as mg

# Modern color palette
COLORS = {
    'dark_bg': '#2D2D2D',
    'light_bg': '#3E3E3E',
    'highlight': '#4F9D9D',
    'text': '#E0E0E0',
    'button': '#5C6BC0',
    'button_hover': '#7986CB',
    'success': '#81C784',
    'warning': '#FFB74D',
    'error': '#E57373'
}

class StyledButton(QPushButton):
    """Modern styled button with hover effects"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFont(QFont('Segoe UI', 10))
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['button']};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['button_hover']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['highlight']};
            }}
            QPushButton:disabled {{
                background-color: #757575; 
            }}
        """)

class StyledComboBox(QComboBox):
    """Modern styled combo box"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont('Segoe UI', 9))
        self.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['light_bg']};
                color: {COLORS['text']};
                border: 1px solid #555;
                padding: 3px;
                border-radius: 3px;
                min-width: 100px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
        """)

class StyledSpinBox(QSpinBox):
    """Modern styled spin box"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont('Segoe UI', 9))
        self.setStyleSheet(f"""
            QSpinBox {{
                background-color: {COLORS['light_bg']};
                color: {COLORS['text']};
                border: 1px solid #555;
                padding: 3px;
                border-radius: 3px;
            }}
        """)

class StyledLineEdit(QLineEdit):
    """Modern styled line edit"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont('Segoe UI', 9))
        self.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLORS['light_bg']};
                color: {COLORS['text']};
                border: 1px solid #555;
                padding: 3px;
                border-radius: 3px;
            }}
        """)

class StyledLabel(QLabel):
    """Modern styled label"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setFont(QFont('Segoe UI', 9))
        self.setStyleSheet(f"color: {COLORS['text']};")

class StyledGroupBox(QGroupBox):
    """Modern styled group box"""
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.setFont(QFont('Segoe UI', 10, QFont.Bold))
        self.setStyleSheet(f"""
            QGroupBox {{
                color: {COLORS['highlight']};
                border: 1px solid {COLORS['highlight']};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }}
        """)

class PathfindingThread(QThread):
    """Thread to handle pathfinding computation"""
    update_signal = pyqtSignal(str)
    result_signal = pyqtSignal(object, object, object, object)
    progress_signal = pyqtSignal(int)

    def __init__(self, grid, start_pos, goal_pos, episodes):
        super().__init__()
        self.grid = grid
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.episodes = episodes

    def run(self):
        pathfinder = qm.QLearningPathfinder(self.grid, self.start_pos, self.goal_pos)
        
        # Pre-check
        self.update_signal.emit("Checking maze validity...")
        if not pathfinder.is_goal_reachable():
            self.update_signal.emit("Warning: Goal may not be reachable")
        
        # Train
        self.update_signal.emit(f"Training for {self.episodes} episodes...")
        success_rate, rewards = pathfinder.train(
            episodes=self.episodes,
            max_steps_per_episode=500,
            verbose=False
        )
        
        # Get path
        path, success = pathfinder.get_best_path()
        
        # Emit results
        self.result_signal.emit(pathfinder, success_rate, path, rewards)

class MazeCanvas(FigureCanvas):
    """Modern styled matplotlib canvas for displaying mazes"""
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=COLORS['dark_bg'])
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, facecolor=COLORS['dark_bg'])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Customize colors
        self.ax.tick_params(colors=COLORS['text'])
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
    def plot_maze(self, grid, path=None, success=True, start_pos=(0,0), goal_pos=None):
        self.ax.clear()
        
        if goal_pos is None:
            goal_pos = (grid.shape[0]-1, grid.shape[1]-1)
        
        # Plot maze with obstacle colors
        cmap = plt.cm.get_cmap('viridis', 9)
        self.ax.imshow(grid, cmap=cmap, vmin=0, vmax=8)
        
        # Plot path if exists
        if path is not None:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            self.ax.plot(path_x, path_y, 'r-', linewidth=2, marker='o', markersize=4)
        
        # Mark start and goal
        self.ax.scatter(start_pos[1], start_pos[0], color='green', s=200, 
                       marker='*', edgecolor='black')
        goal_color = 'yellow' if success else 'red'
        goal_marker = '*' if success else 'X'
        self.ax.scatter(goal_pos[1], goal_pos[0], color=goal_color, s=200, 
                       marker=goal_marker, edgecolor='black')
        
        # Add obstacle legend
        obstacle_labels = [
            'Path (0)', 'Wall (1)', 'Block A (2)', 'Block B (3)',
            'Red Light (4)', 'Green Light (5)', 
            'Low Traffic (6)', 'Med Traffic (7)', 'High Traffic (8)'
        ]
        patches = [mpatches.Patch(color=cmap(i), label=obstacle_labels[i]) 
                  for i in range(9)]
        self.ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), 
                      loc='upper left', borderaxespad=0., facecolor=COLORS['dark_bg'], 
                      edgecolor=COLORS['text'], labelcolor=COLORS['text'])
        
        self.draw()

class TrainingPlot(FigureCanvas):
    """Modern styled training progress plot with stats"""
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=COLORS['dark_bg'])
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Create main layout for figure (3:1 height ratio)
        self.grid = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])
        self.ax = self.fig.add_subplot(self.grid[0])
        self.stats_ax = self.fig.add_subplot(self.grid[1])
        self.stats_ax.axis('off')
        
        # Customize colors
        self.ax.tick_params(colors=COLORS['text'])
        self.ax.xaxis.label.set_color(COLORS['text'])
        self.ax.yaxis.label.set_color(COLORS['text'])
        self.ax.title.set_color(COLORS['text'])
        self.ax.spines['bottom'].set_color(COLORS['text'])
        self.ax.spines['left'].set_color(COLORS['text'])
        
        # Initialize stats text
        self.stats_text = ""
        self.stats_display = self.stats_ax.text(0.02, 0.5, "", 
                                              transform=self.stats_ax.transAxes,
                                              color=COLORS['text'],
                                              fontsize=10,
                                              verticalalignment='center')
        
    def plot_progress(self, rewards, stats=None):
        self.ax.clear()
        self.ax.plot(rewards, color=COLORS['highlight'])
        self.ax.set_title("Training Progress", color=COLORS['text'])
        self.ax.set_xlabel("Episode", color=COLORS['text'])
        self.ax.set_ylabel("Total Reward", color=COLORS['text'])
        self.ax.grid(True, color='#555')
        
        if stats:
            self.update_stats(stats)
        
        self.draw()
    
    def update_stats(self, stats):
        """Update the statistics display below the plot"""
        stats_text = (
            f"Success Rate: {stats['success_rate']:.1%} | "
            f"Path Length: {stats['path_length']} | "
            f"Episodes: {stats['episodes']} | "
            f"Final Reward: {stats['final_reward']:.1f} | "
            f"Final ε: {stats['epsilon']:.3f}"
        )
        self.stats_display.set_text(stats_text)
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Navigation Model")
        self.setGeometry(100, 100, 1400, 800)
        
        # Set dark theme
        self.setStyleSheet(f"background-color: {COLORS['dark_bg']};")
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left panel (controls)
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)
        control_layout.setSpacing(15)
        control_panel.setFixedWidth(350)
        control_panel.setStyleSheet(f"""
            background-color: {COLORS['light_bg']};
            border-radius: 5px;
            padding: 10px;
        """)
        
        # Right panel (visualizations)
        vis_panel = QFrame()
        vis_panel.setFrameShape(QFrame.StyledPanel)
        vis_layout = QVBoxLayout(vis_panel)
        vis_layout.setContentsMargins(5, 5, 5, 5)
        vis_panel.setStyleSheet(f"""
            background-color: {COLORS['light_bg']};
            border-radius: 5px;
        """)
        
        # Add panels to main layout
        main_layout.addWidget(control_panel)
        main_layout.addWidget(vis_panel)
        
        # Initialize UI components
        self.init_controls(control_layout)
        self.init_visualization(vis_layout)
        
        # Current state
        self.current_maze = None
        self.current_path = None
        self.pathfinder = None
        self.start_pos = (0, 0)
        self.goal_pos = None
        
    def init_controls(self, layout):
        """Initialize control panel widgets with modern styling"""
        # Title
        title = StyledLabel("Smart Navigation Model")
        title.setFont(QFont('Segoe UI', 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {COLORS['highlight']}; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Maze type selection
        maze_type_group = StyledGroupBox("Maze Configuration")
        maze_type_layout = QVBoxLayout()
        
        self.maze_type = StyledComboBox()
        self.maze_type.addItems(["City Maps", "Random Maze", "Design Your Own"])
        maze_type_layout.addWidget(StyledLabel("Maze Type:"))
        maze_type_layout.addWidget(self.maze_type)
        
        # City map selection
        self.city_map = StyledComboBox()
        self.city_map.addItems([f"City {i}" for i in range(1, 7)])
        maze_type_layout.addWidget(StyledLabel("City Map:"))
        maze_type_layout.addWidget(self.city_map)
        self.city_map.hide()
        
        # Maze size selection
        self.maze_size = StyledSpinBox()
        self.maze_size.setRange(5, 50)
        self.maze_size.setValue(10)
        maze_type_layout.addWidget(StyledLabel("Maze Size:"))
        maze_type_layout.addWidget(self.maze_size)
        
        # Obstacles toggle
        self.obstacles = StyledComboBox()
        self.obstacles.addItems(["No Obstacles", "With Obstacles"])
        maze_type_layout.addWidget(StyledLabel("Obstacles:"))
        maze_type_layout.addWidget(self.obstacles)
        self.obstacles.hide()
        
        maze_type_group.setLayout(maze_type_layout)
        layout.addWidget(maze_type_group)
        
        # Position inputs
        pos_group = StyledGroupBox("Positions")
        pos_layout = QFormLayout()
        pos_layout.setVerticalSpacing(10)
        
        self.start_input = StyledLineEdit("0,0")
        self.goal_input = StyledLineEdit()
        pos_layout.addRow(StyledLabel("Start (row,col):"), self.start_input)
        pos_layout.addRow(StyledLabel("Goal (row,col):"), self.goal_input)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Training configuration
        train_group = StyledGroupBox("Training Configuration")
        train_layout = QFormLayout()
        train_layout.setVerticalSpacing(10)
        
        self.episodes = StyledSpinBox()
        self.episodes.setRange(100, 10000)
        self.episodes.setValue(1000)
        train_layout.addRow(StyledLabel("Training Episodes:"), self.episodes)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # Main buttons (Load and Train)
        button_layout = QHBoxLayout()
        
        self.load_btn = StyledButton("Load Maze")
        self.load_btn.clicked.connect(self.load_maze)
        button_layout.addWidget(self.load_btn)
        
        self.train_btn = StyledButton("Train Model")
        self.train_btn.clicked.connect(self.train_model)
        self.train_btn.setEnabled(False)
        button_layout.addWidget(self.train_btn)
        
        layout.addLayout(button_layout)
        
        # Design Maze button (appears below main buttons)
        self.design_btn = StyledButton("Design Maze")
        self.design_btn.clicked.connect(self.open_designer)
        self.design_btn.hide()
        layout.addWidget(self.design_btn)
        
        # Status label
        self.status = StyledLabel("Ready")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("""
            padding: 8px;
            border-radius: 4px;
            background-color: #555;
            margin-top: 10px;
        """)
        layout.addWidget(self.status)
        
        # Connect signals
        self.maze_type.currentIndexChanged.connect(self.update_controls)
        
    def init_visualization(self, layout):
        """Initialize visualization widgets with modern styling"""
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
            }}
            QTabBar::tab {{
                background: {COLORS['light_bg']};
                color: {COLORS['text']};
                padding: 8px;
                border: none;
                min-width: 100px;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['highlight']};
                color: white;
            }}
        """)
        
        # Maze tab
        self.maze_tab = QWidget()
        self.maze_layout = QVBoxLayout(self.maze_tab)
        self.maze_layout.setContentsMargins(0, 0, 0, 0)
        self.maze_canvas = MazeCanvas(self.maze_tab, width=8, height=8)
        self.maze_layout.addWidget(self.maze_canvas)
        self.tabs.addTab(self.maze_tab, "Maze View")
        
        # Training tab
        self.train_tab = QWidget()
        self.train_layout = QVBoxLayout(self.train_tab)
        self.train_layout.setContentsMargins(0, 0, 0, 0)
        self.train_canvas = TrainingPlot(self.train_tab, width=8, height=5)  # Increased height
        self.train_layout.addWidget(self.train_canvas)
        self.tabs.addTab(self.train_tab, "Training Progress")
        
        layout.addWidget(self.tabs)

    def update_controls(self):
        """Update which controls are visible based on maze type"""
        maze_type = self.maze_type.currentIndex()
        
        self.city_map.setVisible(maze_type == 0)
        self.obstacles.setVisible(maze_type == 0)
        self.maze_size.setVisible(maze_type in [1, 2])
        self.design_btn.setVisible(maze_type == 2)
        
    def parse_position(self, text):
        """Parse (row,col) position input"""
        try:
            row, col = map(int, text.split(','))
            return (row, col)
        except:
            return None
        
    def load_maze(self):
        """Load the selected maze type"""
        # Parse positions
        self.start_pos = self.parse_position(self.start_input.text()) or (0, 0)
        self.goal_pos = self.parse_position(self.goal_input.text())
        
        maze_type = self.maze_type.currentIndex()
        
        try:
            if maze_type == 0:  # City map
                city_num = self.city_map.currentIndex() + 1
                maze = cm.CITY_MAZES[f"city-{city_num}"]
                if self.obstacles.currentIndex() == 1:
                    maze = oa.create_dynamic_grid(maze)
                
            elif maze_type == 1:  # Random maze
                size = self.maze_size.value()
                maze = mg.generate_maze(size, plot=False)
                
            else:  # Designed maze
                size = self.maze_size.value()
                editor = um.MazeEditor(size, 15)
                maze = editor.run()
            
            self.current_maze = maze
            self.current_path = None
            
            # Validate positions
            if not self.is_valid_pos(self.start_pos, maze):
                raise ValueError("Start position is invalid")
            if self.goal_pos and not self.is_valid_pos(self.goal_pos, maze):
                raise ValueError("Goal position is invalid")
            
            # Plot maze with current positions
            self.maze_canvas.plot_maze(
                maze, 
                start_pos=self.start_pos,
                goal_pos=self.goal_pos
            )
            
            self.train_btn.setEnabled(True)
            self.status.setText("Maze loaded successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load maze:\n{str(e)}")
            self.status.setText("Error loading maze")
    
    def is_valid_pos(self, pos, maze):
        """Check if position is valid in the maze"""
        row, col = pos
        return (0 <= row < maze.shape[0] and 
                0 <= col < maze.shape[1] and 
                maze[row, col] != 1)  # Not a wall
    
    def train_model(self):
        """Train the pathfinding model on the current maze"""
        if self.current_maze is None:
            QMessageBox.warning(self, "Warning", "Please load a maze first")
            return
        
        # Disable buttons during training
        self.load_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.design_btn.setEnabled(False)
        self.status.setText("Training started...")
        
        # Create and start worker thread
        self.worker = PathfindingThread(
            self.current_maze,
            start_pos=self.start_pos,
            goal_pos=self.goal_pos,
            episodes=self.episodes.value()
        )
        
        # Connect signals
        self.worker.update_signal.connect(self.update_status)
        self.worker.result_signal.connect(self.training_complete)
        self.worker.finished.connect(self.worker.deleteLater)
        
        # Start thread
        self.worker.start()
    
    def update_status(self, message):
        """Update status message during training"""
        self.status.setText(message)
        QApplication.processEvents()
    
    def training_complete(self, pathfinder, success_rate, path, rewards):
        """Handle training completion"""
        self.pathfinder = pathfinder
        self.current_path = path
        
        # Determine if path reached goal
        if self.goal_pos:
            target = self.goal_pos
        else:
            target = (pathfinder.grid.shape[0]-1, pathfinder.grid.shape[1]-1)
        success = path[-1] == target if path else False
        
        # Update maze visualization
        self.maze_canvas.plot_maze(
            pathfinder.grid, 
            path, 
            success,
            start_pos=self.start_pos,
            goal_pos=self.goal_pos
        )
        
        # Prepare stats for display
        stats = {
            'success_rate': success_rate,
            'path_length': len(path) if path else 0,
            'episodes': self.episodes.value(),
            'final_reward': rewards[-1] if rewards else 0,
            'epsilon': pathfinder.epsilon
        }
        
        # Update training progress plot with stats
        self.train_canvas.plot_progress(rewards, stats)
        
        # Re-enable buttons
        self.load_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.design_btn.setEnabled(True)
        
        # Show results in status
        self.status.setText(
            f"Training complete! Success rate: {success_rate:.1%}. " +
            f"Path found: {'Yes' if success else 'No'}"
        )
        
        if not success:
            QMessageBox.warning(self, "Result", 
                              "No complete path found to the goal")
    
    def open_designer(self):
        """Open the maze designer"""
        size = self.maze_size.value()
        editor = um.MazeEditor(size, 15)
        maze = editor.run()
        
        if maze is not None:
            self.current_maze = maze
            self.current_path = None
            self.maze_canvas.plot_maze(
                maze,
                start_pos=self.start_pos,
                goal_pos=self.goal_pos
            )
            self.train_btn.setEnabled(True)
            self.status.setText("Custom maze loaded successfully")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style and font
    app.setStyle('Fusion')
    font = QFont('Segoe UI', 9)
    app.setFont(font)
    
    # Set palette for dark theme
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS['dark_bg']))
    palette.setColor(QPalette.WindowText, QColor(COLORS['text']))
    palette.setColor(QPalette.Base, QColor(COLORS['light_bg']))
    palette.setColor(QPalette.AlternateBase, QColor(COLORS['dark_bg']))
    palette.setColor(QPalette.ToolTipBase, QColor(COLORS['highlight']))
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, QColor(COLORS['text']))
    palette.setColor(QPalette.Button, QColor(COLORS['button']))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(COLORS['highlight']))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    pygame.init()  # For maze designer
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())