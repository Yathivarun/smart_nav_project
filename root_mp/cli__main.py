import map_display as md
import city_maps as cm
import q_model as qm
import user_maze as um
import obstacle_assign as oa
import maze_generator as mg


def test_maze(grid, start_pos=(0,0), goal_pos=None, episodes=1000):
    """Universal tester for any maze scenario"""
    print("\n" + "="*50)
    print("Testing New Maze Configuration")
    print("="*50)
    
    # Initialize pathfinder
    pathfinder = qm.QLearningPathfinder(grid, start_pos, goal_pos)
    
    # Pre-check maze validity
    print("\nMaze Analysis:")
    print(f"Start position: {pathfinder.START}")
    print(f"Goal position: {pathfinder.GOAL}")
    print(f"Goal reachable: {pathfinder.is_goal_reachable()}")
    
    # Train the model
    print("\nTraining Progress:")
    success_rate, rewards = pathfinder.train(
        episodes=episodes,
        max_steps_per_episode=500,
        verbose=True
    )
    
    # Get best path
    path, success = pathfinder.get_best_path()
    
    # Results
    print("\nResults:")
    print(f"Training success rate: {success_rate:.1%}")
    print(f"Best path found: {'Yes' if success else 'No'}")
    if success:
        print(f"Path length: {len(path)} steps")
    
    # Visualization
    pathfinder.visualize_path()
    pathfinder.plot_training_progress(rewards)


def check_number(x):
    match x:
        case 1:
            print("Choose 1-6 for a city map ")
            take = input("choose: ")
            take = "city-" + take
            obs = int(input("If you want obstacles press 1 else 2: "))

            if obs == 1:
                md.visualize_grid(oa.create_dynamic_grid(cm.CITY_MAZES[take]))
                print("TRAINING UNDER PROCESS PLEASE WAIT.......")
                test_maze(cm.CITY_MAZES[take])

            else:
                md.visualize_grid(cm.CITY_MAZES[take])
                print("TRAINING UNDER PROCESS PLEASE WAIT.......")
                test_maze(cm.CITY_MAZES[take])

        case 2:
            print("Choose the size of maze (5-40) ")
            take = int(input("choose: "))
            maze = mg.generate_maze(take, plot=False)
            md.visualize_grid(maze)
            print("TRAINING UNDER PROCESS PLEASE WAIT.......")
            test_maze(maze)

        case 3:
            print("Choose map size (10-50)")
            take = int(input("Choose: "))
            editor = um.MazeEditor(take, 15)
            maze = editor.run()
            md.visualize_grid(maze)
            print("TRAINING UNDER PROCESS PLEASE WAIT.......")
            test_maze(maze)

        case _:
            print("Bruh you pressed wrong number....exiting")
            exit(0)

while True:
    print("Smart Navigation Model")
    print("1 - City_maps \n2 - Random_Maze \n3 - Design_yourself")
    choice = int(input("choose:"))
    check_number(choice)