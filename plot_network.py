import matplotlib.pyplot as plt
import numpy as np
import json
import os

def visualize_expansion(history_file, json_file="data/cities.json", start=None, end=None, solution_file="data/astar_solution.txt"):
    # Read the city data from JSON file
    with open(json_file, 'r') as f:
        cities_data = json.load(f)
    
    # Extract coordinates for all cities
    city_coords = []
    for city in cities_data:
        city_coords.append((city["longitude"], city["latitude"]))
    
    # Read the expansion history
    with open(history_file, 'r') as f:
        history = [list(map(int, line.strip().split(','))) for line in f]
    
    # Read solution path if available
    solution_path = []
    if os.path.exists(solution_file):
        with open(solution_file, 'r') as f:
            solution_path = list(map(int, f.read().strip().split(',')))
    
    # Create animation
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot all cities as small gray dots
    all_x = [coord[0] for coord in city_coords]
    all_y = [coord[1] for coord in city_coords]
    ax.scatter(all_x, all_y, c='lightgray', s=10, alpha=0.5)
    
    # Animation function to update the plot for each step
    for step, indices in enumerate(history):
        ax.clear()
        
        # Plot all cities
        ax.scatter(all_x, all_y, c='lightgray', s=10, alpha=0.5)
        
        # Plot visited cities in this step
        visited_x = [city_coords[i-1][0] for i in indices]  # Adjust for 0-indexing if needed
        visited_y = [city_coords[i-1][1] for i in indices]
        ax.scatter(visited_x, visited_y, c='red', s=50)
        
        # Highlight start and end cities if provided
        if start is not None:
            start_x, start_y = city_coords[start-1]
            ax.scatter(start_x, start_y, c='green', s=120, marker='*', edgecolor='black', zorder=5)
        
        if end is not None:
            end_x, end_y = city_coords[end-1]
            ax.scatter(end_x, end_y, c='blue', s=120, marker='*', edgecolor='black', zorder=5)
        
        # Annotate the step number
        ax.set_title(f"A* Search Expansion - Step {step+1} ({len(indices)} cities)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        plt.tight_layout()
        plt.pause(0.01)  # Pause to show the step
    
    # After showing all expansion steps, display the solution path if available
    if solution_path:
        # Clear the last plot and redraw
        ax.clear()
        ax.scatter(all_x, all_y, c='lightgray', s=10, alpha=0.5)
        
        # Plot visited cities from the last step
        last_indices = history[-1] if history else []
        visited_x = [city_coords[i-1][0] for i in last_indices]
        visited_y = [city_coords[i-1][1] for i in last_indices]
        ax.scatter(visited_x, visited_y, c='red', s=50)
        
        # Plot the solution path
        path_x = [city_coords[i-1][0] for i in solution_path]
        path_y = [city_coords[i-1][1] for i in solution_path]
        
        # Draw the path as a blue line connecting the points
        ax.plot(path_x, path_y, c='blue', linewidth=3, zorder=4)
        
        # Add blue dots for the path cities (larger than visited cities)
        ax.scatter(path_x, path_y, c='blue', s=80, zorder=4, edgecolor='white')
        
        # Highlight start and end cities
        if start is not None:
            start_x, start_y = city_coords[start-1]
            ax.scatter(start_x, start_y, c='green', s=120, marker='*', edgecolor='black', zorder=5)
        
        if end is not None:
            end_x, end_y = city_coords[end-1]
            ax.scatter(end_x, end_y, c='blue', s=120, marker='*', edgecolor='black', zorder=5)
        
        ax.set_title(f"A* Search Solution Path - {len(solution_path)} cities")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.pause(2)  # Pause longer on the solution
    
    plt.show()

# Call the function
visualize_expansion("data/bi_astar_expansion.txt", start=1, end=2, solution_file="data/bi_astar_solution.txt")