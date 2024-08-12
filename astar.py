import os
import cv2
import numpy as np
import heapq
import random

# Define a function to load the terrain masks
def load_terrain_masks(mask_dir, image_filename):
    terrains = ['soil', 'bedrock', 'sand', 'big_rock']
    terrain_masks = {}
    for terrain in terrains:
        mask_path = os.path.join(mask_dir, f"{os.path.splitext(image_filename)[0]}_{terrain}_mask.png")
        terrain_masks[terrain] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return terrain_masks

# Define the A* pathfinding algorithm
def astar(navigable_map, start, goal):
    h, w = navigable_map.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            return reconstruct_path(came_from, current)
        
        neighbors = get_neighbors(current, h, w)
        for neighbor in neighbors:
            if navigable_map[neighbor[1], neighbor[0]] == 0:  # 0 indicates unsafe terrain
                continue
            
            tentative_g_score = g_score[current] + 1  # Uniform cost for each step
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found

# Heuristic function for A* (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Get the neighboring cells in a grid
def get_neighbors(position, h, w):
    x, y = position
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            neighbors.append((nx, ny))
    return neighbors

# Reconstruct the path from start to goal
def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)  # add the start point
    path.reverse()  # reverse the path to go from start to goal
    return path

# Function to describe the path in natural language
def describe_path(path):
    directions = []
    for i in range(1, len(path)):
        prev = path[i - 1]
        curr = path[i]
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        if dx == 1:
            directions.append("right")
        elif dx == -1:
            directions.append("left")
        elif dy == 1:
            directions.append("down")
        elif dy == -1:
            directions.append("up")
    
    direction_str = ", then ".join(directions)
    return f"The path goes {direction_str}."

# Function to create the terrain overlay
def create_terrain_overlay(image, terrain_masks):
    overlay = image.copy()
    terrain_colors = {
        'soil': (0, 255, 0),      # Green
        'bedrock': (255, 0, 0),   # Blue
        'sand': (0, 0, 255),      # Red
        'big_rock': (255, 255, 0) # Cyan
    }
    
    for terrain, mask in terrain_masks.items():
        if mask is not None:
            overlay[mask == 255] = terrain_colors[terrain]
    
    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# Function to visualize the path on the original image with terrain overlays
def visualize_path_on_image(original_image, path, start, goal, terrain_masks):
    # First, create the terrain overlay
    image_with_overlay = create_terrain_overlay(original_image, terrain_masks)

    # Then, draw the path on the image
    for point in path:
        cv2.circle(image_with_overlay, point, radius=2, color=(255, 255, 255), thickness=-1)  # White path points
    
    # Emphasize the start point
    cv2.circle(image_with_overlay, start, radius=7, color=(0, 255, 0), thickness=-1)  # Larger green circle for start
    
    # Emphasize the goal point
    cv2.circle(image_with_overlay, goal, radius=7, color=(0, 0, 255), thickness=-1)  # Larger red circle for goal
    
    return image_with_overlay

# Main function to find and describe the path
def navigate_image(image_filename, mask_dir, images_dir, avoid_terrain='sand'):
    terrain_masks = load_terrain_masks(mask_dir, image_filename)
    
    # Create navigable map
    navigable_map = np.ones_like(terrain_masks['soil'], dtype=np.uint8)
    for terrain, mask in terrain_masks.items():
        if terrain == avoid_terrain:
            navigable_map[mask == 255] = 0  # Set areas to avoid to 0 (unsafe)
    
    h, w = navigable_map.shape
    start = (w // 2, h - 15)  # Near the bottom middle, adjusted for the border
    
    # Create a border around the image where navigation is not allowed (5 pixels smaller on each side)
    navigable_map[:5, :] = 0  # Top border
    navigable_map[-5:, :] = 0  # Bottom border
    navigable_map[:, :5] = 0  # Left border
    navigable_map[:, -5:] = 0  # Right border
    
    # Ensure the goal point is not on the avoid terrain or within the 5-pixel boundary
    goal = None
    while goal is None or navigable_map[goal[1], goal[0]] == 0:
        goal = (random.randint(5, w-6), random.randint(5, h-6))  # Random end point within bounds
        if navigable_map[goal[1], goal[0]] == 0:
            print(f"goal point {goal} is on avoid terrain or too close to the border")
    
    path = astar(navigable_map, start, goal)
    if path is None:
        print("No path found.")
        return
    
    path_description = describe_path(path)
    print(path_description)

    # Load the original image
    original_image_path = os.path.join(images_dir, image_filename)
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        print(f"Failed to load the original image {original_image_path}.")
        return
    
    # Visualize the path with start and end points emphasized over the original image with terrain overlay
    image_with_path = visualize_path_on_image(original_image, path, start, goal, terrain_masks)
    
    # Display the image with the path
    cv2.imshow('Path Visualization', image_with_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, save the image with the path overlaid
    output_path = os.path.join(images_dir, f"{os.path.splitext(image_filename)[0]}_path_overlay.png")
    cv2.imwrite(output_path, image_with_path)
    print(f"Saved image with path overlay to {output_path}")

# Example usage
mask_dir = '/Users/srinivas/Desktop/ai4mars-dataset-merged-0.1/masks'
images_dir = '/Users/srinivas/Desktop/ai4mars-dataset-merged-0.1/msl/images/edr'

#FILE RENAME TO TEST
image_filename = 'NLB_551700847EDR_F0641194NCAM00354M1.JPG'

navigate_image(image_filename, mask_dir, images_dir, avoid_terrain='sand')

# terrain_colors = {
#         'soil': (0, 255, 0),      # Green
#         'bedrock': (255, 0, 0),   # Blue
#         'sand': (0, 0, 255),      # Red
#         'big_rock': (255, 255, 0) # Cyan
#     }