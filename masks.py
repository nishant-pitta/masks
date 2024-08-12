import os
import cv2
import numpy as np
from helper import divide_into_regions, calculate_coverage, calculate_centroid, relative_position
import random

# define paths
base_path = '/Users/srinivas/Desktop/ai4mars-dataset-merged-0.1'
labels_path = os.path.join(base_path, 'msl', 'labels', 'train')  # Change to 'test' for test set
images_path = os.path.join(base_path, 'msl', 'images')
masks_path = os.path.join(base_path, 'masks')
overlay_output_path = os.path.join(base_path, 'overlays')
os.makedirs(overlay_output_path, exist_ok=True)


# helper function to load image
def load_image(image_path):
    print(f"Loading image: {image_path}")
    return cv2.imread(image_path)

# helper function to create terrain-specific masks
def create_terrain_masks(image_shape, mask_path):
    print(f"Loading mask: {mask_path}")
    mask = cv2.imread(mask_path)
    
    # Initialize an empty dictionary for terrain masks
    terrain_masks = {
        'soil': np.zeros(image_shape[:2], dtype=np.uint8),
        'bedrock': np.zeros(image_shape[:2], dtype=np.uint8),
        'sand': np.zeros(image_shape[:2], dtype=np.uint8),
        'big_rock': np.zeros(image_shape[:2], dtype=np.uint8)
    }

    # Define the RGB values for each terrain type
    terrain_labels = {
        'soil': (0, 0, 0),
        'bedrock': (1, 1, 1),
        'sand': (2, 2, 2),
        'big_rock': (3, 3, 3)
    }
    
    # Convert terrain labels to separate binary masks
    for terrain, color in terrain_labels.items():
        terrain_masks[terrain][(mask == color).all(axis=2)] = 1

    print(f"Masks created for: {mask_path}")
    return terrain_masks

# Helper function to load terrain-specific masks
def load_terrain_masks(masks_path, image_filename):
    terrain_masks = {}
    terrains = ['soil', 'bedrock', 'sand', 'big_rock']
    
    for terrain in terrains:
        mask_filename = f"{os.path.splitext(image_filename)[0]}_{terrain}_mask.png"
        mask_path = os.path.join(masks_path, mask_filename)
        if os.path.exists(mask_path):
            terrain_masks[terrain] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            terrain_masks[terrain] = None
    
    return terrain_masks


# Helper function to create terrain-specific overlays
def create_terrain_overlay(image, terrain_masks):
    overlay = image.copy()
    # Define colors for each terrain type
    terrain_colors = {
        'soil': (0, 255, 0),      # Green
        'bedrock': (255, 0, 0),   # Blue
        'sand': (0, 0, 255),      # Red
        'big_rock': (255, 255, 0) # Cyan
    }
    
    # Apply each terrain mask with its specific color
    for terrain, mask in terrain_masks.items():
        if mask is not None:
            overlay[mask == 255] = terrain_colors[terrain]

    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# # Main processing loop
# print(f"Processing labels in: {labels_path}")

# for label_file in os.listdir(labels_path):
#     label_file_path = os.path.join(labels_path, label_file)
#     print(f"Processing file: {label_file_path}")
    
#     image_filename = label_file.replace('.png', '.JPG')
#     print(f"Expected image filename: {image_filename}")
#     image_path = None

#     # Check in all image subdirectories
#     for img_subdir in os.listdir(images_path):
#         potential_image_path = os.path.join(images_path, img_subdir, image_filename)
#         if os.path.exists(potential_image_path):
#             image_path = potential_image_path
#             print(f"Image found: {image_path}")
#             break
    
#     image = load_image(image_path)
#     print(f"Loaded image: {image_path}")
    
#     # Create terrain-specific masks
#     terrain_masks = create_terrain_masks(image.shape, label_file_path)
    
#     # Save each terrain mask with a different filename
#     mask_output_path = os.path.join(base_path, 'masks')
#     os.makedirs(mask_output_path, exist_ok=True)
#     for terrain, mask in terrain_masks.items():
#         mask_filename = os.path.join(mask_output_path, f"{os.path.splitext(image_filename)[0]}_{terrain}_mask.png")
#         cv2.imwrite(mask_filename, mask * 255)
#         print(f"Processed and saved {terrain} mask for {image_filename}")
    
#     # overlay = create_terrain_overlay(image, terrain_masks)
#     # overlay_output_path = os.path.join(base_path, 'overlays')
#     # os.makedirs(overlay_output_path, exist_ok=True)
#     # overlay_filename = os.path.join(overlay_output_path, f"{os.path.splitext(image_filename)[0]}_overlay.png")
#     # cv2.imwrite(overlay_filename, overlay)
#     # print(f"Processed and saved overlay for {image_filename}")

# print("Script completed.")


# Main processing loop
print(f"Processing overlays in: {masks_path}")

# Get a list of all image files for which masks have been created
image_files = [f for f in os.listdir(masks_path) if f.endswith('_soil_mask.png')]
image_files = [f.replace('_soil_mask.png', '.JPG') for f in image_files]

for image_filename in image_files:
    print(f"Processing file: {image_filename}")
    
    image_path = None

    # Check in all image subdirectories
    for img_subdir in os.listdir(images_path):
        potential_image_path = os.path.join(images_path, img_subdir, image_filename)
        if os.path.exists(potential_image_path):
            image_path = potential_image_path
            print(f"Image found: {image_path}")
            break
    
    if image_path is None:
        print(f"Image {image_filename} not found in any subdirectory.")
        continue

    image = load_image(image_path)
    if image is None:
        print(f"Failed to load image {image_path}.")
        continue
    
    # Load terrain-specific masks
    terrain_masks = load_terrain_masks(masks_path, image_filename)
    
    # Ensure at least one mask is loaded before creating an overlay
    if all(mask is None for mask in terrain_masks.values()):
        print(f"No masks found for {image_filename}. Skipping overlay creation.")
        continue
    
    # Create and save overlay
    overlay = create_terrain_overlay(image, terrain_masks)
    overlay_output_path = os.path.join(base_path, 'overlays')
    os.makedirs(overlay_output_path, exist_ok=True)
    overlay_filename = os.path.join(overlay_output_path, f"{os.path.splitext(image_filename)[0]}_overlay.png")
    cv2.imwrite(overlay_filename, overlay)
    print(f"Processed and saved overlay for {image_filename}")

print("Script completed.")
