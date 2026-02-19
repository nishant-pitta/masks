import os
import subprocess
import sys
import cv2
import json
import numpy as np
from helper import divide_into_regions, calculate_coverage, calculate_centroid, relative_position

# define paths
base_path = '/Users/user/Downloads/ai4mars-dataset-merged-0.1'
labels_path = os.path.join(base_path, 'msl', 'labels', 'train')  # Change to 'test' for test set
images_path = os.path.join(base_path, 'msl', 'images')

# helper function to load image
def load_image(image_path):
    print(f"Loading image: {image_path}")
    return cv2.imread(image_path)

# helper function to create masks
def create_mask(image_shape, mask_path):
    print(f"Loading mask: {mask_path}")
    mask = cv2.imread(mask_path)
    
    binary_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # define the RGB values for each terrain type
    terrain_labels = {
        'soil': (0, 0, 0),
        'bedrock': (1, 1, 1),
        'sand': (2, 2, 2),
        'big_rock': (3, 3, 3),
        'null': (255, 255, 255)
    }
    
    # Convert terrain labels to binary mask (only for terrain, excluding rover and null labels)
    for terrain, color in terrain_labels.items():
        if terrain != 'null':
            binary_mask[(mask == color).all(axis=2)] = 1

    print(f"Mask created for: {mask_path}")
    return binary_mask

# helper function to create overlay
def create_overlay(image, binary_mask):
    overlay = image.copy()
    overlay[binary_mask == 1] = (0, 0, 255)  # Red color overlay
    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# main processing loop
print(f"Processing labels in: {labels_path}")
for label_file in os.listdir(labels_path):
    label_file_path = os.path.join(labels_path, label_file)
    print(f"Processing file: {label_file_path}")
    
    image_filename = label_file.replace('.png', '.JPG')
    print(f"Expected image filename: {image_filename}")
    image_path = None

    # check in all image subdirectories
    for img_subdir in os.listdir(images_path):
        potential_image_path = os.path.join(images_path, img_subdir, image_filename)
        if os.path.exists(potential_image_path):
            image_path = potential_image_path
            print(f"Image found: {image_path}")
            break
    
    image = load_image(image_path)
    
    
    print(f"Loaded image: {image_path}")
    
    # create binary mask
    mask = create_mask(image.shape, label_file_path)
    
    # save binary mask
    mask_output_path = os.path.join(base_path, 'masks')
    os.makedirs(mask_output_path, exist_ok=True)
    mask_filename = os.path.join(mask_output_path, f"{os.path.splitext(image_filename)[0]}_mask1.png")
    cv2.imwrite(mask_filename, mask * 255)
    print(f"Processed and saved mask for {image_filename}")
    
    # create and save overlay to check
    overlay = create_overlay(image, mask)
    overlay_output_path = os.path.join(base_path, 'overlays')
    os.makedirs(overlay_output_path, exist_ok=True)
    overlay_filename = os.path.join(overlay_output_path, f"{os.path.splitext(image_filename)[0]}_overlay1.png")
    cv2.imwrite(overlay_filename, overlay)
    print(f"Processed and saved overlay for {image_filename}")

    # terrain analysis
    coverages = {terrain: calculate_coverage(create_mask(image.shape, label_file_path)) for terrain in ['soil', 'bedrock', 'sand', 'big_rock']}
    predominant_terrain = max(coverages, key=coverages.get)
    centroid = calculate_centroid(create_mask(image.shape, label_file_path))

    print(f"Coverage for {label_file}: {coverages}")
    print(f"Predominant terrain for {label_file}: {predominant_terrain}")
    print(f"Centroid for {predominant_terrain}: {centroid}")

print("Script completed.")
