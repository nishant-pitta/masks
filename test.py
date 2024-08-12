import os
import cv2
import numpy as np
from helper import divide_into_regions, calculate_coverage, calculate_centroid, relative_position
from langchain_huggingface import HuggingFaceEndpoint
import os
from getpass import getpass


# Helper function to load mask
def load_mask(mask_path):
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Main function to analyze terrain
def analyze_terrain(mask_dir, image_filename):
    terrains = ['soil', 'bedrock', 'sand', 'big_rock']
    terrain_info = {}

    for terrain in terrains:
        mask_path = os.path.join(mask_dir, f"{os.path.splitext(image_filename)[0]}_{terrain}_mask.png")
        mask = load_mask(mask_path)
        
        if mask is None:
            print(f"Mask for {terrain} not found at {mask_path}. Skipping.")
            continue

        # Calculate overall coverage
        coverage = calculate_coverage(mask)
        
        # Divide mask into regions and calculate coverage for each region
        regions = divide_into_regions(mask)
        region_coverages = {key: calculate_coverage(region) for key, region in regions.items()}
        
        # Determine predominant region
        predominant_region = max(region_coverages, key=region_coverages.get)
        
        # Calculate centroid for terrain localization
        centroid = calculate_centroid(mask)
        
        terrain_info[terrain] = {
            'terrain_type': terrain,
            'coverage': coverage,
            'region_coverages': region_coverages,
            'predominant_region': predominant_region,
            'centroid': centroid
        }

    return terrain_info

# Analyze two different terrains for relative localization
def analyze_relative_localization(mask_dir, image_filename, terrain1, terrain2):
    mask_path1 = os.path.join(mask_dir, f"{os.path.splitext(image_filename)[0]}_{terrain1}_mask.png")
    mask_path2 = os.path.join(mask_dir, f"{os.path.splitext(image_filename)[0]}_{terrain2}_mask.png")

    mask1 = load_mask(mask_path1)
    mask2 = load_mask(mask_path2)
    
    if mask1 is None or mask2 is None:
        print(f"Error loading one of the masks: {mask_path1}, {mask_path2}")
        return
    
    centroid1 = calculate_centroid(mask1)
    centroid2 = calculate_centroid(mask2)
    
    relative_pos = relative_position(centroid1, centroid2)
    
    return relative_pos

# Paths
base_path = '/Users/srinivas/Desktop/ai4mars-dataset-merged-0.1'
mask_dir = os.path.join(base_path, 'masks')
image_filename = 'NLB_461768943EDR_F0401378NCAM00190M1.JPG'

# Analyze and output terrain information
terrain_info = analyze_terrain(mask_dir, image_filename)

for terrain, info in terrain_info.items():
    print(f"Terrain Information for {terrain.capitalize()}: {info}")

relative_pos = analyze_relative_localization(mask_dir, image_filename, 'sand', 'bedrock')
print(f"Relative Position of Sand to Rock: {relative_pos}")

# Terrain Localization
for terrain, info in terrain_info.items():
    print(f"The {info['terrain_type']} is predominantly located in the {info['predominant_region']} of the image.")

# Relative Localization
print(f"The sand is to the {relative_pos} of the rock.")

# Terrain Coverage
for terrain, info in terrain_info.items():
    print(f"The {info['terrain_type']} covers approximately {info['coverage']:.2f}% of the image area.")
