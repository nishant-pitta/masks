import os
import cv2
import numpy as np
from helper import divide_into_regions, calculate_coverage, calculate_centroid, relative_position
# Load mask
def load_mask(mask_path):
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Main function to analyze terrain
def analyze_terrain(mask_path, terrain_type):
    mask = load_mask(mask_path)
    
    # Calculate overall coverage
    coverage = calculate_coverage(mask)
    
    # Divide mask into regions and calculate coverage for each region
    regions = divide_into_regions(mask)
    region_coverages = {key: calculate_coverage(region) for key, region in regions.items()}
    
    # Determine predominant region
    predominant_region = max(region_coverages, key=region_coverages.get)
    
    # Calculate centroid for terrain localization
    centroid = calculate_centroid(mask)
    
    return {
        'terrain_type': terrain_type,
        'coverage': coverage,
        'region_coverages': region_coverages,
        'predominant_region': predominant_region,
        'centroid': centroid
    }

# Analyze two different terrains for relative localization
def analyze_relative_localization(mask_path1, mask_path2):
    mask1 = load_mask(mask_path1)
    mask2 = load_mask(mask_path2)
    
    if mask1 is None or mask2 is None:
        print(f"Error loading one of the masks: {mask_path1}, {mask_path2}")
        return
    
    centroid1 = calculate_centroid(mask1)
    centroid2 = calculate_centroid(mask2)
    
    relative_pos = relative_position(centroid1, centroid2)
    
    return relative_pos

# Example
mask_path_sand = '/Users/srinivas/Downloads/ai4mars-dataset-merged-0.1/masks/NLB_453957460EDR_F0321020NCAM00285M1_mask.png'
mask_path_rock = '/Users/srinivas/Downloads/ai4mars-dataset-merged-0.1/masks/NLB_421444916EDR_F0060000NCAM00364M1_mask.png'

terrain_info_sand = analyze_terrain(mask_path_sand, 'sand')
terrain_info_rock = analyze_terrain(mask_path_rock, 'rock')

print("Terrain Information for Sand:", terrain_info_sand)
print("Terrain Information for Rock:", terrain_info_rock)

relative_pos = analyze_relative_localization(mask_path_sand, mask_path_rock)
print(f"Relative Position of Sand to Rock: {relative_pos}")

# Templates for questions and answers
# Terrain Localization
print(f"The {terrain_info_sand['terrain_type']} is predominantly located in the {terrain_info_sand['predominant_region']} of the image.")

# Relative Localization
print(f"The {terrain_info_sand['terrain_type']} is to the {relative_pos} of the {terrain_info_rock['terrain_type']}.")

# Terrain Coverage
print(f"The {terrain_info_sand['terrain_type']} covers approximately {terrain_info_sand['coverage']:.2f}% of the image area.")
