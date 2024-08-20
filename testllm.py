import os
import cv2
import numpy as np
from langchain_huggingface import HuggingFaceEndpoint
import random
from helper import divide_into_regions, calculate_coverage, calculate_centroid, relative_position

## Helper function to load mask
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

# Function to generate and ask the LLM a question based on the template
def ask_llm(template: str, **kwargs):
    prompt = f"[TASK] Generate a single sentence that accurately reflects the data provided. [/TASK] [INFO] {template.format(**kwargs)} [/INFO]"
    response = llm.invoke(prompt)
    return response

# Example usage of the LLM with terrain analysis data
def get_llm_responses(terrain_info, relative_position):
    # Questions and templates
    localization_answer = {}
    coverage_answer = {}

    #used chatgpt to make variations
    localization_template_variations = [
        "The {terrain_type} is predominantly located in the {predominant_region} of the image.",
        "You’ll find the {terrain_type} primarily in the {predominant_region} section of the image.",
        "The {terrain_type} occupies most of the {predominant_region} area in the image.",
        "The {terrain_type} is mainly concentrated in the {predominant_region} part of the image.",
    ]
    
    coverage_template_variations = [
        "The {terrain_type} covers approximately {coverage:.2f}% of the image area.",
        "About {coverage:.2f}% of the image area is taken up by {terrain_type}.",
        "The image has around {coverage:.2f}% of its area covered by {terrain_type}.",
        "Roughly {coverage:.2f}% of the image is occupied by {terrain_type}.",
    ]
    
    relative_template_variations = [
        "The {terrain_type_1} is to the {relative_position} of the {terrain_type_2}.",
        "The {terrain_type_1} is situated {relative_position} of the {terrain_type_2}.",
        "You’ll find the {terrain_type_1} {relative_position} to the {terrain_type_2}.",
        "The {terrain_type_1} lies {relative_position} in relation to the {terrain_type_2}.",
    ]
    
    for terrain, info in terrain_info.items():
        # Select a random template for localization and coverage
        localization_template = random.choice(localization_template_variations)
        coverage_template = random.choice(coverage_template_variations)
        
        localization_answer[terrain] = ask_llm(
            localization_template,
            terrain_type=info['terrain_type'],
            predominant_region=info['predominant_region'],
        )

        coverage_answer[terrain] = ask_llm(
            coverage_template,
            terrain_type=info['terrain_type'],
            coverage=info['coverage'],
        )
        
    relative_location_answer = ask_llm(
        random.choice(relative_template_variations),
        terrain_type_1='sand',
        relative_position=relative_position,
        terrain_type_2='bedrock',
    )
    
    return localization_answer, coverage_answer, relative_location_answer

# Initialize 
HUGGINGFACEHUB_API_TOKEN = "{INSERT KEY}"

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    max_length=64,  # single-sentence output
    temperature=0.3,  #  for deterministic output
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

# Paths
base_path = '/Users/srinivas/Desktop/ai4mars-dataset-merged-0.1'
mask_dir = os.path.join(base_path, 'masks')
image_filename = 'NLB_537224463EDR_F0600180NCAM07813M1.JPG'

# Analyze and output terrain information
terrain_info = analyze_terrain(mask_dir, image_filename)
relative_pos = analyze_relative_localization(mask_dir, image_filename, 'sand', 'bedrock')

# Get LLM responses
localization_answer, coverage_answer, relative_location_answer = get_llm_responses(terrain_info, relative_pos)

# Print LLM responses
for terrain in terrain_info.keys():
    print(localization_answer[terrain])
    print(coverage_answer[terrain])

print(relative_location_answer)
