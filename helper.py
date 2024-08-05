import numpy as np
import os
import cv2
import numpy as np

def divide_into_regions(mask, regions=(2, 2)):
    height, width = mask.shape
    region_height = height // regions[0]
    region_width = width // regions[1]
    
    regions_dict = {}
    for i in range(regions[0]):
        for j in range(regions[1]):
            region_key = f"region_{i}_{j}"
            regions_dict[region_key] = mask[
                i * region_height:(i + 1) * region_height,
                j * region_width:(j + 1) * region_width
            ]
    return regions_dict

def calculate_coverage(mask):
    total_area = mask.size
    terrain_area = np.count_nonzero(mask)
    coverage = (terrain_area / total_area) * 100
    return coverage

def calculate_centroid(mask):
    moments = cv2.moments(mask)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = mask.shape[1] // 2, mask.shape[0] // 2
    return cx, cy

def relative_position(centroid1, centroid2):
    dx = centroid2[0] - centroid1[0]
    dy = centroid2[1] - centroid1[1]
    
    position = ''
    if dy < 0:
        position += 'north'
    elif dy > 0:
        position += 'south'
    
    if dx > 0:
        position += ' east'
    elif dx < 0:
        position += ' west'
    
    return position.strip()
