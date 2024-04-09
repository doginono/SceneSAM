import yaml
import os

# Define the grid
sample_pixel_farther_values = [10, 25, 40]
normalize_point_number_values = [10, 25, 40]
depthCondition = [0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0,2.0]
# Load the original YAML file
with open('/home/rozenberszki/D_Project/wsnsl/configs/Scannet++/56a0ec536c.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Iterate over the grid
for sample_pixel_farther in sample_pixel_farther_values:
    for normalize_point_number in normalize_point_number_values:
        # Update the values in the config
        config['segmenter']['samplePixelFarther'] = sample_pixel_farther
        config['segmenter']['normalizePointNumber'] = normalize_point_number
        config['segmenter']['depthCondition'] = depthCondition
        # Write the updated config to a new YAML file
        filename = f'/home/rozenberszki/D_Project/wsnsl/configs/Scannet++/56a0ec536c_{sample_pixel_farther}_{normalize_point_number}_{depthCondition}.yaml'
        with open(filename, 'w') as file:
            yaml.safe_dump(config, file)