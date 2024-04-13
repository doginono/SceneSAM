from plyfile import PlyData, PlyElement
import numpy as np


def adjust_brightness(colors, factor):
    # Ensure no value falls outside the range [0, 255]
    return np.clip(colors * factor, 0, 255).astype(np.uint8)


def modify_ply_brightness(input_file, output_file, brightness_factor):
    # Load the PLY file
    plydata = PlyData.read(input_file)

    # Check if the vertex properties include color information
    if "red" in plydata["vertex"]._property_lookup:
        # Extract color data
        red = plydata["vertex"]["red"]
        green = plydata["vertex"]["green"]
        blue = plydata["vertex"]["blue"]

        # Adjust the brightness
        plydata["vertex"]["red"] = adjust_brightness(red, brightness_factor)
        plydata["vertex"]["green"] = adjust_brightness(green, brightness_factor)
        plydata["vertex"]["blue"] = adjust_brightness(blue, brightness_factor)

        # Save the modified PLY
        plydata.write(output_file)
        print(f"Modified PLY saved to {output_file}")
    else:
        print("No color data found in PLY file.")


# Example usage
input_ply_file = (
    "/home/rozenberszki/project/replica_dataset/room_0/habitat/mesh_semantic.ply"
)
output_ply_file = (
    "/home/rozenberszki/project/replica_dataset/room_0/habitat/mesh_semantic_0_6.ply"
)
brightness_factor = 0.6  # Increase brightness by 20%

modify_ply_brightness(input_ply_file, output_ply_file, brightness_factor)
