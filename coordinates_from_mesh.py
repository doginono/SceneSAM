from plyfile import PlyData, PlyElement


def read_ply(filename):
    # Load the PLY file
    plydata = PlyData.read(filename)

    # Extract vertex data
    vertex = plydata["vertex"]
    x = vertex["x"]
    y = vertex["y"]
    z = vertex["z"]

    return x, y, z


def find_bounds(x, y, z):
    # Compute min and max for each dimension
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    z_min, z_max = min(z), max(z)

    return [
        [x_min, x_max],
        [y_min, y_max],
        [z_min, z_max],
    ]


def main():
    filename = "/home/rozenberszki/scene0423_02/scene0423_02_vh_clean_2.ply"  # Path to your PLY file
    x, y, z = read_ply(filename)
    bounds = find_bounds(x, y, z)

    print("Bounds of the mesh:")
    print(f"X-axis: min = {bounds[0][0]}, max = {bounds[0][1]}")
    print(f"Y-axis: min = {bounds[1][0]}, max = {bounds[1][1]}")
    print(f"Z-axis: min = {bounds[2][0]}, max = {bounds[2][1]}")
    print(f"Overall: {bounds}")


if __name__ == "__main__":
    main()
