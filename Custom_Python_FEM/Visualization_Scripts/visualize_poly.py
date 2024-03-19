import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def read_poly_file(file_path):
    """
    Reads a .poly file and returns the coordinates.
    Assumes the first line is a header and subsequent lines contain vertex data.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        coordinates = [list(map(float, line.split()[1:])) for line in lines[1:] if len(line.split()) == 4]
    return np.array(coordinates)

def plot_polygons(coordinates):
    """
    Plots the vertices in a 3D space.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], marker='o', color='blue')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('3D Polygon Visualization')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize .poly files in 3D')
    parser.add_argument('file_path', type=str, help='Path to the .poly file')
    args = parser.parse_args()

    coordinates = read_poly_file(args.file_path)
    plot_polygons(coordinates)

if __name__ == "__main__":
    main()

