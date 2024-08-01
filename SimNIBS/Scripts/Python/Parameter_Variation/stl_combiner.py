import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_and_prepare_mesh(file_path):
    # Load the mesh from file
    mesh = trimesh.load(file_path)
    return mesh

def plot_mesh(mesh, ax, color, is_point_cloud=False):
    vertices = mesh.vertices
    faces = mesh.faces
    if is_point_cloud:
        # Plot as point cloud
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color=color, s=1, alpha=0.6)
    else:
        # Plot as solid mesh with face color
        face_collection = Poly3DCollection(vertices[faces], alpha=0.8, facecolor=color, edgecolor='k', linewidth=0.1)
        ax.add_collection3d(face_collection)

def main():
    # Paths to your STL files
    file_paths = ['/home/cogitatorprime/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/cuttof/1.5cm_1.5mA_AF4-PO4_AF3-PO3_ellipse/nifti/40p_cutoff_thresholded_volume.stl',
                  '/home/cogitatorprime/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/cuttof/1.5cm_1.5mA_AF4-PO4_AF3-PO3_ellipse/nifti/60p_cutoff_thresholded_volume.stl', 
                  '/home/cogitatorprime/sandbox/TI_Pipeline/SimNIBS/Scripts/Python/Parameter_Variation/cuttof/1.5cm_1.5mA_AF4-PO4_AF3-PO3_ellipse/nifti/80p_cutoff_thresholded_volume.stl']
    meshes = [load_and_prepare_mesh(path) for path in file_paths]

    # Determine which mesh has the most vertices
    max_vertices_index = np.argmax([len(mesh.vertices) for mesh in meshes])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define colors
    colors = ['red', 'green', 'blue']
    
    for index, (mesh, color) in enumerate(zip(meshes, colors)):
        if index == max_vertices_index:
            # Plot the largest mesh as a point cloud
            plot_mesh(mesh, ax, color, is_point_cloud=True)
        else:
            # Plot other meshes with solid colors
            plot_mesh(mesh, ax, color, is_point_cloud=False)

    scale = np.concatenate([mesh.vertices for mesh in meshes]).flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

if __name__ == "__main__":
    main()