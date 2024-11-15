import numpy as np
import gmsh

gmsh.initialize()
gmsh.model.add("model")
gmsh.merge('/home/cogitatorprime/sandbox/SimNIBS/Scripts/Python/neshted_sphere/neshted_spheres.msh')

# Find out the highest dimension of the entities in the model
dims = gmsh.model.getDimension()

# Get the bounding box of the highest dimension entities
boundingBoxes = []
entities = gmsh.model.getEntities(dims)
for entity in entities:
    bbox = gmsh.model.getBoundingBox(dim=dims, tag=entity[1])
    boundingBoxes.append(bbox)

# Calculate the overall bounding box if there are multiple entities
if boundingBoxes:
    # Initialize min and max coordinates with the first bounding box
    minX, minY, minZ, maxX, maxY, maxZ = boundingBoxes[0]
    for bbox in boundingBoxes[1:]:
        minX = min(minX, bbox[0])
        minY = min(minY, bbox[1])
        minZ = min(minZ, bbox[2])
        maxX = max(maxX, bbox[3])
        maxY = max(maxY, bbox[4])
        maxZ = max(maxZ, bbox[5])

    # The midpoint along the Z-axis
    z_cut = (minZ + maxZ) / 2

    # Calculate midpoints along the X and Y axes
    x_mid = (minX + maxX) / 2
    y_mid = (minY + maxY) / 2

    # Calculate the size of the plane based on the model's extents to ensure it covers the entire model
    plane_size = max(maxX - minX, maxY - minY)

    # Create a rectangle centered at the model's midpoint and positioned at z_cut along the Z-axis
    # Adjust the rectangle's corner position to center it based on x_mid and y_mid
    plane_tag = gmsh.model.occ.addRectangle(x_mid - plane_size / 2, y_mid - plane_size / 2, z_cut, plane_size, plane_size)

gmsh.model.occ.synchronize()

# Visualize the result
if gmsh.model.getDimension() > 0:
    gmsh.fltk.run()

# Perform the cutting operation
if dims == 3:
    # For 3D, assuming cutting volumes
    volumes = gmsh.model.getEntities(3)
    for volume in volumes:
        _, new_volume_tags = gmsh.model.occ.cut([(3, volume[1])], [(2, plane_tag)], removeObject=False, removeTool=False)
elif dims == 2:
    # For 2D, assuming cutting surfaces
    surfaces = gmsh.model.getEntities(2)
    for surface in surfaces:
        _, new_surface_tags = gmsh.model.occ.cut([(2, surface[1])], [(2, plane_tag)], removeObject=False, removeTool=False)

gmsh.model.occ.synchronize()


gmsh.finalize()