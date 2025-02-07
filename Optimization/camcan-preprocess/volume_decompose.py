import gmsh

# Initialize Gmsh and set the terminal option to get output.
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

# Load the mesh file "head.msh"
gmsh.open("head_2.msh")

# Save the name of the original model (usually "Model.1")
original_model = gmsh.model.getCurrent()

# Get all 3D entities (volumes) in the model.
volumes = gmsh.model.getEntities(3)
print("Found %d volume(s) in the mesh." % len(volumes))

# Loop over all volumes.
for (dim, vol_tag) in volumes:
    print("Processing volume with tag:", vol_tag)
    
    # Extract the mesh nodes for the current volume.
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes(dim, vol_tag)
    # Extract the mesh elements for the current volume.
    # getElements returns a triple:
    #   - element types,
    #   - element tags, and 
    #   - a flat list of node tags for each element.
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, vol_tag)
    
    # Create a new model that will hold just this volume's mesh.
    new_model_name = "volume_" + str(vol_tag)
    gmsh.model.add(new_model_name)
    gmsh.model.setCurrent(new_model_name)
    
    # Add the extracted nodes and elements to the new model.
    # (We use the same dimension and volume tag; note that the new model does
    # not have a full geometry but it is sufficient for writing out a .msh file.)
    gmsh.model.mesh.addNodes(dim, vol_tag, nodeTags, nodeCoords)
    gmsh.model.mesh.addElements(dim, vol_tag, elemTypes, elemTags, elemNodeTags)
    
    # Write out the mesh for this volume to a separate file.
    outName = new_model_name + ".msh"
    gmsh.write(outName)
    print("Wrote submesh to", outName)
    
    # Remove the new model so that we can process the next volume.
    gmsh.model.remove()  # removes the current model (the new one)
    gmsh.model.setCurrent(original_model)

# Finalize Gmsh.
gmsh.finalize()
