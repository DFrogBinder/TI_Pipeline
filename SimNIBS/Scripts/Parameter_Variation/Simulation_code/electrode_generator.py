import gmsh
import numpy as np

class Electrode_Generator:
    def __init__(self, resolution, filename):
        self.resolution = resolution
        self.filename = filename
    def generate_rectangular_electrode(length, width, thickness, resolution, filename):
        """
        Generates a rectangular electrode mesh with given dimensions and resolution using Gmsh.

        Parameters:
            length (float): Length of the rectangle in cm.
            width (float): Width of the rectangle in cm.
            thickness (float): Thickness of the electrode in cm.
            resolution (float): Element size for the mesh in cm.
            filename (str): Name of the output mesh file.

        Returns:
            None
        """
        gmsh.initialize()
        gmsh.model.add("rectangular_electrode")

        # Define the points of the rectangle
        p1 = gmsh.model.geo.addPoint(0, 0, 0, resolution)
        p2 = gmsh.model.geo.addPoint(length, 0, 0, resolution)
        p3 = gmsh.model.geo.addPoint(length, width, 0, resolution)
        p4 = gmsh.model.geo.addPoint(0, width, 0, resolution)
        p5 = gmsh.model.geo.addPoint(0, 0, thickness, resolution)
        p6 = gmsh.model.geo.addPoint(length, 0, thickness, resolution)
        p7 = gmsh.model.geo.addPoint(length, width, thickness, resolution)
        p8 = gmsh.model.geo.addPoint(0, width, thickness, resolution)

        # Define the lines of the rectangle
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)
        l5 = gmsh.model.geo.addLine(p1, p5)
        l6 = gmsh.model.geo.addLine(p2, p6)
        l7 = gmsh.model.geo.addLine(p3, p7)
        l8 = gmsh.model.geo.addLine(p4, p8)
        l9 = gmsh.model.geo.addLine(p5, p6)
        l10 = gmsh.model.geo.addLine(p6, p7)
        l11 = gmsh.model.geo.addLine(p7, p8)
        l12 = gmsh.model.geo.addLine(p8, p5)

        # Define surfaces
        cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        s1 = gmsh.model.geo.addPlaneSurface([cl1])

        cl2 = gmsh.model.geo.addCurveLoop([l5, l9, -l12, -l1])
        s2 = gmsh.model.geo.addPlaneSurface([cl2])

        cl3 = gmsh.model.geo.addCurveLoop([l6, l10, -l7, -l2])
        s3 = gmsh.model.geo.addPlaneSurface([cl3])

        cl4 = gmsh.model.geo.addCurveLoop([l8, l11, -l3, -l7])
        s4 = gmsh.model.geo.addPlaneSurface([cl4])

        cl5 = gmsh.model.geo.addCurveLoop([l4, l8, -l12, -l5])
        s5 = gmsh.model.geo.addPlaneSurface([cl5])

        cl6 = gmsh.model.geo.addCurveLoop([l9, l10, l11, l12])
        s6 = gmsh.model.geo.addPlaneSurface([cl6])

        gmsh.model.geo.synchronize()

        # Define volume
        sl = gmsh.model.geo.addSurfaceLoop([s1, s2, s3, s4, s5, s6])
        gmsh.model.geo.addVolume([sl])

        # Mesh generation
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)

        # Save mesh to file
        gmsh.write(filename)
        gmsh.finalize()

    def generate_circular_electrode(radius, thickness, resolution, filename):
        """
        Generates a circular electrode mesh with given radius, thickness, and resolution using Gmsh.

        Parameters:
            radius (float): Radius of the circular electrode in cm.
            thickness (float): Thickness of the electrode in cm.
            resolution (float): Element size for the mesh in cm.
            filename (str): Name of the output mesh file.

        Returns:
            None
        """
        gmsh.initialize()
        gmsh.model.add("circular_electrode")

        # Define points for the circular base
        gmsh.model.geo.addDisk(0, 0, 0, radius, radius)
        gmsh.model.geo.synchronize()

        # Extrude to create thickness
        extruded_entities = gmsh.model.geo.extrude([(2, 1)], 0, 0, thickness, numElements=[int(thickness / resolution)])

        gmsh.model.geo.synchronize()

        # Mesh generation
        gmsh.model.mesh.generate(3)

        # Save mesh to file
        gmsh.write(filename)
        gmsh.finalize()

# Example usage
generate_rectangular_electrode(length=3.0, width=2.0, thickness=0.5, resolution=0.1, filename="rectangular_electrode.vtk")
generate_circular_electrode(radius=1.5, thickness=0.5, resolution=0.1, filename="circular_electrode.vtk")