import vtk

def visualize_vtk_file(vtk_file_path):
    # Create a reader for the given .vtk file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()

    # Create a mapper that will hold the geometry and the associated data
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    # Create an actor that will be used to display the mesh
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a renderer and add the actor to it
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)  # Setting background color

    # Create a render window and add the renderer to it
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)  # Set the window size

    # Create a render window interactor and set the render window to it
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Initialize the render window interactor and start the rendering loop
    render_window_interactor.Initialize()
    render_window.Render()
    render_window_interactor.Start()

# Replace 'path_to_your_vtk_file.vtk' with the path to your .vtk file
vtk_file_path = '/home/cogitatorprime/sandbox/TI_Pipeline/tTIS/Simple_Export_Save_Dir/fem.vtk'
visualize_vtk_file(vtk_file_path)

