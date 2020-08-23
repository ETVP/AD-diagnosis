import vtk


# stl_reader = vtk.vtkSTLReader()
# stl_reader.SetFileName("/home/fan/Desktop/cube.stl")
nii_reader = vtk.vtkNIFTIImageReader()
nii_reader.SetFileName(r"/home/fan/Desktop/processed/mp_ad/ADNI_002_S_2010_MR_MPRAGE_br_raw_20110122171028174_96_S98069_I213880.nii")
nii_reader.TimeAsVectorOn()
nii_reader.Update()

# size = nii_reader.GetOutput().GetDimensions()
# print(size)


mapper = vtk.vtkPolyDataMapper()

mapper.SetInputConnection(nii_reader.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create a rendering window and renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

# Create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Assign actor to the renderer
ren.AddActor(actor)

# Enable user interface interactor
iren.Initialize()
renWin.Render()
iren.Start()