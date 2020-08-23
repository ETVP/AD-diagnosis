import vtk
import os


def TestDisplay():
    """Display the output"""

    inpath = r"/home/fan/Desktop/processed/ad/Patient Output Volume.nii"
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(inpath)
    reader.Update()

    size = reader.GetOutput().GetDimensions()
    center = reader.GetOutput().GetCenter()
    spacing = reader.GetOutput().GetSpacing()
    center1 = (center[0], center[1], center[2])
    center2 = (center[0], center[1], center[2])
    if size[2] % 2 == 1:
        center1 = (center[0], center[1], center[2] + 0.5*spacing[2])
    if size[0] % 2 == 1:
        center2 = (center[0] + 0.5*spacing[0], center[1], center[2])
    vrange = reader.GetOutput().GetScalarRange()

    map1 = vtk.vtkImageSliceMapper()
    map1.BorderOn()
    map1.SliceAtFocalPointOn()
    map1.SliceFacesCameraOn()
    map1.SetInputConnection(reader.GetOutputPort())
    map2 = vtk.vtkImageSliceMapper()
    map2.BorderOn()
    map2.SliceAtFocalPointOn()
    map2.SliceFacesCameraOn()
    map2.SetInputConnection(reader.GetOutputPort())

    slice1 = vtk.vtkImageSlice()
    slice1.SetMapper(map1)
    slice1.GetProperty().SetColorWindow(vrange[1]-vrange[0])
    slice1.GetProperty().SetColorLevel(0.5*(vrange[0]+vrange[1]))
    slice2 = vtk.vtkImageSlice()
    slice2.SetMapper(map2)
    slice2.GetProperty().SetColorWindow(vrange[1]-vrange[0])
    slice2.GetProperty().SetColorLevel(0.5*(vrange[0]+vrange[1]))

    ratio = size[0]*1.0/(size[0]+size[2])

    ren1 = vtk.vtkRenderer()
    ren1.SetViewport(0,0,ratio,1.0)
    ren2 = vtk.vtkRenderer()
    ren2.SetViewport(ratio,0.0,1.0,1.0)
    ren1.AddViewProp(slice1)
    ren2.AddViewProp(slice2)

    cam1 = ren1.GetActiveCamera()
    cam1.ParallelProjectionOn()
    cam1.SetParallelScale(0.5*spacing[1]*size[1])
    cam1.SetFocalPoint(center1[0], center1[1], center1[2])
    cam1.SetPosition(center1[0], center1[1], center1[2] - 100.0)

    cam2 = ren2.GetActiveCamera()
    cam2.ParallelProjectionOn()
    cam2.SetParallelScale(0.5*spacing[1]*size[1])
    cam2.SetFocalPoint(center2[0], center2[1], center2[2])
    cam2.SetPosition(center2[0] + 100.0, center2[1], center2[2])

    style = vtk.vtkInteractorStyleImage()
    style.SetInteractionModeToImageSlicing()

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(style)

    renwin = vtk.vtkRenderWindow()
    renwin.SetSize((size[0] + size[2]) // 2 * 2, size[1] // 2 * 2) # keep size even
    renwin.AddRenderer(ren1)
    renwin.AddRenderer(ren2)

    renwin.Render()

    renwin.SetInteractor(iren)
    iren.Initialize()
    iren.Start()

if __name__ == '__main__':
    TestDisplay()