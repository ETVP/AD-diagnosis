import nrrd
import SimpleITK as sitk
import numpy as np


img_path = r'/home/fan/Desktop/img_data/test/ad.0.5.npy'
img_data = np.load(img_path)
img_data = sitk.GetImageFromArray(img_data)
sitk.WriteImage(img_data, '/home/fan/Desktop/ad.nii')
