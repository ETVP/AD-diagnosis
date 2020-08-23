import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from pylab import ginput


path = "./img/ad.nii"
data = nib.load(path).get_data()
data = np.asarray(data)
data = np.squeeze(data)

plt.imshow(data[:, :, 153])
plt.show()

