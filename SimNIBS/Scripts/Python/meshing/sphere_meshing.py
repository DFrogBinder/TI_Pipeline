import numpy as np
import nibabel as nib

label_img = np.zeros((101,101,101), np.uint16)
xv, yv, zv = np.meshgrid(np.linspace(-50,50,101),
                        np.linspace(-50,50,101),
                        np.linspace(-50,50,101))

# make a two-layer sphere
r = np.sqrt(xv**2 + yv**2 + zv**2)
label_img[r<=40] = 5 # 5 corresponds to scalp
label_img[r<=35] = 2 # 2 corresponds to gray matter

# add a smaller decentered sphere
r = np.sqrt((xv-15)**2 + yv**2 + zv**2)
label_img[r<=15] = 17 # 17 is an arbitrary custom tissue label

# save
affine = np.eye(4)
img = nib.Nifti1Image(label_img, affine)
nib.save(img,'myspheres.nii.gz')
