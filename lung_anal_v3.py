import os
import numpy as np
import nibabel as nib
import napari
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import skeletonize
# from scipy.ndimage import binary_hit_or_miss 
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
from scipy.interpolate import griddata
from utils import display_orthogonal_views

# from skimage.morphology import isotropic_erosion


class analyze_lung:
    @staticmethod
    def define_contour(mask):
        # create magnitude of gradient of left lung mask
        grad_x = np.gradient(mask.astype(float), axis=0)
        grad_y = np.gradient(mask.astype(float), axis=1)
        grad_z = np.gradient(mask.astype(float), axis=2)
        # create magnitude of gradient
        grad_magnitude = (np.abs(grad_x)>0) | (np.abs(grad_y)>0) | (np.abs(grad_z)>0)
        # do cumulative sum in 2. direction of left lung mask
        LM_2 = np.cumsum(mask, axis=0)
        # take only values in range of LM_2 = 1:2
        LM_2 = (LM_2 > 0) & (LM_2 < 5)
        LM_2 = binary_dilation(LM_2, iterations=3)
        # remove voxels in grad_magnitude at location of LM_2
        grad_magnitude[LM_2] = 0

        grad_magnitude = binary_dilation(grad_magnitude, iterations=5)
        grad_magnitude = binary_erosion(grad_magnitude, iterations=3)
        # preserve only the largest binary object
        labeled_objects, num_objects = ndimage.label(grad_magnitude)
        largest_object = np.argmax(np.bincount(labeled_objects.flat)[1:]) + 1
        grad_magnitude = labeled_objects == largest_object
        return grad_magnitude
    def remove_contour(mask, contour,dil=1):
        # remove contour from mask
        contour = binary_dilation(contour, iterations=dil)
        mask[contour] = 0
        return mask
    def compute_mean_intensity(mask, data):
        # compute mean intensity of data in mask
        data[data==0] = np.nan
        mean_intensity = np.nanmean(data[mask])
        return mean_intensity


data_dir = r'D:\Projekty\CTPA_VFN\data\nifti'

# Get a list of all NIfTI files in the directory
nifti_files = [file for file in os.listdir(data_dir) if file.endswith('.nii.gz')]

# Iterate over the NIfTI files
# for nifti_file in nifti_files:
nifti_file = nifti_files[3]
print(nifti_file)

nifti_path = os.path.join(data_dir, nifti_file)
nifti_data = nib.load(nifti_path)
nifti_array = np.array(nifti_data.get_fdata())

# Load the corresponding lung mask
lung_mask_file = nifti_file.replace('.nii', '_lung.nii')
lung_mask_path = os.path.join(data_dir, lung_mask_file).replace('nifti', 'masks')
lung_mask_data = nib.load(lung_mask_path)
lung_mask_array = np.array(lung_mask_data.get_fdata())

# subsampled data
nifti_array = ndimage.zoom(nifti_array, 0.5, order=0)
lung_mask_array = ndimage.zoom(lung_mask_array, 0.5, order=0)

# lung_mask = lung_mask_array[(lung_mask_array >= 10) & (lung_mask_array <= 15)]

left_lung_mask = (lung_mask_array == 10) | (lung_mask_array == 11)
right_lung_mask = ((lung_mask_array == 12) | (lung_mask_array == 13) | (lung_mask_array == 14))    
trachea_mask = (lung_mask_array == 16)

# # Perform morphological erosion on the lung mask
left_lung_mask = binary_dilation(left_lung_mask, iterations=5)
left_lung_mask = binary_erosion(left_lung_mask, iterations=7)

vessels_mask = (nifti_array > -500)

lung_tissue = nifti_array.copy()
lung_tissue[vessels_mask] = 0
lung_tissue[~left_lung_mask] = 0

# # Apply the eroded lung mask to the NIfTI data
# masked_data = nifti_array * (eroded_lung_mask > 0).astype(float)

mean_intensities = []

# for left lung mask
for i in range(8):
    contr_left = analyze_lung.define_contour(left_lung_mask)
    left_lung_mask = analyze_lung.remove_contour(left_lung_mask, contr_left,2)
    mean_intensity_left = analyze_lung.compute_mean_intensity(contr_left, lung_tissue)
    mean_intensities.append(mean_intensity_left)

plt.figure()
plt.plot(mean_intensities)
plt.title('Mean intensity of left lung mask')
plt.xlabel('iteration')
plt.ylabel('mean intensity')
plt.show()


# contr_left = analyze_lung.define_contour(left_lung_mask)
# left_lung_mask = analyze_lung.remove_contour(left_lung_mask, contr_left,3)
# mean_intensity_left = analyze_lung.compute_mean_intensity(contr_left, nifti_array)



# # for right lung mask
# contr_right = analyze_lung.define_contour(np.flip(right_lung_mask, axis=0))
# contr_right = np.flip(contr_right, axis=0)
# mean_intensity_right = analyze_lung.compute_mean_intensity(contr_right, nifti_array)
# print(mean_intensity_right)

# # Create a viewer and display the contours with the original data
# viewer = napari.Viewer()
# viewer.add_image(vessels_mask, name='nifti_array')
# napari.run()

viewer = napari.Viewer()
viewer.add_image(lung_tissue, name='nifti_array')
napari.run()

viewer = napari.Viewer()
viewer.add_image(nifti_array, name='nifti_array')
viewer.add_labels(contr_left>0, name='lung')
# viewer.add_labels(trachea_mask>0, name='trachea')
napari.run()

# # display masked data via orthogonal views
# display_orthogonal_views(masked_data)

