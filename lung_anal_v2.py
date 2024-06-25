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

# from skimage.morphology import isotropic_erosion


class SkeletonAnalyzer:
    @staticmethod
    def find_endpoints(skeleton):
        skeleton_modified = skeleton.copy()
        endpoints_list = []
        # Find the endpoints of the skeleton
        endpoints = np.argwhere(skeleton == 1)
        for endpoint in endpoints:
            x, y, z = endpoint
            # Check if the endpoint is a true endpoint
            if np.sum(skeleton[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]) > 2:
                skeleton_modified[x, y, z] = 0
                endpoints_list.append((x, y, z))
        return skeleton_modified


data_dir = r'D:\Projekty\CTPA_VFN\data\nifti'

# Get a list of all NIfTI files in the directory
nifti_files = [file for file in os.listdir(data_dir) if file.endswith('.nii.gz')]

# Iterate over the NIfTI files
# for nifti_file in nifti_files:
nifti_file = nifti_files[0]
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

# Perform morphological erosion on the lung mask
eroded_lung_mask = binary_erosion(lung_mask_array, iterations=5)

# Apply the eroded lung mask to the NIfTI data
masked_data = nifti_array * (eroded_lung_mask > 0)

left_lung_mask = (lung_mask_array == 10) | (lung_mask_array == 11)
right_lung_mask = ((lung_mask_array == 12) | (lung_mask_array == 13) | (lung_mask_array == 14))    
trachea_mask = (lung_mask_array == 16)

# Smooth the left lung mask
S_left_lung_mask = ndimage.gaussian_filter(left_lung_mask.astype(float), sigma=0.5)

# compute gradient of left lung mask in all direction separately
grad_x = np.gradient(S_left_lung_mask, axis=0) 
grad_y = np.gradient(S_left_lung_mask, axis=1)
grad_z = np.gradient(S_left_lung_mask, axis=2)
# compute angle of gradient in all direction separately in degrees
grad_angle_x = np.degrees(np.arctan2(grad_y, grad_z))
grad_angle_y = np.degrees(np.arctan2(grad_x, grad_z))
grad_angle_z = np.degrees(np.arctan2(grad_x, grad_y))

# compute gradient
grad_x = np.gradient(left_lung_mask.astype(float), axis=0)
grad_y = np.gradient(left_lung_mask.astype(float), axis=1)
grad_z = np.gradient(left_lung_mask.astype(float), axis=2)

# create magnitude of gradient as sum of abs value of gradient in all directions
grad_magnitude = (grad_x < 0) | (np.abs(grad_y) > 0) | (np.abs(grad_z) > 0)

# multiplay the gradient magnitude with binarized angle image - only in angle range 0-45 degrees
# grad_magnitude = grad_magnitude * ((grad_angle_z >= 0) & (grad_angle_z <= 45))
angles = ((grad_angle_z >= 1) & (grad_angle_z <= 90))


# Create a viewer
viewer = napari.view_image(angles)
napari.run()
