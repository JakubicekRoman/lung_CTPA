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

# Perform further processing or analysis with the NIfTI data and lung mask
masked_data = nifti_array * (lung_mask_array>0)

# Perform morphological erosion on the lung mask
eroded_lung_mask = binary_erosion(lung_mask_array, iterations=5)

# Apply the eroded lung mask to the NIfTI data
masked_data = nifti_array * (eroded_lung_mask > 0)

# Create the vessels mask using the modified lung mask
vessels_mask = (nifti_array > -200)*(eroded_lung_mask)
skeleton_vessels = skeletonize(vessels_mask)
dist_map_vessels = distance_transform_edt(vessels_mask)

left_lung_mask = (lung_mask_array == 10) | (lung_mask_array == 11)
right_lung_mask = ((lung_mask_array == 12) | (lung_mask_array == 13) | (lung_mask_array == 14))    
trachea_mask = (lung_mask_array == 16)

# compute the mean values for each parts of lung in for cycle
# mean_values = []
# for i in range(10, 15):
#     mean_values.append(np.mean(nifti_array[lung_mask_array == i]))

# print(mean_values)

# morph closing the image 
# left_lung_mask_E = binary_dilation(left_lung_mask, iterations=5)
# left_lung_mask_E = binary_dilation(left_lung_mask_E, iterations=5)
# left_lung_mask_E = binary_erosion(left_lung_mask_E, iterations=5)
# left_lung_mask_E = binary_erosion(left_lung_mask_E, iterations=5)

# find non zero values in the distance map multiplied by the skeleton
points = np.where(dist_map_vessels * skeleton_vessels>0)
# store values of non zero values
values = dist_map_vessels[points]

# viewer = napari.view_image(right_lung_mask>0)   
# viewer.add_image(vessels_mask)       
# napari.run()

# create grid for interpolation
points_lung = np.where(left_lung_mask>0)
# grid_z0 = griddata((points[0], points[1], points[2]), values, (points_lung[0], points_lung[1], points_lung[2]), method='nearest')
grid_z0 = griddata(points, values, points_lung, method='nearest')

# create new image with interpolated values
new_image = np.zeros_like(lung_mask_array)
new_image[points_lung] = grid_z0

# viewer = napari.view_image(dist_map_vessels)
# napari.run()

viewer = napari.view_image(new_image)
napari.run()

# # Perform binary opening on the trachea mask
# trachea_mask = binary_erosion(trachea_mask, iterations=3)
# trachea_mask = binary_dilation(trachea_mask, iterations=3)

# # Perform binary closing on the trachea mask
# trachea_mask = binary_dilation(trachea_mask, iterations=5)
# trachea_mask = binary_erosion(trachea_mask, iterations=5)

# skeleton_trachea = skeletonize(trachea_mask)

# # Find the endpoints of the skeletonized trachea
# endpoints = SkeletonAnalyzer.find_endpoints(skeleton_trachea)

# # # Display the maximum projection of the endpoints
# maximum_lung = np.max(lung_mask_array, axis=1)
# plt.imshow(maximum_lung, cmap='gray')
# plt.axis('off')
# plt.show()

# maximum_ends = np.max(endpoints, axis=1)
# max_trachea_mask = np.max(trachea_mask, axis=1)
# plt.imshow(max_trachea_mask&~maximum_ends, cmap='gray')
# plt.axis('off')
# plt.show()

# # find endpoint as white pixels in the image (coordinates)
# loc_ends = np.argwhere(endpoints == 1)
# print('endpoints: ' + str(loc_ends))

# # find loc ends on the left with the min of third coordinate
# trachea = loc_ends[np.argmax(loc_ends[:, 2])]
# # remove this point
# loc_ends = np.delete(loc_ends, np.argmax(loc_ends[:, 2]), axis=0)

# # find the right hylus
# right_hylus = loc_ends[np.argmin(loc_ends[:, 0])]
# # find the left hylus
# left_hylus = loc_ends[np.argmax(loc_ends[:, 0])]

# print('left_hylus: ' + str(left_hylus))
# print('right_hylus: ' + str(right_hylus))

# # Create a binary mask for the left hylus
# bin_Hylus = np.zeros_like(lung_mask_array, dtype=np.bool_)
# bin_Hylus[left_hylus[0], left_hylus[1], left_hylus[2]] = True
# bin_Hylus[right_hylus[0], right_hylus[1], right_hylus[2]] = True


# # compute dinstace map (distance transform) from the left hylus
# dist_map = distance_transform_edt(~bin_Hylus)

# # show the distance map in half of second dimension
# plt.imshow(dist_map[:, :, int(dist_map.shape[2] / 2)])
# # plt.imshow(dist_map)
# plt.axis('off')
# plt.show()

# dist_map_masked = dist_map * (eroded_lung_mask > 0)
# # show the distance map using napari
# viewer = napari.view_image(dist_map_masked)
# napari.run()

# for left lungs measured mean value of data, but in specific postiion in distance map

# T2 = 65
# T1 = 60
# dist_map_masked = dist_map * (left_lung_mask > 0)    
# mean_value = np.mean(nifti_array[(dist_map_masked < T2) & (dist_map_masked > T1)])   

# # show threholds space of distance map using napari
# pom = (dist_map_masked < T2) & (dist_map_masked > T1)

# viewer = napari.view_image(trachea_mask)
# napari.run()

# viewer = napari.view_image(dist_map_masked)
# napari.run()
