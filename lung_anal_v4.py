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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import median_filter
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

# from skimage.morphology import isotropic_erosion


class analyze_lung:
    @staticmethod
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
nifti_file = nifti_files[5]
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

vessels_mask = (nifti_array > -600)
vessels_mask = binary_dilation(vessels_mask, iterations=1)

lung_tissue = nifti_array.copy()
lung_tissue[vessels_mask] = np.nan
lung_tissue[~left_lung_mask] = np.nan
# lung_tissue = median_filter(lung_tissue, size=5)

# compute and display histogram of intesities of lung tissue
# plt.figure()
# plt.hist(lung_tissue[~np.isnan(lung_tissue)], bins=256)
# plt.xlabel('Intensity')
# plt.ylabel('Frequency')
# plt.title('Histogram of lung tissue intensities')
# plt.show()


# segment the lung tissue by k-means clustering for three clusses
scaler = StandardScaler()
lung_tissue_vekt = lung_tissue[~np.isnan(lung_tissue)].reshape(-1,1)
# stadardize to range 0-1 according to -1000 to -500
# lung_tissue_vekt = (lung_tissue_vekt - (-1000) ) / (-600 - (-1000))
lung_tissue_vekt[lung_tissue_vekt>-600] = -600
# subquantize the lung_tissue_vekt to 256 level of shades
# lung_tissue_vekt = np.round(lung_tissue_vekt*512).astype(int)
lung_tissue_vekt = np.round(lung_tissue_vekt).astype(int)

# plt.figure()
# plt.hist(lung_tissue_vekt[~np.isnan(lung_tissue_vekt)], bins=64)
# plt.xlabel('Intensity')
# plt.ylabel('Frequency')
# plt.title('Histogram of lung tissue intensities')
# plt.show()

gm = BayesianGaussianMixture(n_components=3, random_state=42).fit(lung_tissue_vekt)
print(gm.means_)
print(np.sqrt(gm.covariances_))
print(gm.weights_)

# gm = GaussianMixture(n_components=3, random_state=42).fit(lung_tissue_vekt)
# print(gm.means_)
# print(np.sqrt(gm.covariances_))

# display histogram of data and estimated Gassians distributions together
plt.figure()
plt.hist(lung_tissue_vekt[~np.isnan(lung_tissue_vekt)], bins=425, density=True)
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of lung tissue intensities')
x = np.linspace(-600, -1000, 1000)
x = np.expand_dims(x, axis=1)
y = np.zeros_like(x)
for i in range(3):
    y = y + gm.weights_[i]*np.exp(-0.5*(x-gm.means_[i])**2/gm.covariances_[i])/np.sqrt(2*np.pi*gm.covariances_[i])
    plt.plot(x, gm.weights_[i]*np.exp(-0.5*(x-gm.means_[i])**2/gm.covariances_[i])/np.sqrt(2*np.pi*gm.covariances_[i]))
plt.plot(x, y)
plt.show()



# # fit the kmeans model
# # speciy special postion of centroids
# kmeans = KMeans(n_clusters=3,init=np.array([[64],[160],[180]]), random_state=0).fit(lung_tissue_vekt)

# lung_tissue_classes = np.zeros(lung_tissue.shape)
# # on the position of matrix lung_tissue_classes, where is not nan, insert value of kmenas_lables_
# lung_tissue_classes[~np.isnan(lung_tissue)] = kmeans.labels_+1

# lung_tissue_classes = lung_tissue_classes.astype(int)  # Convert to integer type

# display_orthogonal_views(lung_tissue_classes)

viewer = napari.Viewer()
viewer.add_image(nifti_array, name='lung tissue')
napari.run()

# # viewer = napari.Viewer()
# # viewer.add_image(lung_tissue, name='nifti_array',contrast_limits=[-1000, -500])
# # viewer.add_labels((lung_tissue_classes==2) | (lung_tissue_classes==3), name='lung')
# # napari.run()

# viewer = napari.Viewer()
# viewer.add_image(nifti_array, name='nifti_array',contrast_limits=[-1000, -600])
# viewer.add_labels((lung_tissue_classes==1), name='1')
# viewer.add_labels((lung_tissue_classes==2), name='2')
# viewer.add_labels((lung_tissue_classes==3), name='3')
# napari.run()

# # viewer = napari.Viewer()
# # viewer.add_image(nifti_array, name='nifti_array',contrast_limits=[-1000, -500])
# # viewer.add_labels((lung_tissue_classes==2), name='lung')
# # # viewer.add_labels((lung_tissue_classes==2) | (lung_tissue_classes==3), name='lung')
# # napari.run()

# # viewer = napari.Viewer()
# # viewer.add_image(nifti_array, name='nifti_array',contrast_limits=[-1000, -500])
# # viewer.add_labels((lung_tissue_classes==3), name='lung')
# # # viewer.add_labels((lung_tissue_classes==2) | (lung_tissue_classes==3), name='lung')
# # napari.run()

# # # display masked data via orthogonal views
# # display_orthogonal_views(masked_data)

# # plt.figure()
# # plt.hist(lung_tissue_vekt, bins=256)
# # plt.xlabel('Intensity')
# # plt.ylabel('Frequency')
# # plt.title('Histogram of lung tissue intensities')
# # plt.show()