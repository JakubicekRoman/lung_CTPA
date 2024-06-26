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
from scipy.ndimage import median_filter
import pandas as pd
from scipy import ndimage
from utils import int_analyze, data_subsampling, lung_separate



data_dir = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\nifti'

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

data, lung_mask = data_subsampling(nifti_array, lung_mask_array, factor=0.5)

left_lung, right_lung, trachea, vessels_mask = lung_separate(data, lung_mask)

results = pd.DataFrame(columns=['file', 'superior lobe of left lung', 'inferior lobe of left lung',
                                 'superior lobe of right lung', 'middle lobe of right lung', 'inferior lobe of right lung'])  

res=[]

gm = int_analyze(data, lung_mask<15 & lung_mask<9, vessels_mask)
res.append(gm.means_.squeeze())
res.append(np.sqrt(gm.covariances_.squeeze()))
res.append(gm.weights_)

for part in range(10,15):
    print(part)
    # display_orthogonal_views(mask==part, slice_index=None)
    gm = int_analyze(data, lung_mask==part, vessels_mask)
    res.append(gm.means_.squeeze())
    res.append(np.sqrt(gm.covariances_.squeeze()))
    res.append(gm.weights_)
    print(res)

# convert res to numpy matrix
res = np.array(res).reshape(6, 3, 3)

# gm = int_analyze(data, mask==11, vessels_mask)
# print(gm.means_)

# gm = int_analyze(data, mask==12, vessels_mask)
# print(gm.means_)

# display_orthogonal_views(data, slice_index=None)

# results = results.append({'file': nifti_file, 'left_mean': gm.means_[0], 'left_std': np.sqrt(gm.covariances_[0]), 'right_mean': gm.means_[1], 'right_std': np.sqrt(gm.covariances_[1]), 'trachea_mean': gm.means_[2], 'trachea_std': np.sqrt(gm.covariances_[2]), 'vessels_mean': np.nan, 'vessels_std': np.nan}, ignore_index=True)     

    # gm = analyze_lung.int_analyze(data, mask==part, vessels_mask)
    # results = results.append({'file': nifti_file, 'left_mean': gm.means_[0], 'left_std': np.sqrt(gm.covariances_[0]), 'right_mean': gm.means_[1], 'right_std': np.sqrt(gm.covariances_[1]), 'trachea_mean': gm.means_[2], 'trachea_std': np.sqrt(gm.covariances_[2]), 'vessels_mean': np.nan, 'vessels_std': np.nan}, ignore_index=True)    


    # print(gm.means_)
    # print(np.sqrt(gm.covariances_))
    # print(gm.weights_)

# save the table into csv file

# viewer = napari.Viewer()
# viewer.add_image(vessels_mask, name='lung tissue')
# napari.run()

