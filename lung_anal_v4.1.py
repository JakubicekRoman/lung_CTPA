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

# create new folder
if not os.path.exists(data_dir.replace('nifti','result_img')):
    os.makedirs(data_dir.replace('nifti','result_img'))

# Get a list of all NIfTI files in the directory
nifti_files = [file for file in os.listdir(data_dir) if file.endswith('.nii.gz')]

# Iterate over the NIfTI files
# for nifti_file in nifti_files:
for pat in range(0,1):
    nifti_file = nifti_files[pat]
    print(nifti_file)

    nifti_path = os.path.join(data_dir, nifti_file)
    nifti_data = nib.load(nifti_path)
    nifti_array = np.array(nifti_data.get_fdata())

    # Load the corresponding lung mask
    lung_mask_file = nifti_file.replace('.nii', '_lung.nii')
    lung_mask_path = os.path.join(data_dir, lung_mask_file).replace('nifti', 'masks')
    lung_mask_data = nib.load(lung_mask_path)
    lung_mask_array = np.array(lung_mask_data.get_fdata())

    dataO, lung_mask = data_subsampling(nifti_array, lung_mask_array, factor=0.5)

    data = median_filter(dataO, size=5)

    left_lung, right_lung, trachea, vessels_mask = lung_separate(data, lung_mask)

    results = pd.DataFrame(columns=['file', 'whole_mean1', 'whole_mean2', 'whole_mean3',
                                    'Whole_std1', 'Whole_std2', 'Whole_std3', 
                                    'Whole_rate1', 'Whole_rate2', 'Whole_rate3',])

    res=[]

    # # --------------- zkusebni odstavec
    # path_save = os.path.join(nifti_path.replace('.nii.gz','_whole.png').replace('nifti','result_img'))
    # gmW, val = int_analyze(data, (lung_mask==10), vessels_mask, path_save)
    # res.append(val.means_)
    # res.append(np.sqrt(val.covariances_))
    # res.append(val.weights_)

    # # global of whole lung
    path_save = os.path.join(nifti_path.replace('.nii.gz','_whole.png').replace('nifti','result_img'))
    gmW, val = int_analyze(data, ((lung_mask<15) & (lung_mask>9)), vessels_mask, path_save)
    res.append(val.means_)
    res.append(np.sqrt(val.covariances_))
    res.append(val.weights_)

    # path_save = os.path.join(nifti_path.replace('.nii.gz','_Left.png').replace('nifti','result_img'))
    # gm = int_analyze(data, left_lung, vessels_mask, path_save)
    # res.append(gm.means_.squeeze().T)
    # res.append(np.sqrt(gm.covariances_.squeeze()).T)
    # res.append(gm.weights_.T)

    # path_save = os.path.join(nifti_path.replace('.nii.gz','_Right.png').replace('nifti','result_img'))
    # gm = int_analyze(data, right_lung, vessels_mask, path_save)
    # res.append(gm.means_.squeeze().T)
    # res.append(np.sqrt(gm.covariances_.squeeze()).T)
    # res.append(gm.weights_.T)

    # parts = ['SL', 'IL', 'SR', 'MR', 'IR']
    # k = 0
    # for part in range(10,15):
    #     print(part)
    #     path_save = os.path.join(nifti_path.replace('.nii.gz','_'+parts[k]+'.png').replace('nifti','result_img'))
    #     gm = int_analyze(data, lung_mask==part, vessels_mask, path_save)
    #     res.append(gm.means_.squeeze())
    #     res.append(np.sqrt(gm.covariances_.squeeze()))
    #     res.append(gm.weights_)
    #     print(res)
    #     k+=1

    # # predict the cluster for each voxel of nifti data masked by lung mask by gmW
    # lung_tissue = data.copy()
    # lung_tissue[vessels_mask] = np.nan
    # # lung_tissue[~((lung_mask<15) & (lung_mask>9))] = np.nan
    # lung_tissue[~(lung_mask==10)] = np.nan
    # lung_tissue_vekt = lung_tissue[~np.isnan(lung_tissue)].reshape(-1,1)
    # lung_tissue_vekt[lung_tissue_vekt>-600] = -600
    # lung_tissue_vekt = np.round(lung_tissue_vekt).astype(int)
    # gmW_labels = gmW.predict(lung_tissue_vekt)

    # labels = np.zeros_like(lung_tissue)
    # labels[~np.isnan(lung_tissue)] = gmW_labels+1

    # labels = labels.reshape(lung_tissue.shape)

    # convert res to numpy matrix
    res = np.array(res)

    # save the results to xlsx file for each patient as one raw in excel files
    results.loc[pat] = [nifti_file]+res.flatten().tolist()
    results.to_excel(data_dir.replace('nifti','')+'results.xlsx', index=False)


# viewer = napari.Viewer()
# viewer.add_image(dataO, name='lung tissue', contrast_limits=[-1000, -500])
# viewer.add_labels(labels.astype(int), name='GMM whole lung')
# napari.run()

# viewer = napari.Viewer()
# viewer.add_image(labels, name='lung tissue')
# napari.run()
