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
from utils import int_analyze, data_subsampling, lung_separate, predict_mask
import openpyxl
from scipy.stats import entropy


data_dir = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\data2\nifti'

# create new folder
if not os.path.exists(data_dir.replace('nifti','result_img')):
    os.makedirs(data_dir.replace('nifti','result_img'))

# Get a list of all NIfTI files in the directory
nifti_files = [file for file in os.listdir(data_dir) if file.endswith('.nii.gz')]

results = pd.DataFrame(columns=['file', 'whole_mean1', 'whole_mean2', 'whole_mean3',
                                'Whole_std1', 'Whole_std2', 'Whole_std3', 
                                'Whole_rate1', 'Whole_rate2', 'Whole_rate3',
                                 'Left_mean1', 'Left_mean2', 'Left_mean3',
                                 'Left_std1', 'Left_std2', 'Left_std3',
                                 'Left_rate1', 'Left_rate2', 'Left_rate3',
                                 'Right_mean1', 'Right_mean2', 'Right_mean3',
                                 'Right_std1', 'Right_std2', 'Right_std3',
                                 'Right_rate1', 'Right_rate2', 'Right_rate3',
                                 'SL_mean1', 'SL_mean2', 'SL_mean3',
                                 'SL_std1', 'SL_std2', 'SL_std3',
                                 'SL_rate1', 'SL_rate2', 'SL_rate3',
                                 'IL_mean1', 'IL_mean2', 'IL_mean3',
                                 'IL_std1', 'IL_std2', 'IL_std3',
                                 'IL_rate1', 'IL_rate2', 'IL_rate3',
                                 'SR_mean1', 'SR_mean2', 'SR_mean3',
                                 'SR_std1', 'SR_std2', 'SR_std3',
                                 'SR_rate1', 'SR_rate2', 'SR_rate3',
                                 'MR_mean1', 'MR_mean2', 'MR_mean3',
                                 'MR_std1', 'MR_std2', 'MR_std3',
                                 'MR_rate1', 'MR_rate2', 'MR_rate3',
                                 'IR_mean1', 'IR_mean2', 'IR_mean3',
                                 'IR_std1', 'IR_std2', 'IR_std3',
                                 'IR_rate1', 'IR_rate2', 'IR_rate3'])


factor = 0.5
# Iterate over the NIfTI files
# for pat in range(2,3):
for pat in range(0,len(nifti_files)):
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

    dataO, lung_mask = data_subsampling(nifti_array, lung_mask_array, factor=factor)

    left_lung, right_lung, trachea, vessels_mask = lung_separate(dataO, lung_mask)

    # data = median_filter(dataO*((~vessels_mask)*(left_lung | right_lung)).astype(int), size=5)
    data = median_filter(dataO, size=5)
    # data = data*(~vessels_mask)*(left_lung | right_lung)
    data = data*(left_lung | right_lung)
    res=[]

    # # --------------- zkusebni odstavec
    # path_save = os.path.join(nifti_path.replace('0.nii.gz','_whole.png').replace('nifti','result_img'))
    # gmW, val = int_analyze(data, (lung_mask==10), vessels_mask, path_save)
    # res.append(val.means_)
    # res.append(np.sqrt(val.covariances_))
    # res.append(val.weights_)

    # # global of whole lung
    path_save = os.path.join(nifti_path.replace('.nii.gz','_whole.png').replace('nifti','result_img'))
    gmW, val = int_analyze(data, ((left_lung) | (right_lung)), vessels_mask, path_save)
    res.append(val.means_)
    res.append(np.sqrt(val.covariances_))
    res.append(val.weights_)

    pred = predict_mask(data, vessels_mask, ((left_lung) | (right_lung)), gmW)
    factorUp = (np.size(nifti_array,0)/np.size(pred,0), np.size(nifti_array,1)/np.size(pred,1), np.size(nifti_array,2)/np.size(pred,2))
    pred = ndimage.zoom(pred, factorUp, order=0)
    labels_nifti = nib.Nifti1Image(pred, nifti_data.affine)
    nib.save(labels_nifti, nifti_path.replace('.nii.gz','_labels_whole.nii.gz').replace('nifti','result_img'))

    # # global of left lung
    path_save = os.path.join(nifti_path.replace('.nii.gz','_Left.png').replace('nifti','result_img'))
    gm, val = int_analyze(data, left_lung, vessels_mask, path_save)
    res.append(val.means_)
    res.append(np.sqrt(val.covariances_))
    res.append(val.weights_)
    predL = predict_mask(data, vessels_mask, left_lung, gm)

    # # global of rigth lung
    path_save = os.path.join(nifti_path.replace('.nii.gz','_Right.png').replace('nifti','result_img'))
    gm, val  = int_analyze(data, right_lung, vessels_mask, path_save)
    res.append(val.means_)
    res.append(np.sqrt(val.covariances_))
    res.append(val.weights_)
    predR = predict_mask(data, vessels_mask, right_lung, gm)

    pred = predL + predR
    factorUp = (np.size(nifti_array,0)/np.size(pred,0), np.size(nifti_array,1)/np.size(pred,1), np.size(nifti_array,2)/np.size(pred,2))
    pred = ndimage.zoom(pred, factorUp, order=0)
    labels_nifti = nib.Nifti1Image(pred, nifti_data.affine)
    nib.save(labels_nifti, nifti_path.replace('.nii.gz','_labels_LR.nii.gz').replace('nifti','result_img'))

    # # partial of lung lobes
    parts = ['SL', 'IL', 'SR', 'MR', 'IR']
    k = 0
    pred = np.zeros_like(data)
    for part in range(10,15):
        print(part)
        path_save = os.path.join(nifti_path.replace('.nii.gz','_'+parts[k]+'.png').replace('nifti','result_img'))
        gm, val  = int_analyze(data, lung_mask==part, vessels_mask, path_save)
        res.append(val.means_)
        res.append(np.sqrt(val.covariances_))
        res.append(val.weights_)
        predTemp = predict_mask(data, vessels_mask, lung_mask==part, gm)
        pred = pred + predTemp
        k+=1
    factorUp = (np.size(nifti_array,0)/np.size(pred,0), np.size(nifti_array,1)/np.size(pred,1), np.size(nifti_array,2)/np.size(pred,2))
    pred = ndimage.zoom(pred, factorUp, order=0)
    labels_nifti = nib.Nifti1Image(pred, nifti_data.affine)
    nib.save(labels_nifti, nifti_path.replace('.nii.gz','_labels_partial.nii.gz').replace('nifti','result_img'))
    nib.save(nifti_data, nifti_path.replace('.nii.gz','_original.nii.gz').replace('nifti','result_img'))

    # # convert res to numpy matrix
    res = np.array(res)

    # save the results to xlsx file for each patient as one raw in excel files
    results.loc[pat] = [nifti_file.replace('.nii.gz','')]+res.flatten().tolist()
    
results.to_excel(data_dir.replace('nifti','')+'results_intensity.xlsx', index=False)


# viewer = napari.Viewer()
# viewer.add_image(dataO, name='lung tissue', contrast_limits=[-1000, -500])
# viewer.add_labels(labels.astype(int), name='GMM whole lung')
# napari.run()

# viewer = napari.Viewer()
# viewer.add_image(vessels_mask)
# napari.run()
