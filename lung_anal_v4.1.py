import os
import numpy as np
import nibabel as nib
# import napari
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
from utils import int_analyze, data_subsampling, lung_separate, predict_mask, to_flattened_array
import openpyxl
from scipy.stats import entropy

# data_dir = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\data1\nifti'
# data_dir = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\data3\nifti'

data_dir = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\data_all\nifti'

# create new folder
if not os.path.exists(data_dir.replace('nifti','result_img')):
    os.makedirs(data_dir.replace('nifti','result_img'))

# Get a list of all NIfTI files in the directory
nifti_files = [file for file in os.listdir(data_dir) if file.endswith('.nii.gz')]

results = pd.DataFrame(columns=['file', 'mdl_meanHypo', 'mdl_meanOligo', 'mdl_meanHyper',
                                'mdl_stdHypo', 'mdl_stdOligo', 'mdl_stdHyper',
                                'mdl_rateHypo', 'mdl_rateOligo', 'mdl_rateHyper',
                                'whole_meanHypo', 'whole_meanOligo', 'whole_meanHyper',
                                'whole_stdHypo', 'whole_stdOligo', 'whole_stdHyper',
                                'whole_rateHypo', 'whole_rateOligo', 'whole_rateHyper',
                                'Left_meanHypo', 'Left_meanOligo', 'Left_meanHyper',
                                'Left_stdHypo', 'Left_stdOligo', 'Left_stdHyper',
                                'Left_rateHypo', 'Left_rateOligo', 'Left_rateHyper',
                                'Right_meanHypo', 'Right_meanOligo', 'Right_meanHyper',
                                'Right_stdHypo', 'Right_stdOligo', 'Right_stdHyper',
                                'Right_rateHypo', 'Right_rateOligo', 'Right_rateHyper',
                                'SL_meanHypo', 'SL_meanOligo', 'SL_meanHyper',
                                'SL_stdHypo', 'SL_stdOligo', 'SL_stdHyper',
                                'SL_rateHypo', 'SL_rateOligo', 'SL_rateHyper',
                                'IL_meanHypo', 'IL_meanOligo', 'IL_meanHyper',
                                'IL_stdHypo', 'IL_stdOligo', 'IL_stdHyper',
                                'IL_rateHypo', 'IL_rateOligo', 'IL_rateHyper',
                                'SR_meanHypo', 'SR_meanOligo', 'SR_meanHyper',
                                'SR_stdHypo', 'SR_stdOligo', 'SR_stdHyper',
                                'SR_rateHypo', 'SR_rateOligo', 'SR_rateHyper',
                                'MR_meanHypo', 'MR_meanOligo', 'MR_meanHyper',
                                'MR_stdHypo', 'MR_stdOligo', 'MR_stdHyper',
                                'MR_rateHypo', 'MR_rateOligo', 'MR_rateHyper',
                                'IR_meanHypo', 'IR_meanOligo', 'IR_meanHyper',
                                'IR_stdHypo', 'IR_stdOligo', 'IR_stdHyper',
                                'IR_rateHypo', 'IR_rateOligo', 'IR_rateHyper'])


# nifti_files = [nifti_files[41]]    # export image for paper of one patient

factor = 0.5
# Iterate over the NIfTI files
for pat in range(53,54):
# for pat in range(0,len(nifti_files)):
    nifti_file = nifti_files[pat]
    print(nifti_file)
    print(str(pat/len(nifti_files)*100)+'%')

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
    gmW, val, indx = int_analyze(dataO, ((left_lung) | (right_lung)), vessels_mask, path_save)
    res.append(val.means_)
    res.append(np.sqrt(val.covariances_))
    res.append(val.weights_)

    pred = predict_mask(data, vessels_mask, ((left_lung) | (right_lung)), gmW, indx)
    factorUp = (np.size(nifti_array,0)/np.size(pred,0), np.size(nifti_array,1)/np.size(pred,1), np.size(nifti_array,2)/np.size(pred,2))
    pred = ndimage.zoom(pred, factorUp, order=0)
    labels_nifti = nib.Nifti1Image(pred, nifti_data.affine)
    nib.save(labels_nifti, nifti_path.replace('.nii.gz','_labels_whole.nii.gz').replace('nifti','result_img'))

    # # # statistics of whole/parts lung

    for part in range(0,3):
        if part==0:
            part_mask = pred*ndimage.zoom((left_lung) | (right_lung), factorUp, order=0)
        elif part==1:
            part_mask = pred*ndimage.zoom(left_lung, factorUp, order=0)
        else:
            part_mask = pred*ndimage.zoom(right_lung, factorUp, order=0)
        
        res.append(nifti_array[(part_mask)==3].mean())
        res.append(nifti_array[(part_mask)==2].mean())
        res.append(nifti_array[(part_mask)==1].mean())
        res.append(nifti_array[(part_mask)==3].std())
        res.append(nifti_array[(part_mask)==2].std())
        res.append(nifti_array[(part_mask)==1].std())
        res.append(np.sum((part_mask)==3) / np.sum(part_mask>0))
        res.append(np.sum((part_mask)==2) / np.sum(part_mask>0))
        res.append(np.sum((part_mask)==1) / np.sum(part_mask>0))

    # # partial of lung lobes
    parts = ['SL', 'IL', 'SR', 'MR', 'IR']
    k = 0
    for part in range(10,15):
        path_save = os.path.join(nifti_path.replace('.nii.gz','_'+parts[k]+'.png').replace('nifti','result_img'))
        part_mask = pred*ndimage.zoom(lung_mask==part, factorUp, order=0)
        res.append(nifti_array[(part_mask)==3].mean())
        res.append(nifti_array[(part_mask)==2].mean())
        res.append(nifti_array[(part_mask)==1].mean())
        res.append(nifti_array[(part_mask)==3].std())
        res.append(nifti_array[(part_mask)==2].std())
        res.append(nifti_array[(part_mask)==1].std())
        res.append(np.sum((part_mask)==3) / np.sum(part_mask>0))
        res.append(np.sum((part_mask)==2) / np.sum(part_mask>0))
        res.append(np.sum((part_mask)==1) / np.sum(part_mask>0))
        k+=1
        
    # factorUp = (np.size(nifti_array,0)/np.size(pred,0), np.size(nifti_array,1)/np.size(pred,1), np.size(nifti_array,2)/np.size(pred,2))
    # pred = ndimage.zoom(pred, factorUp, order=0)
    # labels_nifti = nib.Nifti1Image(pred, nifti_data.affine)
    # nib.save(labels_nifti, nifti_path.replace('.nii.gz','_labels_partial.nii.gz').replace('nifti','result_img'))
    nib.save(nifti_data, nifti_path.replace('.nii.gz','_original.nii.gz').replace('nifti','result_img'))

    # # convert res to numpy matrix
    res = to_flattened_array(res)

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
