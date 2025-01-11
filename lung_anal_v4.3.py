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
from skimage.measure import label

from utils import lung_separate, display_orthogonal_views, find_objects, vol_strel, def_perifer_map
import pandas as pd

# from skimage.morphology import isotropic_erosion


data_dir = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\data_all\result_img'

# Get a list of all NIfTI files in the directory
nifti_files = [file for file in os.listdir(data_dir) if file.endswith('original.nii.gz')]

results = pd.DataFrame(columns=['file',
                                'right_slope','right_rateHypo_Per', 'right_rateHypo_Mid','right_rateHypo_Cen',
                                'right_rateOligo_Per', 'right_rateOligo_Mid','right_rateOligo_Cen',
                                'right_rateHyper_Per', 'right_rateHyper_Mid','right_rateHyper_Cen',
                                'left_slope','left_rateHypo_Per', 'left_rateHypo_Mid','left_rateHypo_Cen',
                                'left_rateOligo_Per', 'left_rateOligo_Mid','left_rateOligo_Cen',
                                'left_rateHyper_Per', 'left_rateHyper_Mid','left_rateHyper_Cen',
                                'SL_slope','SL_rateHypo_Per', 'SL_rateHypo_Mid','SL_rateHypo_Cen',
                                'SL_rateOligo_Per', 'SL_rateOligo_Mid','SL_rateOligo_Cen',
                                'SL_rateHyper_Per', 'SL_rateHyper_Mid','SL_rateHyper_Cen',
                                'IL_slope','IL_rateHypo_Per', 'IL_rateHypo_Mid','IL_rateHypo_Cen',
                                'IL_rateOligo_Per', 'IL_rateOligo_Mid','IL_rateOligo_Cen',
                                'IL_rateHyper_Per', 'IL_rateHyper_Mid','IL_rateHyper_Cen',
                                'SR_slope','SR_rateHypo_Per', 'SR_rateHypo_Mid','SR_rateHypo_Cen',
                                'SR_rateOligo_Per', 'SR_rateOligo_Mid','SR_rateOligo_Cen',
                                'SR_rateHyper_Per', 'SR_rateHyper_Mid','SR_rateHyper_Cen',
                                'MR_slope','MR_rateHypo_Per', 'MR_rateHypo_Mid','MR_rateHypo_Cen',
                                'MR_rateOligo_Per', 'MR_rateOligo_Mid','MR_rateOligo_Cen',
                                'MR_rateHyper_Per', 'MR_rateHyper_Mid','MR_rateHyper_Cen',
                                'IR_slope','IR_rateHypo_Per', 'IR_rateHypo_Mid','IR_rateHypo_Cen',
                                'IR_rateOligo_Per', 'IR_rateOligo_Mid','IR_rateOligo_Cen',
                                'IR_rateHyper_Per', 'IR_rateHyper_Mid','IR_rateHyper_Cen'                                
                                ])

results_Loc = results.copy()

for pat in range(0,len(nifti_files)):
# for pat in range(0,42):
# for pat in [0,1]:
    nifti_file = nifti_files[pat]
    print(nifti_file)

    factor = 0.5
    res = []

    nifti_path = os.path.join(data_dir, nifti_file)
    nifti_data = nib.load(nifti_path)
    nifti_array = np.array(nifti_data.get_fdata())

    velOrig = np.shape(nifti_array)

    # Load the corresponding lung mask
    lung_mask_file = nifti_file.replace('original.nii.gz', 'lung.nii.gz')
    lung_mask_path = os.path.join(data_dir.replace('\\result_img', '\\masks'), lung_mask_file)
    lung_mask_data = nib.load(lung_mask_path)
    lung_mask_array = np.array(lung_mask_data.get_fdata())

    labels = nib.load(data_dir+os.sep+nifti_file.replace('original.nii', 'labels_whole.nii')).get_fdata()

    nifti_array = ndimage.zoom(nifti_array, factor, order=0)
    lung_mask_array = ndimage.zoom(lung_mask_array, factor, order=0)
    labels = ndimage.zoom(labels, factor, order=0)

    left_lung, right_lung, trachea, vessels_mask = lung_separate(nifti_array, lung_mask_array)

    pulmonary_vein = (lung_mask_array == 53)
    m = np.max(label(pulmonary_vein,connectivity=1), axis=(0,1,2))
    # print(m)

    for i in range(0,20):
        if m > 2:
            pulmonary_vein = binary_dilation(pulmonary_vein, iterations=1, structure=vol_strel())
            m = np.max(label(pulmonary_vein,connectivity=1), axis=(0,1,2))
            # print(m)
        else:
            break
    
    if np.max(label(pulmonary_vein,connectivity=1), axis=(0,1,2)) == 1:
        pulmonary_vein = (lung_mask_array == 53)
        m = np.max(label(pulmonary_vein,connectivity=1), axis=(0,1,2))
        for i in range(0,20):
            if m > 3:
                pulmonary_vein = binary_dilation(pulmonary_vein, iterations=1, structure=vol_strel())
                m = np.max(label(pulmonary_vein,connectivity=1), axis=(0,1,2))
                # print(m)
            else:
                pulmonary_vein = find_objects(pulmonary_vein, num_objects=2)
                break

    pulmonary_vein = label(pulmonary_vein,connectivity=1)

    # pulmonary_vein = binary_dilation(pulmonary_vein, iterations=7, structure=morph_anal.vol_strel())
    # pulmonary_vein = binary_erosion(pulmonary_vein, iterations=5)
    # find two largest binary objects in the pulmonary_vein mask and remain only them
    # pulmonary_vein = morph_anal.find_objects(pulmonary_vein, num_objects=2)

    positions1 = np.mean(np.argwhere(pulmonary_vein == 1), axis=0)
    positions2 = np.mean(np.argwhere(pulmonary_vein == 2), axis=0)

    positionsR = np.mean(np.argwhere(right_lung == 1), axis=0)
    # positionsL = np.mean(np.argwhere(left_lung == 1), axis=0)
    d1, d2 = np.linalg.norm(positions1-positionsR), np.linalg.norm(positions2-positionsR)
    if d1 > d2:
        positions1, positions2 = positions2, positions1

    for meth in ['Loc', 'Glob']:
    # meth = 'Glob'
        res = []

        bin2, bin3 = def_perifer_map(lung_mask_array, positions1, positions2, local=(meth == 'Loc'))

        # display_orthogonal_views(bin3.astype(float), slice_index=np.mean(np.argwhere(mask_part), axis=0).astype(int))

        # factorUp = (np.size(nifti_array,0)/np.size(bin2,0), np.size(nifti_array,1)/np.size(bin2,1), np.size(nifti_array,2)/np.size(bin2,2))
        # bin2 = ndimage.zoom(bin2, factorUp, order=0)
        # lung_mask_data = nib.load(lung_mask_path)
        # lung_mask_array = np.array(lung_mask_data.get_fdata())
        # left_lung, right_lung, trachea, vessels_mask = lung_separate(nifti_array, lung_mask_array)

        # viewer = napari.Viewer()
        # viewer.add_image(bin3)
        # napari.run()

        for part in [0,1,2,3,4,5,6]:
        # for part in [2,3,4,5,6]:
            if part==0:
                mask_part = right_lung.copy()
            elif part==1:
                mask_part = left_lung.copy()
            elif part==2:
                mask_part = lung_mask_array==10
            elif part==3:
                mask_part = lung_mask_array==11
            elif part==4:
                mask_part = lung_mask_array==12
            elif part==5:
                mask_part = lung_mask_array==13
            elif part==6:
                mask_part = lung_mask_array==14

            valMean = np.zeros(14)
            for lbl in np.arange(1, 15):
                valMean[lbl-1] = np.mean(nifti_array[ (bin2==lbl) & ((~vessels_mask) & mask_part) ])

            valMean = valMean[~np.isnan(valMean)]
            valMean = valMean[::-1]
            ind2 = np.linspace(0,1,len(valMean))[~np.isnan(valMean)]
            slope, bias = np.polyfit(ind2, valMean, 1)

            # plt.ion()
            # plt.plot(ind2, valMean, '+')
            # # plt.plot(ind2, valM2, 'b')
            # # show fitted line in the plot
            # x = np.linspace(0,1,100)
            # y = slope*x + bias
            # plt.plot(x, y, 'r' )
            # plt.xlabel('Relative distance from hilum to pleura')
            # plt.ylabel('Mean Attenuation [HU] in contour')
            # # plt.ylim(-950, -500)
            # plt.show()
            # # plt.savefig(nifti_path.replace('_original.nii.gz','_slope'+ str(part) + '.png'), format='png')
            # plt.close()

            res.append( slope )
            for label_value in [3, 2, 1]:
                for region in [1, 2, 3]:
                    res.append(np.nanmean(np.sum(labels[(bin3 * mask_part) == region] == label_value) / np.sum(labels[(bin3 * mask_part) == region] > 0)))
            

        # if part==1:
        plot_name = nifti_path.replace('_original.nii.gz','_regions_LR_' + meth + '.png')
        display_orthogonal_views(bin3.astype(int), slice_index=np.mean(np.argwhere(right_lung), axis=0).astype(int), save_path=plot_name)
        factorUp = (velOrig[0]/np.size(bin3,0), velOrig[1]/np.size(bin3,1), velOrig[2]/np.size(bin3,2))
        bin3 = ndimage.zoom(bin3, factorUp, order=0)
        periph_nifti = nib.Nifti1Image(bin3, nifti_data.affine)
        nib.save(periph_nifti, nifti_path.replace('_original.nii.gz','_regions_LR_' + meth + '.nii.gz'))

        # elif part==6:
            # plot_name = nifti_path.replace('_original.nii.gz','_regions_lobes.png')            
            # display_orthogonal_views(bin3.astype(int), slice_index=np.mean(np.argwhere(right_lung), axis=0).astype(int), save_path=plot_name)
            # factorUp = (velOrig[0]/np.size(bin3,0), velOrig[1]/np.size(bin3,1), velOrig[2]/np.size(bin3,2))
            # bin3 = ndimage.zoom(bin2, factorUp, order=0)
            # periph_nifti = nib.Nifti1Image(bin3, nifti_data.affine)
            # # nib.save(periph_nifti, nifti_path.replace('_original.nii.gz','_regions_lobes.nii.gz'))
            # bin3 = np.zeros_like(nifti_array, dtype=np.bool_)

        res = np.array(res)

        if meth == 'Loc':
            results_Loc.loc[pat] = [nifti_file.replace('_original.nii.gz','')]+res.flatten().tolist()  
            results_Loc.to_excel(data_dir.replace('\\result_img','')+'\\results_periphery_'+meth+'.xlsx', index=False)
        else:
            results.loc[pat] = [nifti_file.replace('_original.nii.gz','')]+res.flatten().tolist()  
            results.to_excel(data_dir.replace('\\result_img','')+'\\results_periphery_'+meth+'.xlsx', index=False)


# viewer = napari.Viewer()
# viewer.add_image((labels==1) | (left_lung>0), name='lung tissue')
# viewer.add_image((hylL)*2 + (left_lung>0), name='lung tissue')
# viewer.add_image((hylL) , name='lung tissue')
# napari.run()

# viewer = napari.Viewer()
# viewer.add_image(pulmonary_vein, name='CT', contrast_limits=[-1000, -500])
# napari.run()

# viewer = napari.Viewer()
# viewer.add_image(nifti_array, name='CT', contrast_limits=[-1000, -500])
# viewer.add_labels(bin2.astype(int), name='region')
# napari.run()


