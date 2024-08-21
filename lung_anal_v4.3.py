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
from utils import lung_separate, display_orthogonal_views
import pandas as pd

# from skimage.morphology import isotropic_erosion

class morph_anal:
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
    
    # find largest objects in the mask
    def find_objects( mask, num_objects=1):
        labels = label(mask,connectivity=1)
        labels_reduced = np.zeros_like(labels)
        vel = np.zeros(np.max(labels))
        for i in range(1, np.max(labels) + 1):
            vel[i-1] = np.sum(labels == i)
        vel = np.argsort(vel)[::-1]+1
        for i in range(0, num_objects):
            labels_reduced = labels_reduced + ((labels == (vel[i]))*(i+1))
        return labels_reduced
    
    def vol_strel():
        structure=np.ones((3,3,3), dtype=bool)
        structure[0,0,0] = False
        structure[0,0,2] = False
        structure[0,2,0] = False
        structure[0,2,2] = False
        structure[2,0,0] = False
        structure[2,0,2] = False
        structure[2,2,0] = False
        structure[2,2,2] = False
        return structure


data_dir = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\data2\result_img'

# Get a list of all NIfTI files in the directory
nifti_files = [file for file in os.listdir(data_dir) if file.endswith('original.nii.gz')]

results = pd.DataFrame(columns=['file','right_slope','right_rate_L1_1','right_rate_L1_2','right_rate_L1_3','right_rate_L2_1','right_rate_L2_2','right_rate_L2_3',
                                'left_slope','left_rate_L1_1','left_rate_L1_2','left_rate_L1_3','left_rate_L2_1','left_rate_L2_2','left_rate_L2_3',
                                'SL_slope','SL_rate_L1_1','SL_rate_L1_2','SL_rate_L1_3','SL_rate_L2_1','SL_rate_L2_2','SL_rate_L2_3',
                                'IL_slope','IL_rate_L1_1','IL_rate_L1_2','IL_rate_L1_3','IL_rate_L2_1','IL_rate_L2_2','IL_rate_L2_3',
                                'SR_slope','SR_rate_L1_1','SR_rate_L1_2','SR_rate_L1_3','SR_rate_L2_1','SR_rate_L2_2','SR_rate_L2_3',
                                'MR_slope','MR_rate_L1_1','MR_rate_L1_2','MR_rate_L1_3','MR_rate_L2_1','MR_rate_L2_2','MR_rate_L2_3',
                                'IR_slope','IR_rate_L1_1','IR_rate_L1_2','IR_rate_L1_3','IR_rate_L2_1','IR_rate_L2_2','IR_rate_L2_3'
                                ])
# Iterate over the NIfTI files
# for nifti_file in nifti_files:

for pat in range(0,len(nifti_files)):
# for pat in range(0,2):
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
            pulmonary_vein = binary_dilation(pulmonary_vein, iterations=1, structure=morph_anal.vol_strel())
            m = np.max(label(pulmonary_vein,connectivity=1), axis=(0,1,2))
            # print(m)
        else:
            break
    
    if np.max(label(pulmonary_vein,connectivity=1), axis=(0,1,2)) == 1:
        pulmonary_vein = (lung_mask_array == 53)
        m = np.max(label(pulmonary_vein,connectivity=1), axis=(0,1,2))
        for i in range(0,20):
            if m > 3:
                pulmonary_vein = binary_dilation(pulmonary_vein, iterations=1, structure=morph_anal.vol_strel())
                m = np.max(label(pulmonary_vein,connectivity=1), axis=(0,1,2))
                # print(m)
            else:
                pulmonary_vein = morph_anal.find_objects(pulmonary_vein, num_objects=2)
                break

    pulmonary_vein = label(pulmonary_vein,connectivity=1)

    # pulmonary_vein = binary_dilation(pulmonary_vein, iterations=7, structure=morph_anal.vol_strel())
    # pulmonary_vein = binary_erosion(pulmonary_vein, iterations=5)
    # find two largest binary objects in the pulmonary_vein mask and remain only them
    # pulmonary_vein = morph_anal.find_objects(pulmonary_vein, num_objects=2)

    positions1 = np.mean(np.argwhere(pulmonary_vein == 1), axis=0)
    positions2 = np.mean(np.argwhere(pulmonary_vein == 2), axis=0)

    # third number of position2 (in Z axis) must be lower then third number of position1
    if positions1[0] > positions2[0]:
        positions1, positions2 = positions2, positions1

    bin2 = np.zeros_like(nifti_array, dtype=np.bool_)

    for part in [0,1,2,3,4,5,6]:
    # for part in [0,1]:
    # for part in [2,3,4,5,6]:
        if part==0:
            mask_part = right_lung.copy()
            H = positions1.copy()
            max_H = 0.45 # vetsi znamena posouvani k hilu
            dil_contr = 20
        elif part==1:
            mask_part = left_lung.copy()
            H = positions2.copy()
            max_H = 0.45
            dil_contr = 20
        elif part==2:
            mask_part = lung_mask_array==10
            H = positions2.copy()
            max_H = 0.35
            dil_contr = 15
        elif part==3:
            mask_part = lung_mask_array==11
            H = positions2.copy()
            max_H = 0.35
            dil_contr = 15
        elif part==4:
            mask_part = lung_mask_array==12
            H = positions1.copy()
            max_H = 0.35
            dil_contr = 20
        elif part==5:
            mask_part = lung_mask_array==13
            H = positions1.copy()
            max_H = 0.3
            dil_contr = 15
        elif part==6:
            mask_part = lung_mask_array==14
            H = positions1.copy()
            max_H = 0.35
            dil_contr = 15

        hyl = np.zeros_like(mask_part, dtype=np.bool_)
        hyl[int(H[0]), int(H[1]), int(H[2])] = True
        dist_map_H = distance_transform_edt(~hyl)

        # create contour of lung mask
        contour = mask_part & ~binary_erosion(mask_part > 0, iterations=1)
        contour = (contour > 0) & ~(binary_dilation((lung_mask_array == 53), iterations=dil_contr,structure=morph_anal.vol_strel()) > 0)
        # contour = contour * ~(binary_dilation((nifti_array < 550) & (nifti_array > 450), iterations=1,structure=morph_anal.vol_strel()))

        # contour = morph_anal.find_objects(contour, num_objects=1)
        dist_map_Contr = distance_transform_edt(~contour)
        dist_map = ( ( dist_map_Contr ) / ( dist_map_Contr + dist_map_H ) ) * mask_part

# viewer = napari.Viewer()
# viewer.add_image(dist_map)
# napari.run()

        num = 11
        steps_thr = np.linspace(0,max_H,num)
        step = steps_thr[1] - steps_thr[0]

        valM = np.zeros(int(num))
        valL1 = np.zeros(int(num))
        valL2 = np.zeros(int(num))
        # valS = np.zeros(int(num))
        i = 0; lab = 1
        # bin2 = np.zeros_like(mask_part, dtype=np.bool_)
        for thr in steps_thr:
        # for thr in steps_thr[0:3]:
            bin = (dist_map > thr) & (dist_map < (thr + step))
            # display_orthogonal_views(bin.astype(float), slice_index=np.mean(np.argwhere(left_lung), axis=0).astype(int))
            valM[i] = np.mean(nifti_array[ bin & (~vessels_mask) ])
            # valS[i] = np.std(nifti_array[ bin & (~vessels_mask) ])
            valL1[i] = np.sum( labels[ bin & (~vessels_mask) ]==1 )/ np.sum( labels[ bin & (~vessels_mask) ]>0 ) 
            valL2[i] = np.sum( labels[ bin & (~vessels_mask) ]==2 )/ np.sum( labels[ bin & (~vessels_mask) ]>0 )             
            i += 1

            bin2 = bin2 + (bin*lab)
            if i == 3:
                # display_orthogonal_views(bin2.astype(float), slice_index=H.astype(int))
                # display_orthogonal_views(bin2.astype(float), slice_index=np.mean(np.argwhere(mask_part), axis=0).astype(int))
                # bin2 = np.zeros_like(mask_part, dtype=np.bool_)
                lab = 2
            elif i == 6:
                # display_orthogonal_views(bin2.astype(float), slice_index=H.astype(int))
                # display_orthogonal_views(bin2.astype(float), slice_index=np.mean(np.argwhere(mask_part), axis=0).astype(int))
                # bin2 = np.zeros_like(mask_part, dtype=np.bool_)
                lab = 3
            # elif i == 11:
                # display_orthogonal_views(bin2.astype(float), slice_index=H.astype(int))
                # display_orthogonal_views(bin2.astype(float), slice_index=np.mean(np.argwhere(mask_part), axis=0).astype(int))
                # bin2 = np.zeros_like(mask_part, dtype=np.bool_)

        # display_orthogonal_views(bin2.astype(float), slice_index=np.mean(np.argwhere(mask_part), axis=0).astype(int))


            # plt.savefig(file_path, format='png')
            # plt.close()
        
        # display_orthogonal_views(bin2.astype(float), slice_index=np.mean(np.argwhere(left_lung), axis=0).astype(int))

        # plt.plot(np.linspace(0,1,len(valM)), valM)
        # plt.show()
        # plt.plot(np.linspace(0,1,len(valS)), valS)
        # plt.show()
        # plt.plot(np.linspace(0,1,len(valL1)), valL1)
        # plt.plot(np.linspace(0,1,len(valL2)), valL2)
        # plt.show()

        # plt.imshow(bin[:,:,int(H[2])])
        # plt.show()

        # fit linear curve to the data valM and give me the slope and ignore nan values
        valM2 = valM[~np.isnan(valM)]
        ind2 = np.linspace(0,1,len(valM))[~np.isnan(valM)]
        slope, bias = np.polyfit(ind2, valM2, 1)

        # plt.plot(ind2, valM2)
        # # show fitted line in the plot
        # x = np.linspace(0,1,100)
        # y = slope*x + bias
        # plt.plot(x, y)
        # plt.xlabel('Relative distance from hilus')
        # plt.ylabel('Mean intensity in contour')
        # # plt.ylim(-830, -650)
        # plt.show()

        res.append( slope )
        res.append( np.nanmean(valL1[0:3]) )
        res.append( np.nanmean(valL1[3:6]) )
        res.append( np.nanmean(valL1[7:12]) )
        res.append( np.nanmean(valL2[0:3]) )
        res.append( np.nanmean(valL2[3:8]) )
        res.append( np.nanmean(valL2[7:12]) )

        if part==1:
            plot_name = nifti_path.replace('_original.nii.gz','_regions_LR.png')
            display_orthogonal_views(bin2.astype(int), slice_index=np.mean(np.argwhere(right_lung), axis=0).astype(int), save_path=plot_name)
            factorUp = (velOrig[0]/np.size(bin2,0), velOrig[1]/np.size(bin2,1), velOrig[2]/np.size(bin2,2))
            bin2 = ndimage.zoom(bin2, factorUp, order=0)
            periph_nifti = nib.Nifti1Image(bin2, nifti_data.affine)
            nib.save(periph_nifti, nifti_path.replace('_original.nii.gz','_regions_LR.nii.gz'))
            bin2 = np.zeros_like(nifti_array, dtype=np.bool_)
        elif part==6:
            plot_name = nifti_path.replace('_original.nii.gz','_regions_lobes.png')            
            display_orthogonal_views(bin2.astype(int), slice_index=np.mean(np.argwhere(right_lung), axis=0).astype(int), save_path=plot_name)
            factorUp = (velOrig[0]/np.size(bin2,0), velOrig[1]/np.size(bin2,1), velOrig[2]/np.size(bin2,2))
            bin2 = ndimage.zoom(bin2, factorUp, order=0)
            periph_nifti = nib.Nifti1Image(bin2, nifti_data.affine)
            nib.save(periph_nifti, nifti_path.replace('_original.nii.gz','_regions_lobes.nii.gz'))
            bin2 = np.zeros_like(nifti_array, dtype=np.bool_)

    res = np.array(res)
    results.loc[pat] = [nifti_file.replace('_original.nii.gz','')]+res.flatten().tolist()


results.to_excel(data_dir.replace('\\result_img','')+'\\results_periphery.xlsx', index=False)


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


