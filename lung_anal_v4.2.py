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
import openpyxl
from scipy.stats import entropy
from skimage.feature import peak_local_max

results = pd.DataFrame(columns=['file', 'whole_1_entropy', 'whole_1_num_max', 'whole_1_max', 'whole_1_mean', 'whole_1_surf',
                                'whole_2_entropy', 'whole_2_num_max', 'whole_2_max', 'whole_2_mean', 'whole_2_surf',
                                'left_1_entropy', 'left_1_num_max', 'left_1_max', 'left_1_mean', 'left_1_surf',
                                'left_2_entropy', 'left_2_num_max', 'left_2_max', 'left_2_mean', 'left_2_surf',
                                'right_1_entropy', 'right_1_num_max', 'right_1_max', 'right_1_mean', 'right_1_surf',
                                'right_2_entropy', 'right_2_num_max', 'right_2_max', 'right_2_mean', 'right_2_surf'
                                ])

data_dir = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\result_img'

# Get a list of all NIfTI files in the directory
nifti_files = [file for file in os.listdir(data_dir) if file.endswith('original.nii.gz')]

factor = 0.5
for pat in [1]:
# for pat in range(0,len(nifti_files)):

    nifti_file = nifti_files[pat]
    print(nifti_file)

    res = []

    nifti_path = os.path.join(data_dir, nifti_file)
    nifti_data = nib.load(nifti_path)
    nifti_array = np.array(nifti_data.get_fdata())

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

    for part in [0,1,2]:
        if part==0:
            labels_1 = labels * (right_lung.copy() + left_lung.copy())
        elif part==1:
            labels_1 = labels * (right_lung.copy())
        elif part==2:
            labels_1 = labels * (left_lung.copy())
        

        for lab in [1,2]:
            # find coordinates of all one voxel of one and two label in labels matrix, but not label three
            loc = np.argwhere((labels_1 == lab))
            # create 3D histogram of unique values in loc
            n_bins = 32
            hist1 = np.histogramdd(loc, bins=[np.linspace(0, np.size(labels_1, 0), n_bins+1),np.linspace(0, np.size(labels_1, 1), n_bins+1),np.linspace(0, np.size(labels_1, 2), n_bins+1)],density=True)[0]
            ent02 = entropy(hist1.flatten())
            res.append(ent02)

            DistLable = distance_transform_edt((labels_1 == lab))
            # find local maxima in distance matrix
            
            loc = peak_local_max(DistLable, min_distance=2, labels=labels_1 == lab)

            # get distance values from distance matrix in local maxima position
            valsD = DistLable[(loc[:,0]),(loc[:,1]),(loc[:,2])] * (1/factor)

            # print(np.shape(loc))
            # print(np.min(valsD))
            # print(np.max(valsD))
            # print(np.mean(valsD))

            res.extend([np.shape(loc)[0], np.max(valsD), np.mean(valsD)])

            # compute ratio of surface and volume of the label
            surface =  ((DistLable<2) & (DistLable>0))
            surface[binary_dilation(vessels_mask, iterations=1)] = False
            res.append( np.sum( surface ) / np.sum( DistLable>0 ) )
            
    res = np.array(res)

    # save the results to xlsx file for each patient as one raw in excel files
    results.loc[pat] = [nifti_file.replace('_original.nii.gz','')]+res.flatten().tolist()
        
results.to_excel(data_dir.replace('\\result_img','')+'\\results_entrop.xlsx', index=False)


# viewer = napari.Viewer()
# viewer.add_image(DistLable, name='lung tissue')
# viewer.add_labels(labels_1.astype(int), name='GMM whole lung')
# napari.run()

 # compute entropy of 3D histogram
    # ent01 = np.sum(-hist1*np.log(hist1+1e-64))

    # # show in the same figure three marginal 1D histograms of the 3D matrix hist1 as sum along dimensions
    # plt.plot(hist1.sum(axis=(1,2)))
    # plt.plot(hist1.sum(axis=(0,2)))
    # plt.plot(hist1.sum(axis=(0,1)))
    # plt.show()

    # # show histogram without zero values
    # plt.hist(hist1[hist1>0].flatten(),bins=64)
    # plt.show()

    # # compute marginal entropy of 3D histogram
    # ent1 = entropy((hist1).sum(axis=(1,2)))
    # ent2 = entropy((hist1).sum(axis=(0,2)))
    # ent3 = entropy((hist1).sum(axis=(0,1)))

    # res.append(ent1)
    # res.append(ent2)
    # res.append(ent3)

# compute and save into variable 1D histogram of the first column of loc
# hist_1 = np.histogram(loc[:,0], bins=np.arange(0, 512, 1), density=True)[0]

# hist_2 = np.histogram(loc[:,0], bins=np.arange(0, 512, 10), density=True)[0]

# # show 1D  histogram of the first column of loc
# plt.hist(loc[:,0], bins=np.arange(0, 512, 10), density=True)
# plt.show()

# compute entropy only for non zero values
# ent1 = entropy(hist_1)
# ent2 = entropy(hist_2)

#     # save the results to xlsx file for each patient as one raw in excel files
#     results.loc[pat] = [nifti_file.replace('.nii.gz','')]+res.flatten().tolist()

# results.to_excel(data_dir.replace('nifti','')+'results.xlsx', index=False)

# viewer = napari.Viewer()
# viewer.add_image(dataO, name='lung tissue', contrast_limits=[-1000, -500])
# viewer.add_labels(labels.astype(int), name='GMM whole lung')
# napari.run()

# change size of matrix hist1 into the same size as lables matrix

# viewer = napari.Viewer()
# viewer.add_image(right_lung, name='lung tissue')
# napari.run()
