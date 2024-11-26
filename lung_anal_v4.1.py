import os
# import numpy as np
import nibabel as nib
# from scipy.ndimage import median_filter
import pandas as pd
# from scipy import ndimage
# from utils import int_analyze, data_subsampling, lung_separate, predict_mask
# import openpyxl
import sys
import argparse

print('Lung analysis started')

# Path to the folder containing DICOM files
parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', action='store',  help="Path to folder with input resaved nifti files")

args = parser.parse_args()

if args.input is None:
    sys.exit("Input folder does not specified")    
    
data_folder = args.input
if not os.path.exists(data_folder):
    sys.exit("Input folder does not exist")


# Get a list of all NIfTI files in the directory
nifti_files = [file for file in os.listdir(args.input) if file.endswith('_segm.nii.gz')]

# read excel file Outcomes.xlsx from arg.output folder                        
output_folder = os.path.join(data_folder, 'Outcomes.xlsx')
if not os.path.exists(output_folder):
    sys.exit("Outcomes.xlsx file does not exist in the input folder")

df = pd.read_excel(output_folder)

factor = 0.5
# Iterate over the NIfTI files
# for pat in range(2,3):
for pat in range(0,len(nifti_files)):
    nifti_file = nifti_files[pat]
    print(nifti_file)
    print(str(pat/len(nifti_files)*100)+'%')

    nifti_path = os.path.join(data_folder, nifti_file)
    nifti_data = nib.load(nifti_path)
    nifti_array = np.array(nifti_data.get_fdata())

    # Load the corresponding lung mask
    lung_mask_file = nifti_file.replace('.nii', '_segm.nii')
    lung_mask_path = os.path.join(data_folder, lung_mask_file).replace('nifti', 'masks')
    lung_mask_data = nib.load(lung_mask_path)
    lung_mask_array = np.array(lung_mask_data.get_fdata())

    dataO, lung_mask = data_subsampling(nifti_array, lung_mask_array, factor=factor)

    left_lung, right_lung, trachea, vessels_mask = lung_separate(dataO, lung_mask)

    data = median_filter(dataO, size=5)
    data = data*(left_lung | right_lung)
    res=[]

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

    df.insert(6, 'Int_1_mean', res[0][0])
    df.insert(7, 'Int_1_cov', res[1][0])
    df.insert(8, 'Int_1_weight', res[2][0])
    df.insert(9, 'Int_2_mean', res[0][1])
    df.insert(10, 'Int_2_cov', res[1][1])
    df.insert(11, 'Int_2_weight', res[2][1])
    df.insert(12, 'Int_3_mean', res[0][2])
    df.insert(13, 'Int_3_cov', res[1][2])
    df.insert(14, 'Int_3_weight', res[2][2])
    
df.to_excel(output_folder, index=False)

# viewer = napari.Viewer()
# viewer.add_image(dataO, name='lung tissue', contrast_limits=[-1000, -500])
# viewer.add_labels(labels.astype(int), name='GMM whole lung')
# napari.run()

# viewer = napari.Viewer()
# viewer.add_image(vessels_mask)
# napari.run()
