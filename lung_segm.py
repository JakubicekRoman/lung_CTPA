import os
import nibabel as nib
import napari
import matplotlib.pyplot as plt
from totalsegmentator.python_api import totalsegmentator
import sys

# Specify the folder path where the NIfTI files are located
folder_data_path = r'D:\Projekty\CTPA_VFN\data\nifti'
folder_mask_path = r'D:\Projekty\CTPA_VFN\data\masks\\'
os.makedirs(folder_mask_path, exist_ok=True)

# Get a list of all NIfTI files in the folder
nifti_files = [file for file in os.listdir(folder_data_path) if file.endswith('.nii.gz')]

for file in nifti_files:    
    file_dada_path = os.path.join(folder_data_path, file)
    file_output_path = folder_mask_path + file.replace('.nii.gz','_lung.nii.gz')

    # Call totalsegmentator as an executable file
    # os.system('totalsegmentator -h')

    lung_parts = 'pulmonary_vein trachea lung_upper_lobe_left lung_upper_lobe_right lung_middle_lobe_right lung_lower_lobe_left lung_lower_lobe_right'
    os.system('totalsegmentator -i {} -o {} --task {} --roi_subset {} --fast --verbose -ml'.format(file_dada_path, file_output_path, 'total',lung_parts))

print("Segm has been finished")

# # Select a slice index to display
# # Extract the slice from the NIfTI data
# slice_index = 50
# slice_data = nifti_data[:, :, slice_index]
# plt.imshow(slice_data, cmap='gray')
# plt.axis('off')
# plt.show()

# # Create a viewer
# viewer = napari.view_image(nifti_data)
# napari.run()
