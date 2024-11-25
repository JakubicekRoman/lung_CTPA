import os
from totalsegmentator.python_api import totalsegmentator
import sys
import argparse

# Path to the folder containing DICOM files
parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', action='store',  help="Path to folder with input resaved nifti files")

args = parser.parse_args()

if args.input is None:
    sys.exit("Input folder does not specified")    
    
data_folder = args.input
if not os.path.exists(data_folder):
    sys.exit("Input folder does not exist")



# Get a list of all NIfTI files in the folder
nifti_files = [file for file in os.listdir(data_folder) if file.endswith('.nii.gz')]

print("Start lung segmentation")

for file in nifti_files:    
  
    file_dada_path = os.path.join(data_folder, file)
    file_output_path = args.input + os.sep + file.replace('.nii.gz','_segm.nii.gz')

    lung_parts = 'pulmonary_vein trachea lung_upper_lobe_left lung_upper_lobe_right lung_middle_lobe_right lung_lower_lobe_left lung_lower_lobe_right'
    # os.system('totalsegmentator -i {} -o {} --task {} --roi_subset {} --fast --verbose -ml'.format(file_dada_path, file_output_path, 'total',lung_parts))
    os.system('totalsegmentator -i {} -o {} --task {} --roi_subset {} --fast -ml'.format(file_dada_path, file_output_path, 'total',lung_parts))

print("Segmentation has been finished")

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


# example of usage:
# python lung_segm.py -i .\data\test_data\nifti -o .\data\test_data\lung_masks