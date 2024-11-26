import os
import subprocess
import argparse
import sys

# Path to the folder containing DICOM files
# parser = argparse.ArgumentParser()
# parser.add_argument('-i','--input', action='store',  help="Path to input dicom files")
# parser.add_argument('-o','--output', action='store',  help="Path to ouput folder")

# args = parser.parse_args()

# dicom_folder = args.input
# output_folder = args.output

def convert_to_nii(dicom_folder, output_folder):
    # dicom_folder = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\test_data\dicoms'
    # output_folder = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\test_data\Results'

    if not os.path.exists(dicom_folder):
        sys.exit("Input folder does not exist")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # find all folders in this path
    folders = [f for f in os.listdir(dicom_folder) if os.path.isdir(os.path.join(dicom_folder, f))]

    print("There are ", len(folders), " found cases")

    for folder in folders:
        # Run dcm2niix command to convert each DICOM in path folders to NIfTI
        subprocess.run(['dcm2niix.exe', '-z','y', '-f', folder , '-o', output_folder, 
                        os.path.join(dicom_folder, folder)])

    print("-------------------------")
    print("Converting to Nifti has been finished")
