import sys
import sys
import os
import matplotlib.pyplot as plt
import nibabel as nib
import subprocess

# Path to the folder containing DICOM files
# dicom_folder = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\data1\dicoms'
# dicom_folder = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\data2\dicoms'
dicom_folder = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\data3\dicoms'

# find all folders in this path
folders = [f for f in os.listdir(dicom_folder) if os.path.isdir(os.path.join(dicom_folder, f))]

# Path to the output folder for NIfTI files
output_folder = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\data3\nifti'
os.makedirs(output_folder, exist_ok=True)

for folder in folders:
    # Run dcm2niix command to convert each DICOM in path folders to NIfTI
    subprocess.run(['dcm2niix.exe', '-z','y', '-f', folder , '-o', output_folder, 
                    os.path.join(dicom_folder, folder)])

print("Resaving has been finished")
