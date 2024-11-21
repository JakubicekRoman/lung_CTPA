import os
import subprocess
import argparse

# Path to the folder containing DICOM files
parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', action='store',  help="Path to input dicom files")
parser.add_argument('-o','--output', action='store',  help="Path to ouput folder")

args = parser.parse_args()

dicom_folder = args.input
output_folder = args.output

# find all folders in this path
folders = [f for f in os.listdir(dicom_folder) if os.path.isdir(os.path.join(dicom_folder, f))]

# Path to the output folder for NIfTI files
os.makedirs(output_folder, exist_ok=True)

print("Start resaving DICOM files to NIfTI")
print("Input folder: ", dicom_folder)
print("Output folder: ", output_folder)
print("there are ", len(folders), " found folders")

for folder in folders:
    # Run dcm2niix command to convert each DICOM in path folders to NIfTI
    subprocess.run(['dcm2niix.exe', '-z','y', '-f', folder , '-o', output_folder, 
                    os.path.join(dicom_folder, folder)])

print("-------------------------")
print("Resaving has been finished")

