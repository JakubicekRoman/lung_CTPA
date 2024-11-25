import os
import subprocess
import argparse
import sys
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
import pandas as pd

# Path to the folder containing DICOM files
parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', action='store',  help="Path to folder with input dicoms files")
parser.add_argument('-o','--output', action='store',  help="Path to ouput folder for results")

args = parser.parse_args()

if args.input is None:
    sys.exit("Input folder is not specified")

data_folder = args.input
if not os.path.exists(data_folder):
    sys.exit("Input folder does not exist")

if args.output is None:
    print("Output folder is not specified. New folder in the data folder will be created")
    # args.output will be the same as data_folder but in one level up
    args.output = os.path.join(os.path.dirname(data_folder), 'Results')


os.makedirs(args.output, exist_ok=True)

print("Input folder: ", data_folder)
print("Output folder: ", args.output)

# convertToNii = os.path.join(os.path.dirname(os.path.abspath(__file__)),'convertToNii.py')
# subprocess.run(['python', convertToNii, '-i', data_folder, '-o', args.output])

# lung_segm = os.path.join(os.path.dirname(os.path.abspath(__file__)),'lung_segm.py')
# subprocess.run(['python', lung_segm, '-i', args.output])

df = pd.DataFrame(columns=['Folder_name'])

df.insert(1, 'Patient Name', 'Patient Name')
df.insert(2, 'Patient ID', 'Patient ID')
df.insert(3, 'Accession No', 'Accession No')
df.insert(4, 'Date', 'Date')

df.to_excel(os.path.join( args.output , 'Outcomes.xlsx'), index=False)


# anal_1 = os.path.join(os.path.dirname(os.path.abspath(__file__)),'lung_segm.py')
# subprocess.run(['python', lung_segm, '-i', args.output])
