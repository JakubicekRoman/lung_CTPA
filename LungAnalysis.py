import os
import argparse
import sys
import nibabel as nib
import numpy as np
import pandas as pd
from analysis import get_pat_info, lungSegmentation, convertToNii
from analysis import lung_analysis

# # Path to the folder containing DICOM files
parser = argparse.ArgumentParser()

parser.add_argument('-i','--input', action='store',  help="Path to folder with input dicoms files")
parser.add_argument('-o','--output', action='store',  help="Path to ouput folder for results")

args = parser.parse_args()

# args.input = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\test_data\dicoms'
# args.output = None

if args.input is None:
    sys.exit("Input folder is not specified")

data_folder = args.input
if not os.path.exists(data_folder):
    sys.exit("Input folder does not exist")

if args.output is None:
    print("Output folder is not specified. New folder in the data folder will be created")
    args.output = os.path.join(os.path.dirname(data_folder), 'Results')


os.makedirs(args.output, exist_ok=True)

print("Input folder: ", data_folder)
print("Output folder: ", args.output)

convertToNii(data_folder, args.output)

lungSegmentation(args.output)

get_pat_info(data_folder, args.output)

print("Lung analysis...")
lung_analysis(args.output)
print("Analysis has been completed")
