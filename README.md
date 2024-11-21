# Pulmonary perfusion changes quantification
This work aimed to develop an automated method for quantifying the distribution and severity of perfusion changes on CT pulmonary angiography (CPTA) in patients with chronic thromboembolic pulmonary hypertension (CTEPH)

## Description
For now, only vessel segmentation is included.
A tool for the vessel segmentation in AO images based on segmentation neural network nnUNet second version.

General information about this tool:

* It works with folder containing png images
* A "Input path" to a folder containing png files is required as input
* Output segmentation masks are saved into user's given "Output path" as set of png images


## Requirements
* virtual environment
* python version 3.10
* installed pip and venv

## Virtual environment
in the terminal:
* clone git repository from github
```
git clone https://github.com/JakubicekRoman/lung_CTPA.git
```
* set current folder of lung_CTPA in the terminal
```
cd .\lung_CTPA\
```
* for PIP installation, check Python version (major version ```python3 --version``` and all installed versions ```ls -ls /usr/bin/python*```)

Install python, pip and venv (if not already):
```
sudo apt install python3.10
sudo apt install python3-pip
sudo apt install python3.10-venv
```

Create virtual environment:
```
python3.10 -m venv ".\.venv\lung_CTPA"
```

Activate venv
```
.\.venv\lung_CTPA\bin\activate
```

Install the packages from .txt file:
```
python3 -m pip install -r requirements.txt
```

## Calling of the program:

### 1st step
Converting dicom data to nifti files (fucntion calling dcm2niix). You may call dcm2niix directly, but this function extracts also patient's information from dicom header and save that into results excel file. If you have own nifti files, the results excel file will not contain patient info, only name of folder (series)
```
python3 covnertToNii.py -h
python3 covnertToNii.py --input folder_with_dicoms --output folder_for_saving
```

Example of calling:
```

```

### 2nd step

