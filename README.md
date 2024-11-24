# Pulmonary perfusion changes quantification
This work aimed to develop an automated method for quantifying the distribution and severity of perfusion changes on CT pulmonary angiography (CPTA) in patients with chronic thromboembolic pulmonary hypertension (CTEPH)

## Description
For now, only vessel segmentation is included.
A tool for the vessel segmentation in AO images based on segmentation neural network nnUNet second version.

General information about this tool:

* It works for folder containing dicom files
* The algorithm has three basic steps
  * resaving dicom files to nifti format using dcm2niix [1]
  * lung segmentation using TotalSegmentator [2]
  * automated quantification of lung mosaic:
    * intensity analysis
    * heterogeneity analysis
    * centralization analysis
* output is an excel file with characterization values for each patient folder

## Requirements
* virtual environment
* installed python (tested on 3.10 version)
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

Install python (if not already):
download [**here**](https://www.python.org/downloads/)

```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
pip install virtualenv
```

Create virtual environment:
```
python3.10 -m venv ".\.venv"
```

Activate venv
```
.\.venv\Scripts\activate
```

Install the required packages from .txt file:
```
python3 -m pip install -r requirements.txt
```

## Data directory structure
All data needs to be **dicom** files and one patient (case) needs to be in an single folder 
Example of directory structure of two patient data:
```
... PATH_DICOM_DATA\

+---S44670 
|   +---S40
|   |       DIRFILE
|   |       I10
|   |       I100
|   |       I100
|   |       ...
+---S44660
|      DIRFILE
|      I10
|      I100
|      ...
...
```

## Calling of the program:

Call Ii the command window with activated virtual enviroment:

```
python3 LungAnalysis.py -h
python3 LungAnalysis.py --input folder_with_dicoms --output folder_for_saving
```

Example of calling:
```
python3 LungAnalysis.py --input .\data\dicoms --output .\data\results
python3 LungAnalysis.py --input .\data\dicoms
```

## Licence
The tool is possible to use for academic and reseach purposes (MIT License). The proposed approach was submitted to ... 
Please cite the following paper when using tool:
---- It will be added!

## References
[1] Chris Rorden. "dcm2niix: Next-generation DICOM to NIfTI conversion." (2022). 
Available from: https://github.com/rordenlab/dcm2niix

[2] Wasserthal, Jakob et al. “TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images.” Radiology. Artificial intelligence vol. 5,5 e230024. 5 Jul. 2023, doi:10.1148/ryai.230024
GitHub: https://github.com/wasserth/TotalSegmentator
