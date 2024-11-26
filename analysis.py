import os
import numpy as np
import nibabel as nib
from scipy.ndimage import median_filter
import pandas as pd
from scipy import ndimage
from utils import int_analyze, data_subsampling, lung_separate, predict_mask, morph_anal, central_analysis
from scipy.stats import entropy
from scipy.ndimage import binary_dilation
import openpyxl
import pydicom
import glob
import sys
import subprocess
from skimage.measure import label


def lung_analysis(data_folder):

    # Get a list of all NIfTI files in the directory
    nifti_files = [file for file in os.listdir(data_folder) if file.endswith('_segm.nii.gz')]

    # read excel file Outcomes.xlsx from arg.output folder                        
    output_folder = os.path.join(data_folder, 'Outcomes.xlsx')
    if not os.path.exists(output_folder):
        sys.exit("Outcomes.xlsx file does not exist in the input folder")

    df = pd.read_excel(output_folder)

    factor = 0.5
    # i = df.shape[0]+1
    # Iterate over the NIfTI files
    for pat in range(0,len(nifti_files)):
        nifti_file = nifti_files[pat].replace('_segm.nii.gz','.nii.gz')
        print("Process: " + str(pat/len(nifti_files)*100)+'% - ' + nifti_file.replace('.nii.gz',''))

        nifti_path = os.path.join(data_folder, nifti_file)
        nifti_data = nib.load(nifti_path)
        nifti_array = np.array(nifti_data.get_fdata())

        # Load the corresponding lung mask
        lung_mask_file = nifti_file.replace('.nii.gz', '_segm.nii.gz')
        lung_mask_path = os.path.join(data_folder, lung_mask_file)
        lung_mask_data = nib.load(lung_mask_path)
        lung_mask_array = np.array(lung_mask_data.get_fdata())

        dataO, lung_mask = data_subsampling(nifti_array, lung_mask_array, factor=factor)

        left_lung, right_lung, trachea, vessels_mask = lung_separate(dataO, lung_mask)

        data = median_filter(dataO, size=5)
        data = data*(left_lung | right_lung)
        res=[]

        # global intensity analysis
        path_save = os.path.join(nifti_path.replace('.nii.gz','_distrib.png'))
        gmW, val = int_analyze(data, ((left_lung) | (right_lung)), vessels_mask, path_save)
        res.append(val.means_)
        res.append(np.sqrt(val.covariances_))
        res.append(val.weights_)

        pred = predict_mask(data, vessels_mask, ((left_lung) | (right_lung)), gmW)

        # entropy analysis
        labels_1 = pred * (right_lung.copy() + left_lung.copy())
        n_bins = 32

        loc = np.argwhere((labels_1 == 1))
        hist1 = np.histogramdd(loc, bins=[np.linspace(0, np.size(labels_1, 0), n_bins+1),np.linspace(0, np.size(labels_1, 1), n_bins+1),np.linspace(0, np.size(labels_1, 2), n_bins+1)],density=True)[0]
        ent01 = entropy(hist1.flatten())

        loc = np.argwhere((labels_1 == 2))
        hist2 = np.histogramdd(loc, bins=[np.linspace(0, np.size(labels_1, 0), n_bins+1),np.linspace(0, np.size(labels_1, 1), n_bins+1),np.linspace(0, np.size(labels_1, 2), n_bins+1)],density=True)[0]
        ent02 = entropy(hist2.flatten())

        # centralization analysis

        pulmonary_vein = (lung_mask == 53)
        m = np.max(label(pulmonary_vein,connectivity=1), axis=(0,1,2))
        # print(m)

        for i in range(0,20):
            if m > 2:
                pulmonary_vein = binary_dilation(pulmonary_vein, iterations=1, structure=morph_anal.vol_strel())
                m = np.max(label(pulmonary_vein,connectivity=1), axis=(0,1,2))
                # print(m)
            else:
                break

        if np.max(label(pulmonary_vein, connectivity=1), axis=(0,1,2)) == 1:
            pulmonary_vein = (lung_mask == 53)
            m = np.max(label(pulmonary_vein,connectivity=1), axis=(0,1,2))
            for i in range(0,20):
                if m > 3:
                    pulmonary_vein = binary_dilation(pulmonary_vein, iterations=1, structure=morph_anal.vol_strel())
                    m = np.max(label(pulmonary_vein,connectivity=1), axis=(0,1,2))
                    # print(m)
                else:
                    pulmonary_vein = morph_anal.find_objects(pulmonary_vein, num_objects=2)
                    break

        pulmonary_vein = label(pulmonary_vein,connectivity=1)

        positions1 = np.mean(np.argwhere(pulmonary_vein == 1), axis=0)
        positions2 = np.mean(np.argwhere(pulmonary_vein == 2), axis=0)

        if positions1[0] > positions2[0]:
            positions1, positions2 = positions2, positions1

        slope_L = central_analysis(pred, right_lung, lung_mask==53, positions1, 0.45, 20, dataO, vessels_mask)
        slope_R = central_analysis(pred, left_lung, lung_mask==53, positions2, 0.45, 20, dataO, vessels_mask)

        # if pat == 0:
        if df.shape[1] == 5:
            df.insert(6, 'Int_1_mean', res[0][0])
            df.insert(7, 'Int_1_cov', res[1][0])
            df.insert(8, 'Int_1_weight', res[2][0])
            df.insert(9, 'Int_2_mean', res[0][1])
            df.insert(10, 'Int_2_cov', res[1][1])
            df.insert(11, 'Int_2_weight', res[2][1])
            df.insert(12, 'Int_3_mean', res[0][2])
            df.insert(13, 'Int_3_cov', res[1][2])
            df.insert(14, 'Int_3_weight', res[2][2])
            df.insert(15, 'Entropy_1', ent01)
            df.insert(16, 'Entropy_2', ent02)
            df.insert(17, 'Slope_Left', slope_L)
            df.insert(18, 'Slope_Right', slope_R)
        else:
            df.at[pat, 'Int_1_mean'] = res[0][0]
            df.at[pat, 'Int_1_cov'] = res[1][0]
            df.at[pat, 'Int_1_weight'] = res[2][0]
            df.at[pat, 'Int_2_mean'] = res[0][1]
            df.at[pat, 'Int_2_cov'] = res[1][1]
            df.at[pat, 'Int_2_weight'] = res[2][1]
            df.at[pat, 'Int_3_mean'] = res[0][2]
            df.at[pat, 'Int_3_cov'] = res[1][2]
            df.at[pat, 'Int_3_weight'] = res[2][2]
            df.at[pat, 'Entropy_1'] = ent01
            df.at[pat, 'Entropy_2'] = ent02
            df.at[pat, 'Slope_Left'] = slope_L
            df.at[pat, 'Slope_Right'] = slope_R
    df.to_excel(output_folder, index=False)

    factorUp = (np.size(nifti_array,0)/np.size(pred,0), np.size(nifti_array,1)/np.size(pred,1), np.size(nifti_array,2)/np.size(pred,2))
    pred = ndimage.zoom(pred, factorUp, order=0)
    labels_nifti = nib.Nifti1Image(pred, nifti_data.affine)
    nib.save(labels_nifti, nifti_path.replace('.nii.gz','_labels.nii.gz'))


def lungSegmentation(output_folder):
        
    nifti_files = [file for file in os.listdir(output_folder) if file.endswith('.nii.gz')]

    print("Start lung segmentation")

    for file in nifti_files:    
    
        file_dada_path = os.path.join(output_folder, file)
        file_output_path = output_folder + os.sep + file.replace('.nii.gz','_segm.nii.gz')

        lung_parts = 'pulmonary_vein trachea lung_upper_lobe_left lung_upper_lobe_right lung_middle_lobe_right lung_lower_lobe_left lung_lower_lobe_right'
        os.system('TotalSegmentator.exe -i {} -o {} --task {} --roi_subset {} --fast -ml'.format(file_dada_path, file_output_path, 'total',lung_parts))

    print("Segmentation has been finished")




def get_pat_info(dcm_folder, output_folder):

    df = pd.DataFrame(columns=['Folder_name'])

    df.insert(1, 'Patient Name', 'Patient Name')
    df.insert(2, 'Patient ID', 'Patient ID')
    df.insert(3, 'Accession No', 'Accession No')
    df.insert(4, 'Date', 'Date')
    df.insert(5, 'Description', 'Description')

    # find all nifti files in output folder with lung masks
    nifti_files = [file for file in os.listdir(output_folder) if file.endswith('_segm.nii.gz')]

    i = 0
    for nifti_file in nifti_files:
        # load one dicom file from the folder which is specific in the first column of df
        path_dicom = os.path.join(dcm_folder, nifti_file.replace('_segm.nii.gz',''))
        # find all dicom files also in subfolders via glob
        dicoms = glob.glob(os.path.join(path_dicom, '**', '*.dcm'), recursive=True)
        if len(dicoms) == 0:
            dicoms = glob.glob(os.path.join(path_dicom, '**', 'I*'), recursive=True)

        # load one dicom file
        dicom = dicoms[2]
        # read dicom file
        # dicom = pydicom.dcmread(os.path.join(path_dicom, dicom))    
        dicom = pydicom.dcmread( dicom )    
        # read information from the dicom file of exist
        patient_name = None
        patient_id = None
        accession_no = None
        date = None
        if dicom.PatientName is not None:
            patient_name = dicom.PatientName
        if dicom.PatientID is not None:
            patient_id = dicom.PatientID
        if dicom.AccessionNumber is not None:
            accession_no = dicom.AccessionNumber
        if dicom.AcquisitionDate is not None:
            date = dicom.AcquisitionDate
        if dicom.AcquisitionDate is not None:
            description = dicom.StudyDescription
        
        # add information to the database
        df = pd.concat([df, pd.DataFrame([{
            'Folder_name': str(nifti_file.replace('_segm.nii.gz','')),
            'Patient Name': str(patient_name),
            'Patient ID': str(patient_id),
            'Accession No': str(accession_no),
            'Date': str(date),
            'Description': str(description)
        }])], ignore_index=True)
        

    df.to_excel(os.path.join( output_folder , 'Outcomes.xlsx'), index=False)


def convertToNii(dicom_folder, output_folder):
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