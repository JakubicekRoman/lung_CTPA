# import pandas
import pandas as pd
import os
import pydicom


# read excels tables and add information to the database
# paths to excels files

excels_path = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\data2'
dicom_folder = r'D:\Projekty\CTPA_VFN\lung_CTPA\data\data2\dicoms'

excels_files = [file for file in os.listdir(excels_path) if file.endswith('y.xlsx')]

# find all escels files in the path

# read excels file
# num_file = 0

for num_file in range(len(excels_files)):
    df = pd.read_excel(os.path.join(excels_path, excels_files[num_file]))

    # add column to the database
    # at specific position
    df.insert(1, 'Patient Name', 'Patient Name')
    df.insert(2, 'Patient ID', 'Patient ID')
    df.insert(3, 'Accession No', 'Accession No')
    df.insert(4, 'Date', 'Date')

    # add information to the database to each row
    for i in range(len(df)):
        # load one dicom file from the folder which is specific in the first column of df
        path_dicom = os.path.join(dicom_folder, df.iloc[i, 0])
        # find all its subfolders
        folders = [f for f in os.listdir(path_dicom) if os.path.isdir(os.path.join(path_dicom, f))]
        # find all dicoms contaning I* in name in path_dicom and the one folder
        dicoms = [f for f in os.listdir(os.path.join(path_dicom, folders[0])) if f.startswith('I')] 
        # load one dicom file
        dicom = dicoms[2]
        # read dicom file
        dicom = pydicom.dcmread(os.path.join(path_dicom,folders[0], dicom))
        # read information from the dicom file
        patient_name = dicom.PatientName
        patient_id = dicom.PatientID
        accession_no = dicom.AccessionNumber
        date = dicom.AcquisitionDate
        
        # add information to the database
        df.iloc[i, 1] = str(patient_name)
        df.iloc[i, 2] = str(patient_id)
        df.iloc[i, 3] = str(accession_no)
        df.iloc[i, 4] = str(date)

    df.to_excel(os.path.join(excels_path, excels_files[num_file]).replace('.xlsx','_2.xlsx'), index=False)


