%% load new dicoms

clear all
close all
clc


path = 'D:\Projekty\CTPA_VFN\lung_CTPA\data\dicoms_raw\DATA\DICOM';

dcm = dicomCollection(path);

ID = {};
for i = 1:49
    pat = dcm{i,"Filenames"};
    info = dicominfo(pat{1}(1));

    ID{i,1} = [info.StudyID '0'];
end

ID = sort(ID);

% img = dicomreadVolume(dcm(1,"Filenames"));
% imshow5(squeeze(img))

save('dcm_list.mat','dcm')

