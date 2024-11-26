import numpy as np
import matplotlib.pyplot as plt     
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation
import os
from scipy.ndimage import distance_transform_edt
from skimage.measure import label


def display_orthogonal_views(volume, slice_index=None, save_path=None):
    """
    Display orthogonal views (axial, sagittal, coronal) of a 3D volume.

    Parameters:
    volume (np.ndarray): 3D numpy array representing the volume.
    slice_index (tuple): Tuple of slice indices (axial, sagittal, coronal). If None, the middle slices are used.
    """
    if slice_index is None:
        # Default to middle slices if no indices are provided
        slice_index = (volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2)
    
    axial_slice = volume[slice_index[0], :, :]
    sagittal_slice = volume[:, slice_index[1], :]
    coronal_slice = volume[:, :, slice_index[2]]

    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(axial_slice, cmap='gray')
    axes[0].set_title(f'Axial View (slice {slice_index[0]})')
    axes[0].axis('off')

    axes[1].imshow(sagittal_slice, cmap='gray')
    axes[1].set_title(f'Sagittal View (slice {slice_index[1]})')
    axes[1].axis('off')

    axes[2].imshow(coronal_slice, cmap='gray')
    axes[2].set_title(f'Coronal View (slice {slice_index[2]})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    if save_path is not None:
        plt.savefig(save_path, format='png')
        plt.close()

def compute_mean_intensity(data, mask):
    data[data==0] = np.nan
    mean_intensity = np.nanmean(data[mask])
    return mean_intensity

def data_subsampling(data, mask, factor=0.5):
    data = ndimage.zoom(data, factor, order=0)
    mask = ndimage.zoom(mask, factor, order=0)
    return data, mask

def lung_separate(data, mask):
    left_lung_mask = (mask == 10) | (mask == 11)
    right_lung_mask = ((mask == 12) | (mask == 13) | (mask == 14))    
    trachea_mask = (mask == 16)
    vessels_mask = (data > -600)

    # mask vessels matrix  by right and left mask
    vessels_mask = vessels_mask & (left_lung_mask | right_lung_mask)
    # vessels_mask = np.zeros_like(data, dtype=bool)

    # Dilate the vessels mask to include surrounding tissue
    # vessels_mask = binary_dilation(vessels_mask, iterations=1)

    return left_lung_mask, right_lung_mask, trachea_mask, vessels_mask

def int_analyze(data, mask, vessels, file_path):

    mask1 = binary_dilation(mask, iterations=4)
    mask1 = binary_erosion(mask1, iterations=5)
    vessel2 = binary_dilation(vessels, iterations=1)

    lung_tissue = data.copy()
    lung_tissue = ndimage.zoom(data, 0.5, order=0)
    vessel2 = ndimage.zoom(vessel2, 0.5, order=0)
    mask1 = ndimage.zoom(mask1, 0.5, order=0)
    lung_tissue[vessel2] = np.nan
    lung_tissue[~mask1] = np.nan
    lung_tissue[lung_tissue>-500] = np.nan

    lung_tissue_vekt = lung_tissue[~np.isnan(lung_tissue)].reshape(-1,1)
    lung_tissue_vekt = np.round(lung_tissue_vekt).astype(int)
    # gm = BayesianGaussianMixture(n_components=3,random_state=42).fit(lung_tissue_vekt)
    # gm = BayesianGaussianMixture(n_components=3,random_state=42,tol=1e-1,max_iter=500,init_params='k-means++').fit(lung_tissue_vekt)
    gm = GaussianMixture(n_components=3, random_state=42).fit(lung_tissue_vekt)

    # sort the means and save indexes which them sort others
    indx = np.argsort(gm.means_.squeeze())
    val = BayesianGaussianMixture(n_components=3, random_state=42)
    # val = GaussianMixture(n_components=3, random_state=42)
    val.weights_, val.covariances_, val.means_ = gm.weights_.squeeze()[indx], gm.covariances_.squeeze()[indx], gm.means_.squeeze()[indx]

    display_hist(lung_tissue_vekt, val, file_path)
    return gm, val


def predict_mask(data, vessels, mask, model):
    mask1 = binary_dilation(mask, iterations=2)
    mask1 = binary_erosion(mask1, iterations=4)
    vessel2 = binary_erosion(vessels, iterations=1)
    
    lung_tissue = data.copy()
    lung_tissue[vessel2] = np.nan
    # lung_tissue[~((lung_mask<15) & (lung_mask>9))] = np.nan
    # lung_tissue[~binary_erosion((lung_mask<15) & (lung_mask>9),iterations=1)] = np.nan
    lung_tissue[~(mask1>0)] = np.nan
    lung_tissue[lung_tissue>-500] = np.nan
    lung_tissue_vekt = lung_tissue[~np.isnan(lung_tissue)].reshape(-1,1)
    # lung_tissue_vekt[lung_tissue_vekt>-600] = np.nan
    lung_tissue_vekt = np.round(lung_tissue_vekt).astype(int)
    gmW_labels = model.predict(lung_tissue_vekt)

    # find mean values for each gmW_labels (0,1,2) in lung_tissue_vekt and sort the labels according to means and permute the labels 0,1,2 
    means = np.zeros(3)
    for i in range(3):
        means[i] = np.mean(lung_tissue_vekt[gmW_labels==i])
    indx = np.argsort(means)
    gmW_labels2 = np.array([2 if x==indx[0] else 1 if x==indx[1] else 0 for x in gmW_labels])

    labels = np.zeros_like(lung_tissue)
    labels[~np.isnan(lung_tissue)] = gmW_labels2+1

    labels = labels.reshape(lung_tissue.shape)

    return labels

def display_hist(data, gm, file_path):
    # display histogram of data and estimated Gassians distributions together
    plt.ion()
    # plt.figure()
    plt.hist(data[~np.isnan(data)], bins=np.max(data[~np.isnan(data)]) - np.min(data[~np.isnan(data)]), density=True)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of lung tissue intensities')
    x = np.linspace(-500, -1000, 1000)
    x = np.expand_dims(x, axis=1)
    y = np.zeros_like(x)
    for i in [1,0,2]:
        y = y + gm.weights_[i]*np.exp(-0.5*(x-gm.means_[i])**2/gm.covariances_[i])/np.sqrt(2*np.pi*gm.covariances_[i])
        plt.plot(x, gm.weights_[i]*np.exp(-0.5*(x-gm.means_[i])**2/gm.covariances_[i])/np.sqrt(2*np.pi*gm.covariances_[i]))
    plt.plot(x, y)
    plt.xlim(-1000, -600)
    plt.show()
    # save the figure with plot as png image file
    plt.savefig(file_path, format='png')
    plt.close()


    

class morph_anal:
    @staticmethod
    def find_endpoints(skeleton):
        skeleton_modified = skeleton.copy()
        endpoints_list = []
        # Find the endpoints of the skeleton
        endpoints = np.argwhere(skeleton == 1)
        for endpoint in endpoints:
            x, y, z = endpoint
            # Check if the endpoint is a true endpoint
            if np.sum(skeleton[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]) > 2:
                skeleton_modified[x, y, z] = 0
                endpoints_list.append((x, y, z))
        return skeleton_modified
    
    # find largest objects in the mask
    def find_objects( mask, num_objects=1):
        labels = label(mask,connectivity=1)
        labels_reduced = np.zeros_like(labels)
        vel = np.zeros(np.max(labels))
        for i in range(1, np.max(labels) + 1):
            vel[i-1] = np.sum(labels == i)
        vel = np.argsort(vel)[::-1]+1
        for i in range(0, num_objects):
            labels_reduced = labels_reduced + ((labels == (vel[i]))*(i+1))
        return labels_reduced
    
    def vol_strel():
        structure=np.ones((3,3,3), dtype=bool)
        structure[0,0,0] = False
        structure[0,0,2] = False
        structure[0,2,0] = False
        structure[0,2,2] = False
        structure[2,0,0] = False
        structure[2,0,2] = False
        structure[2,2,0] = False
        structure[2,2,2] = False
        return structure
    

def central_analysis(labels, mask_part, vein, H, max_H, dil_contr, nifti_array, vessels_mask):

    hyl = np.zeros_like(mask_part, dtype=np.bool_)
    hyl[int(H[0]), int(H[1]), int(H[2])] = True
    dist_map_H = distance_transform_edt(~hyl)

    # create contour of lung mask
    contour = mask_part & ~binary_erosion(mask_part > 0, iterations=1)
    contour = (contour > 0) & ~(binary_dilation((vein), iterations=dil_contr,structure=morph_anal.vol_strel()) > 0)
    # contour = contour * ~(binary_dilation((nifti_array < 550) & (nifti_array > 450), iterations=1,structure=morph_anal.vol_strel()))

    # contour = morph_anal.find_objects(contour, num_objects=1)
    dist_map_Contr = distance_transform_edt(~contour)
    dist_map = ( ( dist_map_Contr ) / ( dist_map_Contr + dist_map_H ) ) * mask_part

     # num = 11
    num = 21 # for visualization

    steps_thr = np.linspace(0,max_H,num)
    step = steps_thr[1] - steps_thr[0]

    valM = np.zeros(int(num))
    valL1 = np.zeros(int(num))
    valL2 = np.zeros(int(num))

    i = 0
    for thr in steps_thr:
        bin = (dist_map > thr) & (dist_map < (thr + step))
        valM[i] = np.mean(nifti_array[ bin & (~vessels_mask) ])
        valL1[i] = np.sum( labels[ bin & (~vessels_mask) ]==1 )/ np.sum( labels[ bin & (~vessels_mask) ]>0 ) 
        valL2[i] = np.sum( labels[ bin & (~vessels_mask) ]==2 )/ np.sum( labels[ bin & (~vessels_mask) ]>0 )             
        i += 1


    # fit linear curve to the data valM and give me the slope and ignore nan values
    valM2 = valM[~np.isnan(valM)]
    valM2 = valM2[::-1]
    ind2 = np.linspace(0,1,len(valM))[~np.isnan(valM)]
    slope, bias = np.polyfit(ind2, valM2, 1)

    return slope