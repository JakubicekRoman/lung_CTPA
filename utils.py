import numpy as np
import matplotlib.pyplot as plt     
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation
import os

def display_orthogonal_views(volume, slice_index=None):
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

    # Dilate the vessels mask to include surrounding tissue
    vessels_mask = binary_dilation(vessels_mask, iterations=1)

    return left_lung_mask, right_lung_mask, trachea_mask, vessels_mask

def int_analyze(data, mask, vessels, file_path):

    mask1 = binary_dilation(mask, iterations=2)
    mask1 = binary_erosion(mask1, iterations=3)

    lung_tissue = data.copy()
    lung_tissue[vessels] = np.nan
    lung_tissue[~mask1] = np.nan

    lung_tissue_vekt = lung_tissue[~np.isnan(lung_tissue)].reshape(-1,1)
    lung_tissue_vekt[lung_tissue_vekt>-600] = -600
    lung_tissue_vekt = np.round(lung_tissue_vekt).astype(int)
    gm = BayesianGaussianMixture(n_components=3, random_state=42).fit(lung_tissue_vekt)

    # sort the means and save indexes which them sort others
    indx = np.argsort(gm.means_.squeeze())
    val = BayesianGaussianMixture(n_components=3, random_state=42)
    val.weights_, val.covariances_, val.means_ = gm.weights_.squeeze()[indx], gm.covariances_.squeeze()[indx], gm.means_.squeeze()[indx]

    display_hist(lung_tissue_vekt, val, file_path)
    return gm, val

def display_hist(data, gm, file_path):
    # display histogram of data and estimated Gassians distributions together
    plt.ion()
    # plt.figure()
    plt.hist(data[~np.isnan(data)], bins=425, density=True)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of lung tissue intensities')
    x = np.linspace(-600, -1000, 1000)
    x = np.expand_dims(x, axis=1)
    y = np.zeros_like(x)
    for i in range(3):
        y = y + gm.weights_[i]*np.exp(-0.5*(x-gm.means_[i])**2/gm.covariances_[i])/np.sqrt(2*np.pi*gm.covariances_[i])
        plt.plot(x, gm.weights_[i]*np.exp(-0.5*(x-gm.means_[i])**2/gm.covariances_[i])/np.sqrt(2*np.pi*gm.covariances_[i]))
    plt.plot(x, y)
    plt.show()
    # save the figure with plot as png image file
    plt.savefig(file_path, format='png')
    plt.close()