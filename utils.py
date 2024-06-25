import numpy as np
import matplotlib.pyplot as plt     


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