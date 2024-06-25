import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_erosion
import matplotlib.pyplot as plt


# Function to compute internal forces (smoothness constraint)
def compute_internal_forces(contour, alpha=0.1, beta=0.1):
    n = len(contour)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -2 * alpha - 6 * beta
        A[i, (i - 1) % n] = alpha + 4 * beta
        A[(i + 1) % n, i] = alpha + 4 * beta
        A[(i - 2) % n, i] = beta
        A[(i + 2) % n, i] = beta
    return np.dot(A, contour)

# Function to update the contour based on internal forces
def update_contour(contour, internal_forces, gamma=0.1):
    return contour + gamma * internal_forces

# Generate initial parametric contour (mesh model)
def generate_initial_contour(binary_image):
    indices = np.argwhere(binary_image)
    center = np.mean(indices, axis=0)
    distances = np.linalg.norm(indices - center, axis=1)
    radius = np.mean(distances)
    theta = np.linspace(0, 2 * np.pi, len(indices))
    phi = np.linspace(0, np.pi, len(indices))
    contour = np.array([
        center[0] + radius * np.sin(phi) * np.cos(theta),
        center[1] + radius * np.sin(phi) * np.sin(theta),
        center[2] + radius * np.cos(phi)
    ]).T
    return contour

# Main segmentation function
def active_contour_segmentation(binary_image, iterations=100, alpha=0.1, beta=0.1, gamma=0.1):
    contour = generate_initial_contour(binary_image)
    for _ in range(iterations):
        internal_forces = compute_internal_forces(contour, alpha, beta)
        contour = update_contour(contour, internal_forces, gamma)
    return contour

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


# Example usage
if __name__ == "__main__":
    # Create a sample binary 3D image (replace this with your binary 3D image)
    binary_image = np.zeros((50, 50, 50), dtype=bool)
    binary_image[10:40, 10:40, 10:40] = 1

    # create binary contour of binary image via erosion
    binary_contour = binary_image - binary_erosion(binary_image).astype(np.float32)    

    # Perform segmentation
    segmented_contour = active_contour_segmentation(binary_contour, iterations=50, alpha=0.1, beta=0.1, gamma=0.5)

    # Display orthogonal views of the segmented contour
    display_orthogonal_views(binary_contour)

    print(segmented_contour)

    # Plot the result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(segmented_contour[:, 0], segmented_contour[:, 1], segmented_contour[:, 2], color='r')
    ax.set_title('Segmented Contour')
    plt.show()

