import numpy as np
import scipy.ndimage as ndi
from skimage import measure, morphology
from skimage.filters import sobel
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from utils import display_orthogonal_views

def active_contour_3d(initial_contour, iterations, alpha=0.1, beta=0.1, gamma=0.1):

    # Create initial parametric contour
    verts, faces, _, _ = measure.marching_cubes(initial_contour, 0, step_size=5)
    parametric_contour = verts

    for i in range(iterations):
        # Calculate internal forces
        dx = np.gradient(parametric_contour[:, 0])
        dy = np.gradient(parametric_contour[:, 1])
        dz = np.gradient(parametric_contour[:, 2])
        
        # First order derivative (elasticity)
        elasticity = alpha * (np.roll(parametric_contour, -1, axis=0) - np.roll(parametric_contour, 1, axis=0))
        
        # Second order derivative (stiffness)
        stiffness = beta * (np.roll(parametric_contour, -2, axis=0) - 2 * parametric_contour + np.roll(parametric_contour, 2, axis=0))
        
        # Image forces (edge map)
        # edges = sobel(binary_image.astype(float))
        # external_forces = gamma * ndi.map_coordinates(edges, [parametric_contour[:, 0], parametric_contour[:, 1], parametric_contour[:, 2]], order=1)
        
        # Update parametric contour
        # parametric_contour += elasticity + stiffness - external_forces[:, np.newaxis]
        # parametric_contour += elasticity + stiffness

    
    return parametric_contour, faces

# create a simple binary image -white square
binary_image = np.zeros((50, 50, 50), dtype=bool)
binary_image[10:40, 10:40, 10:40] = 1

# Erode the image to get initial contour
eroded_image = morphology.binary_erosion(binary_image)
initial_contour = binary_image ^ eroded_image

# display_orthogonal_views
# display_orthogonal_views(binary_image)
display_orthogonal_views(initial_contour)

# Perform active contour segmentation
iterations = 1
contour, faces = active_contour_3d(initial_contour, iterations)

# Plotting the result
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a mesh for the contour
mesh = Poly3DCollection(contour[faces], alpha=0.7)
mesh.set_facecolor('cyan')
ax.add_collection3d(mesh)

# Set plot parameters
ax.set_xlim(-10, binary_image.shape[0])
ax.set_ylim(-10, binary_image.shape[1])
ax.set_zlim(-10, binary_image.shape[2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Active Contour Segmentation')

plt.show()
