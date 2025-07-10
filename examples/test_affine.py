import napari
from scipy.ndimage import affine_transform
import numpy as np
import registration_tools.data as rt_data #For generating artificial datasets
import registration_tools.visualization as rt_vis #For visualization
import registration_tools.registration as rt_reg #For registration
import zarr

dataset = rt_data.sphere(
    num_images=10,
    image_size=150,
    num_channels=3,
    min_radius=5,
    max_radius=5,
    jump=3,
    stride=(1, 1, 1)
)
dataset = dataset[0,0]

# Ensure orthogonality by computing cross product
normal = np.array([1,0,0])
angle = np.pi/100
# Rodrigues' rotation formula
K = np.array([
    [0, -normal[2], normal[1]],
    [normal[2], 0, -normal[0]],
    [-normal[1], normal[0], 0]
])
rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
rotation_matrix_aff = np.eye(4)
rotation_matrix_aff[:3,:3] = rotation_matrix

translation = np.array([-75,-75,-75])
translation_aff = np.eye(4)
translation_aff[:3,3] = translation
translation_inv_aff = np.eye(4)
translation_inv_aff[:3,3] = -translation

t = translation_inv_aff @ rotation_matrix_aff @ translation_aff
print(t)
t_rot = t[:3,:3]
t_trans = t[:3,3]

dataset_transformed = affine_transform(dataset, t_rot, offset=t_trans)

viewer = napari.Viewer()
rt_vis.add_image(viewer, dataset, name="Dataset", colormap="reds", blending="additive")
rt_vis.add_image(viewer, dataset_transformed, name="Dataset transformed", colormap="greens", opacity=0.5, blending="translucent")
viewer.dims.ndisplay = 3
viewer.dims.current_step = (0, 0, 0)
napari.run()
