import os
import napari
import registration_tools as rt
import shutil
import registration_tools.data as rt_data
import registration_tools.visualization as rt_vis
import registration_tools.registration as rt_reg

# Create a dataset of spherical images
dataset = rt_data.sphere(
    path='sphere_dataset',
    num_images=10,
    image_size=100,
    num_channels=3,
    min_radius=10,
    max_radius=20,
    jump=0,
    stride=(1, 1, 1),
    decay_factor=0,
)
print(dataset)

# Initialize the napari viewer
viewer = napari.Viewer()

# Plot the images in the dataset
rt_vis.plot_images(viewer, dataset)

# Make video
rt_vis.make_video(
    viewer=viewer,
    save_file='sphere_growing_dataset.gif',
    fps=1,
)

# Register the video
pll, phl = rt_reg.get_pyramid_levels(dataset, maximum_size=100)
# Remove the registered_sphere_dataset directory if it exists
if os.path.exists('registered_sphere_dataset'):
    shutil.rmtree('registered_sphere_dataset')
registered_dataset = rt_reg.register(
    dataset=dataset,
    save_path='registered_sphere_dataset',
    pyramid_highest_level=5,
    pyramid_lowest_level=0,
    registration_type='vectorfield',
    registration_direction='backward',
    verbose=False,
    vectorfield_spacing=3,
)

# Make video registered
viewer.layers.clear()
viewer.dims.ndisplay = 3
rt_vis.plot_images(viewer, registered_dataset)
rt_vis.plot_vectorfield(viewer, registered_dataset)
rt_vis.make_video(
    viewer=viewer,
    save_file='sphere_growing_dataset_registered.gif',
    fps=1,
)

viewer.close()