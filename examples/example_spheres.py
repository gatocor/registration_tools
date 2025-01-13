import os
import napari
import registration_tools as rt
import registration_tools.data as rt_data
import registration_tools.visualization as rt_vis
import registration_tools.registration as rt_reg

# Create a dataset of spherical images
rt_data.sphere(
    path='sphere_dataset',
    num_images=10,
    image_size=100,
    num_channels=3,
    min_radius=5,
    max_radius=5,
    jump=2,
    stride=(1, 1, 1)
)

# Load the dataset
dataset = rt.create_dataset(
    data=[
        'sphere_dataset/channel_0/sphere_{:02d}.tiff',
        'sphere_dataset/channel_1/sphere_{:02d}.tiff',
        'sphere_dataset/channel_2/sphere_{:02d}.tiff',
    ],
    format='XYZ',
    numbers=list(range(10))
)
print(dataset)

# Initialize the napari viewer
viewer = napari.Viewer()

# Plot the images in the dataset
rt_vis.plot_images(viewer, dataset)

# Make video
rt_vis.make_video(
    viewer=viewer,
    save_file='sphere_dataset.gif',
    fps=1,
)

# Register the video
registered_dataset = rt_reg.register(
    dataset=dataset,
    save_path='registered_sphere_dataset',
    pyramid_highest_level=3,
    pyramid_lowest_level=0,
    registration_type='translation',
    registration_direction='backward'
)

# Make video registered
viewer.layers.clear()
rt_vis.plot_images(viewer, registered_dataset)
rt_vis.make_video(
    viewer=viewer,
    save_file='sphere_dataset_registered.gif',
    fps=1,
)

# Make video projections
viewer.layers.clear()
rt_vis.plot_projections_difference(viewer, registered_dataset, 0)
rt_vis.make_video(
    viewer=viewer,
    save_file='sphere_dataset_projections_difference.gif',
    fps=1,
)
