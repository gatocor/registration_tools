# Non-linear registration of growing spheres

This notebook demonstrates how to create, visualize, and rigid register a dataset of growing spherical images using the `registration_tools` package.

## Index
- [Non-linear registration of growing spheres](#non-linear-registration-of-growing-spheres)
  - [Index](#index)
  - [Import the corresponding libraries](#import-the-corresponding-libraries)
  - [Create an artificial dataset of growing spherical images](#create-an-artificial-dataset-of-growing-spherical-images)
  - [Visualizing](#visualizing)
  - [Make video](#make-video)
  - [Register the images](#register-the-images)
  - [Plot the vector field and make videos](#plot-the-vector-field-and-make-videos)

## Import the corresponding libraries

```python
import os
import napari
import registration_tools as rt #Main package
import registration_tools.data as rt_data #For generating artificial datasets
import registration_tools.visualization as rt_vis #For visualization
import registration_tools.registration as rt_reg #For registration
```

## Create an artificial dataset of growing spherical images

We will create a dataset of 10 growing spherical images with 3 channels.

```python
# Create a dataset of growing spherical images
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
```

## Visualizing

Initialize the napari viewer to visualize the dataset.

```python
# Initialize the napari viewer
viewer = napari.Viewer()
```

and load the dataset to napari,

```python
# Plot the images in the dataset
viewer.layers.clear() #Clear the viewer of other layers that may be present
rt_vis.plot_images(viewer, dataset)
```

![../assets/sphere_growing_dataset.gif](../assets/sphere_growing_dataset.gif)

## Make video

Create a video from the images in the dataset.

```python
# Make video
rt_vis.make_video(
    viewer=viewer,
    save_file='sphere_growing_dataset.gif',
    fps=1,
)
```

## Register the images

Register the images in the dataset to correct for any misalignments.

```python
# Register the video
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
```

## Plot the vector field and make videos

Now we can check the deformation field by plotting the vectorfield along with the images to see the deformation.

```python
# Make video registered
viewer.layers.clear()
viewer.dims.ndisplay = 3 #Set the viewer to 3D mode
rt_vis.plot_images(viewer, registered_dataset)
rt_vis.plot_vectorfield(viewer, registered_dataset)

rt_vis.make_video(
    viewer=viewer,
    save_file='sphere_growing_dataset_registered.gif',
    fps=1,
)
```

![../assets/sphere_growing_dataset_registered.gif](../assets/sphere_growing_dataset_registered.gif)
