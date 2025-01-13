Usage
=====

Creating a Dataset
------------------

To create a dataset of spherical images, use the `dataset_sphere` function:

```python
from registration_tools.data import dataset_sphere

dataset_sphere(
    folder='path/to/save/folder',
    num_images=10,
    image_size=100,
    num_channels=3,
    min_radius=5,
    max_radius=20,
    jump=2,
    stride=(1, 2, 3)
)
```

Registering Images
------------------

To register images in a dataset, use the `register` function:

```python
from registration_tools import Dataset
from registration_tools.registration import register

dataset = Dataset('path/to/dataset', format='XYZ', numbers=[1, 2, 3, 4, 5])
register(
    dataset=dataset,
    save_path='path/to/save/folder',
    use_channel=0
)
```

Visualizing Images
------------------

To visualize images in a dataset, use the `visualize_images` function:

```python
from registration_tools.visualization import visualize_images
import napari

viewer = napari.Viewer()
visualize_images(
    viewer=viewer,
    dataset=dataset,
    channels=[0, 1, 2],
    save_folder='path/to/save/folder'
)
```

Plotting Projections
---------------------

To plot the projections of the dataset, use the `plot_projections` function:

```python
from registration_tools.visualization import plot_projections

plot_projections(
    viewer=viewer,
    dataset=dataset,
    projection='XY',
    channels=[0, 1, 2],
    old=False
)
```

Plotting Projections Difference
-------------------------------

To plot the difference between current and old projections of the dataset, use the `plot_projections_difference` function:

```python
from registration_tools.visualization import plot_projections_difference

plot_projections_difference(
    viewer=viewer,
    dataset=dataset,
    projection='XY',
    channel=0,
    old=True
)
```

Plotting Vectorfield
--------------------

To plot the vectorfield of the dataset, use the `plot_vectorfield` function:

```python
from registration_tools.visualization import plot_vectorfield

plot_vectorfield(
    viewer=viewer,
    dataset=dataset
)
```