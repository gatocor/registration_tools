Usage
=====

.. toctree::
   :maxdepth: 2

Creating a Dataset
------------------

For creating a dataset from a folder of images, use the `create_dataset` function:

.. code-block:: python

    from registration_tools import create_dataset

    dataset = create_dataset(
        data=[
            'path/to/dataset/channel_0/image_{:02d}.tiff',
            'path/to/dataset/channel_1/image_{:02d}.tiff',
            'path/to/dataset/channel_2/image_{:02d}.tiff',
        ],
        format='XYZ',
        numbers=[1, 2, 3, 4, 5]
    )

If you already had a dataset created, you can load one using the `load_dataset` function.

.. code-block:: python

    from registration_tools import load_dataset

    dataset = load_dataset('path/to/dataset')

You can create some artificial datasets from the module `registration_tools.data`:

.. code-block:: python

    from registration_tools.data import sphere

    sphere(
        path='path/to/save/folder',
        num_images=10,
        image_size=100,
        num_channels=3,
        min_radius=5,
        max_radius=20,
        jump=2,
        stride=(1, 2, 3)
    )

Registering Images
------------------

To register images in a dataset, use the `register` function:

.. code-block:: python

    from registration_tools import create_dataset
    from registration_tools.registration import register

    dataset = create_dataset(
        data=[
            'path/to/dataset/channel_0/image_{:02d}.tiff',
            'path/to/dataset/channel_1/image_{:02d}.tiff',
            'path/to/dataset/channel_2/image_{:02d}.tiff',
        ],
        format='XYZ',
        numbers=[1, 2, 3, 4, 5]
    )
    dataset_registered = register(
        dataset=dataset,
        save_path='path/to/save/folder',
        pyramid_highest_level=3,
        pyramid_lowest_level=0,
        registration_type='translation',
        registration_direction='backward'
    )

Visualizing Images
------------------

There are several ways you can visualize images in a dataset. The package uses `napari` for visualization.

To visualize images in a dataset, use the `plot_images` function:

.. code-block:: python

    from registration_tools.visualization import plot_images
    import napari

    viewer = napari.Viewer()
    plot_images(
        viewer=viewer,
        dataset=dataset
    )

To plot the projections of the dataset, use the `plot_projections` function.

.. code-block:: python

    from registration_tools.visualization import plot_projections

    plot_projections(
        viewer=viewer,
        dataset=dataset,
        projection=0,
        channels=[0, 1, 2],
        old=False
    )

When registering, you may be interested in seeing the difference between the raw and original dataset. You can use the `plot_projections_difference` function to plot the difference between the two. 

.. code-block:: python

    from registration_tools.visualization import plot_projections_difference

    plot_projections_difference(
        viewer=viewer,
        dataset=dataset,
        projection='XY',
        channel=0,
        old=True
    )

Plotting Vectorfield
--------------------

To plot the vectorfield of the dataset, use the `plot_vectorfield` function:

.. code-block:: python

    from registration_tools.visualization import plot_vectorfield

    plot_vectorfield(
        viewer=viewer,
        dataset=dataset
    )