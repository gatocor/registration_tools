# Working with Datasets

## Index
- [Working with Datasets](#working-with-datasets)
  - [Index](#index)
  - [Types of allowed data](#types-of-allowed-data)
  - [Working with Datasets Structure](#working-with-datasets-structure)
    - [Creating an artificial Dataset](#creating-an-artificial-dataset)
    - [Loading a dataset of separated files](#loading-a-dataset-of-separated-files)
    - [Converting to zarr](#converting-to-zarr)

## Types of allowed data

The basic formats allowed are indexed arrays as:

 - numpy arrays
 - zarr arrays
 - h5/h5f files
 - separated files inside folders (see next section)

where we have zarr as the basic input array for its efficient manipulation using batches of data. You can upload it in any way you want. however, we also provide a function 'load_dataset' to read it that helps to make sure you upload it in the correct format.




```python
import numpy as np
import registration_tools.dataset as rt_dataset
```


```python
#Generate a random dataset with 4 timepoints, 3 channels, and 10x10x10 images
dataset = np.random.rand(4,3,10,10,10)
#Save the dataset to a file
np.save('dataset.npy', dataset)
```

Check the dataset structure and ensure that necessary attributes are present.


```python
rt_dataset.check_dataset_structure(dataset)
```

    Shape:  (4, 3, 10, 10, 10)
    Attributes not found.


You can upload with load dataset instead to make sure all necessary attributes for many functions are found.

**Note: This does not mean that you cannot work with this data. Simnply that mostly sure this data will be asked later on for some function.**


```python
dataset = rt_dataset.load_dataset(
    'dataset.npy',
    axis="TCZYX",
    scale=(1,1,1),
)

rt_dataset.check_dataset_structure(dataset)
```

    Shape:  (4, 3, 10, 10, 10)
    Axis:  TCZYX
    Scale:  (1, 1, 1)


## Working with Datasets Structure

Usually datasets are found in separate files distributed over folders. Usually, datasets coming from a microscope machine have structures similar to

        - dataset
            - ch1
                - file_t1.tif
                - file_t2.tif
                - ...
            - ch2
                - file_t1.tif
                - file_t2.tif
                - ...

In this example we show how to create the data structure `Dataset` to work with this format of data. 

### Creating an artificial Dataset

We are going to generate an artificial dataset and then we will load it. 
 - `registration_tools.data` contains functions to generate artificial datasets to test.
 - `registation_tools.dataset` contains functions to load datasets.


```python
import registration_tools.data as rt_data #For generating artificial datasets
import registration_tools.dataset as rt_dataset #For generating artificial datasets
```

If we provide a folder, the dataset will generate a folder structure in separated files.


```python
dataset = rt_data.sphere(
    out='dataset_sphere',
    num_channels=3,
    image_size=100,  #This indicates to make an image of size image_size x image_size x image_size
    stride=(1,1,2),  #This to downsample the image by a factor of stride per dimension
)
```

We can visualize the structure of our dataset:


```python
rt_dataset.show_dataset_structure('dataset_sphere')
```

    |-- channel_0
        |-- sphere_00.tiff
        |-- sphere_01.tiff
        |-- sphere_02.tiff
        |-- ...
        |-- sphere_07.tiff
        |-- sphere_08.tiff
        |-- sphere_09.tiff
    |-- channel_1
        |-- sphere_00.tiff
        |-- sphere_01.tiff
        |-- sphere_02.tiff
        |-- ...
        |-- sphere_07.tiff
        |-- sphere_08.tiff
        |-- sphere_09.tiff
    |-- channel_2
        |-- sphere_00.tiff
        |-- sphere_01.tiff
        |-- sphere_02.tiff
        |-- ...
        |-- sphere_07.tiff
        |-- sphere_08.tiff
        |-- sphere_09.tiff


### Loading a dataset of separated files

Now we can load this folder structure as an object Dataset.


```python
dataset = rt_dataset.Dataset(
    [
        "dataset_sphere/channel_0/sphere_{:02d}.tiff",
        "dataset_sphere/channel_1/sphere_{:02d}.tiff",
        "dataset_sphere/channel_2/sphere_{:02d}.tiff",
    ],
    axis_data="CT",
    axis_files="XYZ",
    scale=(1,1,2)      # Scale of the dataset, is the same as the stride in the generation
)

dataset
```




    Dataset(shape=(3, 10, 100, 100, 50), axis=CTXYZ, scale=(1, 1, 2))



### Converting to zarr

You can work with this Dataset for most of the functions afterwards. However you may be interested in converting it to a zarr array.


```python
dataset.to_zarr("dataset_spheres.zarr")
```

    Saving to Zarr: 100%|██████████| 30/30 [00:00<00:00, 74.52images/s]

