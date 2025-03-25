import os
import re
from collections import Counter
import numpy as np
from skimage.io import imread, imsave
import json
from copy import deepcopy
import h5py
import zarr
from tqdm import tqdm
from dask import delayed, compute
import dask.array as da
from dask.diagnostics import Callback

class _TqdmCallback(Callback):
    def __init__(self, *args, **kwargs):
        self.tqdm_bar = tqdm(*args, **kwargs)
        super().__init__()
    def _start_state(self, dsk, state):
        self.tqdm_bar.reset(total=len(dsk))
    def _pretask(self, key, dsk, state):
        pass
    def _posttask(self, key, result, dsk, state, id):
        self.tqdm_bar.update(1)
    def _finish(self, dsk, state, errored):
        self.tqdm_bar.close()

def check_dataset_structure(data):
    """
    Check and print the structure of a dataset.

    This function prints the shape of the dataset and checks for the presence
    of specific attributes ('axis' and 'scale'). If these attributes are found,
    their values are printed. If not, a message indicating their absence is printed.

    Parameters:
    data (object): The dataset to be checked. It is expected to have a 'shape' attribute
                   and optionally 'attrs' attribute which is a dictionary containing
                   'axis' and 'scale' keys.

    Returns:
    None
    """

    print("Shape: ", data.shape)
    if hasattr(data, "attrs"):
        if "axis" in data.attrs:
            print("Axis: ", data.attrs["axis"])
        else:
            print("Axis attribute not found.")
        if "scale" in data.attrs:
            print("Scale: ", data.attrs["scale"])
        else:
            print("Scale attribute not found.")
    else:
        print("Attributes not found.")

def show_dataset_structure(folder_path, indent=0, max_files=6):
    """
    Recursively prints the folder structure.

    Parameters:
    folder_path (str): The path to the folder to display.
    indent (int): The indentation level for nested folders.
    max_files (int): The maximum number of files to display per folder.

    Returns:
    None
    """
    for item in sorted(os.listdir(folder_path)):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            print(' ' * indent + '|-- ' + item)
            sub_items = sorted(os.listdir(item_path))
            show_dataset_structure(item_path, indent + 4, max_files)
            if len(sub_items) > max_files:
                for sub_item in sub_items[:3]:
                    print(' ' * (indent + 4) + '|-- ' + sub_item)
                print(' ' * (indent + 4) + '|-- ...')
                for sub_item in sub_items[-3:]:
                    print(' ' * (indent + 4) + '|-- ' + sub_item)
            else:
                for sub_item in sub_items:
                    print(' ' * (indent + 4) + '|-- ' + sub_item)


def load_dataset(path, axis=None, scale=None, h5_key=None):
    """
    Load a dataset from a given file path.

    Parameters:
        path (str): The file path to the dataset. Supported formats are .npy, .h5, .hdf5, and image files.
        axis (optional): The axis attribute of the dataset. If not provided, it will be read from the dataset attributes.
        scale (optional): The scale attribute of the dataset. If not provided, it will be read from the dataset attributes.
        h5_key (str, optional): The key to read from an h5 file. Required if the file is in .h5 or .hdf5 format.

    Returns:
        zarr.core.Array: The loaded dataset as a zarr array.
    """

    if path.endswith('.npy'):
        data = np.load(path)
        dataset = zarr.array(data)
    elif path.endswith('.h5') or path.endswith('.hdf5'):
        if h5_key is None:
            raise ValueError("Must provide a key to read an h5 file.")
        with h5py.File(path, 'r') as f:
            dataset = np.array(f[h5_key])
    else:
        data = imread(path)
        dataset = zarr.array(data)


    # Check for axis and scale in dataset attributes
    if hasattr(dataset, 'attrs'):
        if 'axis' not in dataset.attrs:
            if axis is None:
                raise ValueError("Dataset is missing 'axis' attribute. Please provide it.")
            elif len(axis) != len(dataset.shape):
               raise ValueError("Axis must have the same length as the number of dimensions in the dataset.")
            else:
                dataset.attrs['axis'] = axis
        elif 'axis' in dataset.attrs and axis is not None:
            print("Warning: 'axis' attribute is already present in the dataset. Ignoring provided 'axis' attribute.")

        if 'scale' not in dataset.attrs:
            if scale is None:
                dataset.attrs['scale'] = (1,) * len([i for i in dataset.attrs['axis'] if i in 'XYZ'])
            elif len(scale) != len([i for i in dataset.attrs['axis'] if i in 'XYZ']):
                raise ValueError("Scale must have the same length as the number of spatial dimensions in the dataset.")
            else:
                dataset.attrs['scale'] = scale
    else:
        raise ValueError("Dataset is missing attributes. Please provide 'axis' and 'scale' attributes.")
    
    return dataset

def read_file(path, h5_key=None):
    """
    Reads an image file in various formats including numpy, h5, and other formats supported by scikit-image.
    Args:
        path (str): The path to the image file.
    Returns:
        np.ndarray: The image data as a numpy array.
    """
    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.h5') or path.endswith('.hdf5'):
        if h5_key is None:
            raise ValueError("Must provide a key to read an h5 file.")
        with h5py.File(path, 'r') as f:
            return np.array(f[h5_key])
    else:
        return imread(path)

def _regex_to_list(regex):
    """
    Converts a regex pattern to a list of numbers.

    Args:
        regex (str): The regex pattern to convert to a list of numbers.

    Returns:
        list: A list of numbers.
    """
    files_ = []
    directory = os.path.dirname(regex)
    pattern = os.path.basename(regex)
    regex_pattern = re.compile(pattern)
    files = [os.path.join(directory, file) for file in os.listdir(directory)]
    for num in range(0,len(os.listdir(directory))):
        file = regex.format(num)
        if file in files:
            files_.append(file)
    files_.sort()

    if len(files_) == 0:
        raise FileNotFoundError(f"No files found with pattern {regex}")
    
    return files_

def _expand_lists(data):

    if hasattr(data, '__iter__') and not isinstance(data, str):
        new_data = []
        for subdata in data:
            new_data.append(_expand_lists(subdata))
        return new_data
    elif isinstance(data, str) and re.search(r'\{.*\}', data):
        return _regex_to_list(data)
    elif isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"File not found: {data}")
        return data
    else:
        raise ValueError("Invalid data format")
        
class Dataset:
    """
    A class to represent an image dataset of files saved in several folders.

    Args:

        data (nested list or str): The data of the dataset, which can be a list of paths with regex patterns or a numpy array.
        axis_data (str): A string representing the format of the folder nested structure. Each character must appear only once.
        axis_files (str): A string representing the format of the file dimensions. Each character must appear only once.
        scale (tuple, optional): A tuple representing the scale for the spatial dimensions. Must be the same length as the number of spatial dimensions. If None, the scale is set to (1, 1, ..., 1). Defaults to None.
        h5_key (str, optional): The key to read data from an h5 file, if applicable.

    Examples:

        Having a dataset with the folowwing structure:

        - main_folder
            - ch1
                - file_t1.tif
                - file_t2.tif
                - ...
            - ch2
                - file_t1.tif
                - file_t2.tif
                - ...

        Where each file is a 3D image saved in format 'ZYX' with scale (2., 1., 1.).
        You can create a dataset object with the following code:
        
            dataset = Dataset(data=['ch1/file{0:03d}.tif','ch2/file{0:03d}.tif'], axis_data='CT', axis_files='ZYX', scale=(2., 1., 1.))

    Attributes:

        shape (tuple): The shape of the dataset including both data and file dimensions.
        scale (tuple): The scale of the spatial dimensions.
        _data (np.ndarray): The data of the dataset, expanded from regex patterns or file paths.
        _axis (str): The combined format of data and file dimensions.
        _n_axis (int): The total number of dimensions.
        _axis_data (str): The format of the folder structure.
        _n_axis_data (int): The number of data dimensions.
        _axis_files (str): The format of the file dimensions.
        _n_axis_files (int): The number of file dimensions.
        _axis_spatial (str): The spatial dimensions in the dataset.
        _n_axis_spatial (int): The number of spatial dimensions.
        dtype (np.dtype): The data type of the images.
        _h5_key (str): The key to read data from an h5 file, if applicable.
    """

    def __init__(self, data, axis_data, axis_files, scale=None, h5_key=None):
        """
        Initializes the Dataset object with the provided data, axis information, and scale.

        Args:
            data (nested list or str): The data of the dataset, which can be a list of paths with regex patterns or a numpy array.
            axis_data (str): A string representing the format of the folder nested structure. Each character must appear only once.
            axis_files (str): A string representing the format of the file dimensions. Each character must appear only once.
            scale (tuple, optional): A tuple representing the scale for the spatial dimensions. Must be the same length as the number of spatial dimensions. If None, the scale is set to (1, 1, ..., 1). Defaults to None.
            h5_key (str, optional): The key to read data from an h5 file, if applicable.

        Examples:
            dataset = Dataset(data=[['file_c1_1.tif', 'file_c1_2.tif'], ['file_c2_1.tif', 'file_c2_2.tif']], axis_data='CT', axis_files='ZYX', scale=(1, 2, 3))
            dataset = Dataset(data='file{0:03d}.tif', axis_data='T', axis_files='ZYX', scale=(1, 2, 3))
            dataset = Dataset(data=['ch1/file{0:03d}.tif','ch2/file{0:03d}.tif'], axis_data='CT', axis_files='ZYX', scale=(1, 2, 3))
        """

        self._data = np.array(_expand_lists(data))
        
        if isinstance(axis_data, str):
            if self._data.ndim == len(axis_data):
                self._axis_data = axis_data
                self._n_axis_data = len(axis_data)
            else:
                raise ValueError(f"Axis data must have the same length as the number of dimensions in the data. Dataset has {self._data.ndim} dimensions and axis_data provided is {axis_data}.")
        else:
            raise ValueError("Invalid data format for axis_data")
        
        self._h5_key = h5_key
        img = read_file(self._data.flatten()[0], h5_key=h5_key)
        if isinstance(axis_files, str):
            if img.ndim == len(axis_files):
                self._axis_files = axis_files
                self._n_axis_files = len(axis_files)
            else:
                raise ValueError(f"Axis files must have the same length as the number of dimensions in the files. Image has {img.ndim} dimensions and axis_files is provided {axis_files}.")
        else:
            raise ValueError("Invalid data format for axis_files")
        self.dtype = img.dtype

        self._axis = self._axis_data + self._axis_files
        if len(set(self._axis)) != len(self._axis):
            raise ValueError("There are repeated symbols. Each symbol must appear only once from 'XYZTC'.")
        self._n_axis = len(self._axis)
        
        self._axis_spatial = "".join([dim for dim in self._axis if dim in "XYZ"])
        self._n_axis_spatial = len(self._axis_spatial)

        if scale is None:
            self.scale = (1,) * self._n_axis_spatial
        elif len(scale) == self._n_axis_spatial:
            self.scale = scale
        else:
            raise ValueError("Scale must be the same length as the number of spatial dimensions.")
        
        self.shape = self._data.shape + img.shape
        
        return

    def __repr__(self):
        """
        Returns a string representation of the Dataset object, including its metadata.

        Returns:
            str: A string representation of the Dataset object.
        """
        return f"Dataset(shape={self.shape}, axis={self._axis}, scale={self.scale})"

    def __getitem__(self, index):
        """
        Allows access to the dataset using array indexing and slicing.
        Args:
            index (tuple): A tuple containing the indices for each dimension.
        Returns:
            np.ndarray: The image at the specified indices.
        """
        if isinstance(index, int):
            index = (index,)
        elif self._n_axis < len(index):
            raise ValueError(f"Too many indices provided. Shape of indices should be at most {self._n_axis}.")

        data_slice = tuple(index[:self._n_axis_data])
        file_slice = tuple(index[self._n_axis_data:])

        subdata= self._data[data_slice]
        if np.prod(np.array(subdata.shape)) == 1:
            subdata = subdata.flatten()[0]
            img_total = read_file(subdata, h5_key=self._h5_key)[file_slice]
        else:
            img = read_file(subdata.flatten()[0], h5_key=self._h5_key)[file_slice]
            img_total = np.zeros([*subdata.shape, *img.shape], dtype=self.dtype)
            for idx, subfile in np.ndenumerate(subdata):
                img = read_file(subfile, h5_key=self._h5_key)[file_slice]
                img_total[idx] = img

        return img_total
    
    def to_zarr_legacy(self, file, **kwargs):
        """
        Save the dataset to a Zarr array using the legacy method.

        Parameters:
        -----------
        file : str or zarr.storage.Store
            The file path or Zarr store where the array will be saved.
        **kwargs : dict
            Additional keyword arguments to pass to `zarr.create`.

        Returns:
        --------
        None

        Notes:
        ------
        This method saves the dataset to a Zarr array without parallel processing.
        It iterates over the dataset and saves each image sequentially.
        """

        z = zarr.create(
            store=file,
            shape=self.shape,
            dtype=self.dtype,
            **kwargs
        )

        for idx in tqdm(np.ndindex(self._data.shape), desc="Saving to Zarr", total=np.prod(self._data.shape), unit="images"):
            z[idx] = self[idx]

        z.attrs['axis'] = self._axis
        z.attrs['scale'] = tuple(self.scale)

    def to_zarr_dask(self, file, **kwargs):
        """
        Save the dataset to a Zarr file using Dask for parallel processing.

        Parameters:
        -----------
        file : str or MutableMapping
            The file path or MutableMapping to save the Zarr file.
        **kwargs : dict
            Additional keyword arguments to pass to the `to_zarr` method.

        Returns:
        --------
        None

        Notes:
        ------
        This method uses Dask to parallelize the reading and saving of images.
        The `_read_image` function is delayed to allow for parallel execution.
        A progress bar is displayed using `_TqdmCallback` to show the saving progress.
        """

        @delayed
        def _read_image(path, idx):
            return path[idx]

        shape = self[list(np.ndindex(self._data.shape))[0]].shape
        delayed_arrays = [da.from_delayed(_read_image(self, idx), shape=shape, dtype=self.dtype) for idx in np.ndindex(self._data.shape)]
        new_array = da.stack(delayed_arrays).reshape(self.shape)

        with _TqdmCallback(desc="Saving to Zarr", unit="images"):
            new_array.to_zarr(file, **kwargs)

        #add attributes
        z = zarr.open_array(file, mode="r+")
        z.attrs['axis'] = self._axis
        z.attrs['scale'] = tuple(self.scale)

    def to_zarr(self, file, flavor="dask", **kwargs):
        """
        Save the dataset to a Zarr file.

        Parameters:
        file (str or store): The file path or store to save the Zarr file.
        flavor (str, optional): The method to use for saving. Options are 'legacy' or 'dask'. Default is 'dask'.
        **kwargs: Additional keyword arguments to pass to the respective saving method.

        Raises:
        ValueError: If an invalid flavor is provided.

        """

        if flavor == "legacy":
            self.to_zarr_legacy(file, **kwargs)
        elif flavor == "dask":
            self.to_zarr_dask(file, **kwargs)
        else:
            raise ValueError("Invalid flavor. Choose between 'legacy' and 'dask'.")