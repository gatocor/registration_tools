import os
import re
from collections import Counter
import numpy as np
from skimage.io import imread, imsave
import json
from copy import deepcopy
import h5py
import zarr

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

    Examples:

        Have a dataset with the folowwing structure:

        - main_folder
            - ch1
                - file1.tif
                - ...
            - ch2
                - file1.tif
                - ...

        It is 3D saved in format 'ZYX' with scale (2, 1, 1).
        You can create a dataset object with the following code:
        
        ```python
            dataset = Dataset(data=['ch1/file{0:03d}.tif','ch2/file{0:03d}.tif'], axis_data='CT', axis_files='ZYX', scale=(2, 1, 1))
        ```

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
                raise ValueError("Axis data must have the same length as the number of dimensions in the data.")
        else:
            raise ValueError("Invalid data format for axis_data")
        
        self._h5_key = h5_key
        img = read_file(self._data.flatten()[0], h5_key=h5_key)
        if isinstance(axis_files, str):
            if img.ndim == len(axis_files):
                self._axis_files = axis_files
                self._n_axis_files = len(axis_files)
            else:
                raise ValueError("Axis files must have the same length as the number of dimensions in the files.")
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
        return f"Dataset(data={self._data}, axis_data={self._axis_data}, axis_files={self._axis_files}, scale={self.scale})"

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
    
    def to_zarr(self, file, **kwargs):
        """
        Save the dataset to a Zarr array. Zarr is a format that stores multidimensional arrays in a chunked, compressed, and efficient manner.

        Parameters:
        -----------
        file : str or zarr.storage.Store
            The file path or Zarr store where the array will be saved.
        **kwargs : dict
            Additional keyword arguments to pass to `zarr.create_array`.

        Returns:
        --------
        None
        """

        z = zarr.create_array(
            store=file,
            shape=self.shape,
            dtype=self.dtype,
            **kwargs
        )

        for idx in np.ndindex(self._data.shape):
            z[idx] = self[idx]

        z.attrs['axis'] = self._axis
        z.attrs['scale'] = tuple(self.scale)
