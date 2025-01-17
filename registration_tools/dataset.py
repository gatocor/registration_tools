import os
import re
from collections import Counter
import numpy as np
from skimage.io import imread, imsave
import json

def make_tmp_image(image, path_tmp_file, downsample):
    slices = tuple(slice(None, None, ds) for ds in downsample)
    imsave(path_tmp_file, image[slices])
    return path_tmp_file

def create_dataset(data, format, numbers=None, scale=None):
    """
    Initializes the Dataset object with a list of paths with regex patterns, a list of numbers,
    a format for the numbers, and a scale for the spatial dimensions.

    Args:
        data (list): A list of paths with regex patterns.
        numbers (list): A list of numbers to be substituted in the regex paths.
        format (str): The format to be used for the numbers. It can only contain 'C', 'T', 'X', 'Y', 'Z',
              each representing a dimension in a multidimensional image. Each character must appear only once.
        scale (tuple): A tuple representing the scale for the spatial dimensions. Must be the same length as ndim_spatial.
    """
    self = Dataset()

    self._data = data
    self._channels_separated = True
    if hasattr(data, '__iter__') and not isinstance(data, (str, np.ndarray)):
        self._data = list(data)
        data_format = self.get_format(self._data[0])
        if not all(self.get_format(d) == data_format for d in self._data):
            raise ValueError("All elements in the data list must have the same format (either all file paths, all regex patterns, or all numpy arrays)")
        self._dtype = self.get_format(self._data[0])
    else:
        self._dtype = self.get_format(self._data)
        self._channels_separated = False

    if self._dtype == "regex" and numbers is None:
        raise ValueError("Numbers must be provided when data is a regex pattern")
    elif self._dtype == "regex":
        self._numbers = numbers

    self.check_files_exist()

    if not all(f in {"X", "T", "Y", "Z", "C"} for f in format):
        raise ValueError("Format can only contain 'X', 'Y', 'Z', 'T', and 'C'")
    if len(set(format)) != len(format):
        raise ValueError("Format can only contain each of 'X', 'Y', 'Z', 'T', and 'C' once")
    if self._dtype == "regex" and "T" in format:
        raise ValueError("Format cannot contain 'T' when data is a regex pattern")
    elif self._dtype != "regex" and "T" not in format:
        raise ValueError("Format must contain 'T' for time dimension when data is not a regex pattern")
    self._format = format

    self._ndim_spatial = sum(dim in format for dim in "XYZ")

    if scale is not None:
        if len(scale) != self._ndim_spatial:
            raise ValueError("Scale must be the same length as the number of spatial dimensions (ndim_spatial)")
        self._scale = scale
    else:
        self._scale = (1,) * self._ndim_spatial

    self._pos_symbol = {symbol:i for i, symbol in enumerate(self._format)}

    self._shape, self._ndim = self.check_consistent_shapes()

    if self._dtype == "array":
        self._numbers = [i for i in range(self._shape[self._pos_symbol["T"]])]
    elif self._dtype == "file":
        self._numbers = [i for i in range(self._shape[self._pos_symbol["T"]])]

    if "C" in format and self._channels_separated:
        raise ValueError("If 'C' is in the format, only one path can be provided")
    elif self._channels_separated:
        self._nchannels = len(self._data)
    elif "C" in format:
        self._nchannels = self._shape[self._pos_symbol["C"]]
    else:
        self._nchannels = 1

    return self

def load_dataset(directory):
    """
    Loads the dataset metadata from a JSON file in the specified directory and creates a Dataset object.

    Args:
        directory (str): The directory where the metadata JSON file is located.
    Returns:
        Dataset: The Dataset object created from the metadata.
    """

    metadata_path = os.path.join(directory, "dataset.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    dataset = Dataset()
    dataset._data = metadata["data"]
    dataset._dtype = metadata["dtype"]
    dataset._format = metadata["format"]
    dataset._shape = tuple(metadata["shape"])
    dataset._ndim = metadata["ndim"]
    dataset._ndim_spatial = metadata["ndim_spatial"]
    dataset._nchannels = metadata["nchannels"]
    dataset._channels_separated = metadata["channels_separated"]
    dataset._numbers = metadata["numbers"]
    dataset._scale = metadata["scale"]
    dataset._pos_symbol = metadata["pos_symbol"]
    dataset._transformation = metadata["transformations"]
    dataset._save_folder = directory

    return dataset

class Dataset:
    """
    A class to represent a dataset of multidimensional images.

    Attributes:
        _data (list or np.ndarray): The data of the dataset.
        _dtype (str): The type of the data ('array', 'file', or 'regex').
        _format (str): The format of the data dimensions.
        _shape (tuple): The shape of the data.
        _ndim (int): The number of dimensions of the data.
        _ndim_spatial (int): The number of spatial dimensions of the data.
        _nchannels (int): The number of channels in the data.
        _channels_separated (bool): Whether the channels are separated.
        _numbers (list): The list of numbers for regex patterns.
        _scale (tuple): The scale of the spatial dimensions.
        _pos_symbol (dict): The position of each symbol in the format.
        _transformation (dict): The transformation applied to the data.
        _save_folder (str): The folder where the dataset is saved.
    """

    def __init__(self):
        """
        Initializes the Dataset object with default values.
        """
        self._data = None
        self._dtype = None
        self._format = None
        self._shape = None
        self._ndim = None
        self._ndim_spatial = None
        self._nchannels = None
        self._channels_separated = None
        self._numbers = None
        self._scale = None
        self._pos_symbol = None
        self._transformation = {}
        self._save_folder = None
        
    def get_format(self, data):
        """
        Determines the format of the data.

        Args:
            data (str or np.ndarray): The data to determine the format of.

        Returns:
            str: The format of the data ('array', 'file', or 'regex').
        """
        if isinstance(data, np.ndarray):
            return "array"
        elif isinstance(data, str):
            if os.path.isfile(data):
                return "file"
            elif re.search(r'\{.*\}', data):
                return "regex"
            else:
                raise ValueError("Invalid data format")
        else:
            raise ValueError("Data must be a numpy array, a file path, a regex pattern, or an iterator containing those.")
            
    def check_files_exist(self):
        """
        Checks if all files exist based on the paths with regex patterns and numbers.

        Raises:
            FileNotFoundError: If any file does not exist.
        """
        if self._dtype == "file":
            iterate = [self._data] if not self._channels_separated else self._data
            for path in iterate:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")
        elif self._dtype == "regex":
            iterate = [self._data] if not self._channels_separated else self._data
            for path_with_regex in iterate:
                for number in self._numbers:
                    path = path_with_regex.format(number)
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"File not found: {path}")

        return
    
    def check_consistent_shapes(self):
        """
        Checks if all files have consistent shapes and number of dimensions.

        Returns:
            tuple: A tuple containing the shape and number of dimensions if consistent.

        Raises:
            ValueError: If the shapes or number of dimensions are inconsistent.
        """

        expected_shape = None
        expected_ndim = None

        if self._dtype == "file":
            iterate = [self._data] if not self._channels_separated else self._data
            for path in iterate:
                image = imread(path)
                if expected_shape is None:
                    expected_shape = image.shape
                    expected_ndim = image.ndim
                else:
                    if image.shape != expected_shape or image.ndim != expected_ndim:
                        raise ValueError(f"Inconsistent shape or number of dimensions found in file: {path} with respect other files.")
        elif self._dtype == "regex":
            iterate = [self._data] if not self._channels_separated else self._data
            for path_with_regex in iterate:
                number = self._numbers[0]
                path = path_with_regex.format(number)
                if os.path.exists(path):
                    image = imread(path)
                    if expected_shape is None:
                        expected_shape = image.shape
                        expected_ndim = image.ndim
                    else:
                        if image.shape != expected_shape or image.ndim != expected_ndim:
                            raise ValueError(f"Inconsistent shape or number of dimensions found in file: {path} with respect other files.")
        else:
            iterate = [self._data] if not self._channels_separated else self._data
            for image in iterate:
                if expected_shape is None:
                    expected_shape = image.shape
                    expected_ndim = image.ndim
                else:
                    if image.shape != expected_shape or image.ndim != expected_ndim:
                        raise ValueError(f"Inconsistent shape or number of dimensions found in files.")

        if len(image.shape) != len(self._format):
            raise ValueError(f"Number of dimensions in data does not match the length of the format.")

        return expected_shape, expected_ndim

    def __repr__(self):
        """
        Returns a string representation of the Dataset object, including its metadata.

        Returns:
            str: A string representation of the Dataset object.
        """
        return (f"Dataset(dtype={self._dtype}, format={self._format}, shape={self._shape}, "
                f"ndim={self._ndim}, nchannels={self._nchannels}, channels_separated={self._channels_separated}, scale={self._scale})")
    
    def get_metadata(self):
        """
        Returns the metadata of the dataset.

        Returns:
            dict: A dictionary containing the metadata of the dataset.
        """
        metadata = {
            "data": self._data,
            "dtype": self._dtype,
            "format": self._format,
            "shape": self._shape,
            "ndim": self._ndim,
            "ndim_spatial": self._ndim_spatial,
            "nchannels": self._nchannels,
            "channels_separated": self._channels_separated,
            "numbers": self._numbers,
            "scale": self._scale,
            "pos_symbol": self._pos_symbol,
            "transformations": self._transformation
        }
        return metadata

    def get_time_data(self, index, channel=0, downsample=None):
        """
        Retrieves the image at the specified time index.

        Args:
            index (int): The time index to retrieve the image from.
            channel (int): The channel to retrieve the image from.
            downsample (tuple): The downsample factors for each spatial dimension. If None, no downsample is applied (default).

        Returns:
            np.ndarray: The image at the specified time index.

        Raises:
            ValueError: If the downsample length is not equal to the number of spatial dimensions.
        """

        if downsample is not None:
            if len(downsample) != self._ndim_spatial:
                raise ValueError("Downsample must be the same length as the number of spatial dimensions (ndim_spatial)")
        else:
            downsample = (1,) * self._ndim_spatial

        slices = tuple(slice(None, None, ds) for ds in downsample)

        if self._dtype == "file" and not self._channels_separated:
            path = self._data
            if "C" in self._format:
                image = imread(path)
                if self._pos_symbol["T"] > self._pos_symbol["C"]:
                    return image.take(index, axis=self._pos_symbol["T"]).take(channel, axis=self._pos_symbol["C"])[slices]
                else:
                    return image.take(channel, axis=self._pos_symbol["C"]).take(index, axis=self._pos_symbol["T"])[slices]
            else:
                return imread(path).take(index, axis=self._pos_symbol["T"])[slices]
        elif self._dtype == "file" and self._channels_separated:
            path = self._data[channel]
            image = imread(path)
            return image.take(index, axis=self._pos_symbol["T"])[slices]
        elif self._dtype == "regex" and not self._channels_separated:
            path = self._data.format(self._numbers[index])
            image = imread(path)
            if "C" in self._format:
                return image.take(channel, axis=self._pos_symbol["C"])[slices]
            else:
                return image[slices]
        elif self._dtype == "regex" and self._channels_separated:
            path = self._data[channel].format(self._numbers[index])
            image = imread(path)
            return image[slices]
        elif self._dtype == "array" and not self._channels_separated:
            if "C" in self._format:
                if self._pos_symbol["T"] > self._pos_symbol["C"]:
                    return self._data.take(index, axis=self._pos_symbol["T"]).take(channel, axis=self._pos_symbol["C"])[slices]
                else:
                    return self._data.take(channel, axis=self._pos_symbol["C"]).take(index, axis=self._pos_symbol["T"])[slices]
            else:
                return self._data.take(index, axis=self._pos_symbol["T"])[slices]
        elif self._dtype == "array" and self._channels_separated:
            return self._data[channel].take(index, axis=self._pos_symbol["T"])[slices]

    def get_time_file(self, path_tmp_file, index, channel=0, downsample=None):
        """
        Retrieves the image at the specified time index and saves it to a temporary file.

        Args:
            path_tmp_file (str): The temporary file path to save the downsampled image.
            index (int): The time index to retrieve the image from.
            channel (int): The channel to retrieve the image from.
            downsample (tuple): The downsample factors for each spatial dimension. If None, no downsample is applied (default).

        Returns:
            str: The path to the temporary file containing the image.

        Raises:
            ValueError: If the downsample length is not equal to the number of spatial dimensions.
        """

        if downsample is not None:
            if len(downsample) != self._ndim_spatial:
                raise ValueError("Downsample must be the same length as the number of spatial dimensions (ndim_spatial)")
        else:
            downsample = (1,) * self._ndim_spatial

        if self._dtype == "file" and not self._channels_separated:
            path = self._data
            if "C" in self._format:
                image = imread(path)
                if self._pos_symbol["T"] > self._pos_symbol["C"]:
                    return make_tmp_image(image.take(index, axis=self._pos_symbol["T"]).take(channel, axis=self._pos_symbol["C"]), path_tmp_file, downsample)
                else:
                    return make_tmp_image(image.take(channel, axis=self._pos_symbol["C"]).take(index, axis=self._pos_symbol["T"]), path_tmp_file, downsample)
            else:
                return make_tmp_image(imread(path).take(index, axis=self._pos_symbol["T"]), path_tmp_file, downsample)
        elif self._dtype == "file" and self._channels_separated:
            path = self._data[channel]
            image = imread(path)
            return make_tmp_image(image.take(index, axis=self._pos_symbol["T"]), path_tmp_file, downsample)
        elif self._dtype == "regex" and not self._channels_separated:
            path = self._data.format(self._numbers[index])
            if "C" in self._format:
                image = imread(path)
                return make_tmp_image(image.take(channel, axis=self._pos_symbol["C"]), path_tmp_file, downsample)
            elif any(i != 1 for i in downsample):
                image = imread(path)
                return make_tmp_image(image, path_tmp_file, downsample)
            else:
                return path
        elif self._dtype == "regex" and self._channels_separated:
            path = self._data[channel].format(self._numbers[index])
            if any(i != 1 for i in downsample):
                image = imread(path)
                return make_tmp_image(image, path_tmp_file, downsample)
            else:
                return path
        elif self._dtype == "array" and not self._channels_separated:
            if "C" in self._format:
                if self._pos_symbol["T"] > self._pos_symbol["C"]:
                    return make_tmp_image(self._data.take(index, axis=self._pos_symbol["T"]).take(channel, axis=self._pos_symbol["C"]), path_tmp_file, downsample)
                else:
                    return make_tmp_image(self._data.take(channel, axis=self._pos_symbol["C"]).take(index, axis=self._pos_symbol["T"]), path_tmp_file, downsample)
            else:
                return make_tmp_image(self._data.take(index, axis=self._pos_symbol["T"]), path_tmp_file, downsample)
        elif self._dtype == "array" and self._channels_separated:
            return self._data[channel].take(index, axis=self._pos_symbol["T"])
        else:
            raise ValueError("No file has been generated.")

    def save(self, directory):
        """
        Saves the dataset metadata to a JSON file in the specified directory.

        Args:
            directory (str): The directory where the metadata JSON file will be saved.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        metadata = self.get_metadata()

        if self._dtype == "array":
            if self._channels_separated:
                data = []
                for i, array in enumerate(self._data):
                    channel_dir = os.path.join(directory, f"files_ch{i}")
                    if not os.path.exists(channel_dir):
                        os.makedirs(channel_dir)
                    file_path = os.path.join(channel_dir, f"data.tif")
                    imsave(file_path, array)
                    data.append(file_path)
            else:
                channel_dir = os.path.join(directory, "files_ch0")
                if not os.path.exists(channel_dir):
                    os.makedirs(channel_dir)
                file_path = os.path.join(channel_dir, f"data.tif")
                imsave(file_path, self._data)
                data = file_path

            dtype = "file"
        else:
            data = self._data
            dtype = self._dtype

        metadata["data"] = data
        metadata["dtype"] = dtype
        
        metadata_path = os.path.join(directory, "dataset.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def get_data_iterator(self, channel=0, downsample=None, return_position=False, numbers=None):
        """
        Returns a DataIterator for the dataset.

        Args:
            channel (int): The channel to iterate over.
            downsample (tuple): The downsample factors for each spatial dimension.
            return_position (bool): Whether to return the position along with the data.
            numbers (list): A list of specific numbers to iterate over. Default is None (iterate over all numbers).

        Returns:
            DataIterator: An iterator over the dataset.
        """
        return DataIterator(self, channel, downsample, return_position, numbers)

    def get_file_iterator(self, path_tmp_file, channel=0, downsample=None, return_position=False, numbers=None):
        """
        Returns a FileIterator for the dataset.

        Args:
            path_tmp_file (str): The temporary file path to save the downsampled images.
            channel (int): The channel to iterate over.
            downsample (tuple): The downsample factors for each spatial dimension.
            return_position (bool): Whether to return the position along with the file path.
            numbers (list): A list of specific numbers to iterate over. Default is None (iterate over all numbers).

        Returns:
            FileIterator: An iterator over the dataset.
        """
        return FileIterator(self, path_tmp_file, channel, downsample, return_position, numbers)

    def get_spatial_shape(self):
        """
        Returns the spatial dimensions of the dataset in the order specified by the format.

        Returns:
            tuple: A tuple representing the spatial dimensions of the dataset in the order specified by the format.
        """
        return tuple(self._shape[self._pos_symbol[dim]] for dim in self._format if dim in "XYZ")
    
    def numbers_to_positions(self, numbers=None):
        """
        Converts a list of numbers to their corresponding positions in the dataset.

        Args:
            dataset (Dataset): The dataset object.
            numbers (list): A list of numbers to convert to positions.

        Returns:
            list: A list of positions corresponding to the numbers.
        """
        if numbers is None:
            numbers = self._numbers

        positions = []
        for i in numbers:
            pos = np.where(np.array(self._numbers) == i)[0][0]
            if pos is not None:
                positions.append(pos)
            else:
                raise ValueError(f"Number {i} not found in dataset numbers")
        return positions

class DataIterator:
    def __init__(self, dataset, channel=0, downsample=None, return_position=False, numbers=None):
        """
        Initializes the DataIterator object.

        Args:
            dataset (Dataset): The dataset to iterate over.
            channel (int): The channel to iterate over.
            downsample (tuple): The downsample factors for each spatial dimension.
            return_position (bool): Whether to return the position along with the data.
            numbers (list): A list of specific numbers to iterate over. Default is None (iterate over all numbers).
        """

        positions = dataset.numbers_to_positions(numbers)

        self._dataset = dataset
        self._channel = channel
        self._downsample = downsample
        self._index = 0
        self._max_index = len(positions)
        self._return_position = return_position
        self._positions = positions

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < self._max_index:
            result = self._dataset.get_time_data(self._positions[self._index], self._channel, self._downsample)
            self._index += 1
            if self._return_position:
                return self._index - 1, result
            else:
                return result
        else:
            raise StopIteration

class FileIterator:
    def __init__(self, dataset, path_tmp_file, channel=0, downsample=None, return_position=False, numbers=None):
        """
        Initializes the FileIterator object.

        Args:
            dataset (Dataset): The dataset to iterate over.
            path_tmp_file (str): The temporary file path to save the downsampled images.
            channel (int): The channel to iterate over.
            downsample (tuple): The downsample factors for each spatial dimension.
            return_position (bool): Whether to return the position along with the file path.
            numbers (list): A list of specific numbers to iterate over. Default is None (iterate over all numbers).
        """

        positions = dataset.numbers_to_positions(numbers)

        self._dataset = dataset
        self._path_tmp_file = path_tmp_file
        self._channel = channel
        self._downsample = downsample
        self._index = 0
        self._max_index = len(positions)
        self._return_position = return_position
        self._positions = positions

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < self._max_index:
            result = self._dataset.get_time_file(self._path_tmp_file, self._positions[self._index], self._channel, self._downsample)
            self._index += 1
            if self._return_position:
                return self._index - 1, result
            else:
                return result
        else:
            raise StopIteration
