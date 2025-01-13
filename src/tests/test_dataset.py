import os
import unittest
import numpy as np
from skimage.io import imsave, imread
from registration_tools.dataset import create_dataset, load_dataset, Dataset
import shutil

class TestDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create temporary files for testing
        cls.temp_dir = 'temp_test_dir'
        os.makedirs(cls.temp_dir, exist_ok=True)
        cls.file_paths = []
        for i in range(3):
            file_path = os.path.join(cls.temp_dir, f'image_{i:03d}.tif')
            imsave(file_path, np.random.rand(10, 10))
            cls.file_paths.append(file_path)

        cls.temp_file_monochannel = 'temp_test_file.tif'
        imsave(cls.temp_file_monochannel, np.random.rand(5, 10, 10))

        cls.temp_file_multichannel = 'temp_test_file2.tif'
        imsave(cls.temp_file_multichannel, np.random.rand(5, 5, 10, 10))

    @classmethod
    def tearDownClass(cls):
        # Remove temporary files after testing
        for file_path in cls.file_paths:
            os.remove(file_path)
        shutil.rmtree(cls.temp_dir)

        os.remove(cls.temp_file_monochannel)

        os.remove(cls.temp_file_multichannel)

    def test_dataset_initialization_with_monochannel_files(self):
        dataset = create_dataset(data=self.temp_file_monochannel, format='TXY')
        self.assertEqual(dataset._dtype, 'file')
        self.assertEqual(dataset._format, 'TXY')
        self.assertEqual(dataset._shape, (5, 10, 10))
        self.assertEqual(dataset._ndim, 3)
        self.assertEqual(dataset._nchannels, 1)
        self.assertFalse(dataset._channels_separated)

    def test_dataset_initialization_with_multichannel_files(self):
        dataset = create_dataset(data=self.temp_file_multichannel, format='TCXY')
        self.assertEqual(dataset._dtype, 'file')
        self.assertEqual(dataset._format, 'TCXY')
        self.assertEqual(dataset._shape, (5, 5, 10, 10))
        self.assertEqual(dataset._ndim, 4)
        self.assertEqual(dataset._nchannels, 5)
        self.assertFalse(dataset._channels_separated)

    def test_dataset_initialization_with_multiple_files(self):
        dataset = create_dataset(data=[self.temp_file_monochannel]*2, format='TXY')
        self.assertEqual(dataset._dtype, 'file')
        self.assertEqual(dataset._format, 'TXY')
        self.assertEqual(dataset._shape, (5, 10, 10))
        self.assertEqual(dataset._ndim, 3)
        self.assertEqual(dataset._nchannels, 2)
        self.assertTrue(dataset._channels_separated)

    def test_dataset_initialization_with_regex_single_channel(self):
        regex_pattern = os.path.join(self.temp_dir, 'image_{:03d}.tif')
        dataset = create_dataset(data=regex_pattern, format='XY', numbers=list(range(3)))
        self.assertEqual(dataset._dtype, 'regex')
        self.assertEqual(dataset._format, 'XY')
        self.assertEqual(dataset._shape, (10, 10))
        self.assertEqual(dataset._ndim, 2)
        self.assertEqual(dataset._nchannels, 1)
        self.assertFalse(dataset._channels_separated)

    def test_dataset_initialization_with_regex_pattern_multiple_channels(self):
        regex_pattern = os.path.join(self.temp_dir, 'image_{:03d}.tif')
        dataset = create_dataset(data=[regex_pattern]*2, format='XY', numbers=list(range(3)))
        self.assertEqual(dataset._dtype, 'regex')
        self.assertEqual(dataset._format, 'XY')
        self.assertEqual(dataset._shape, (10, 10))
        self.assertEqual(dataset._ndim, 2)
        self.assertEqual(dataset._nchannels, 2)
        self.assertTrue(dataset._channels_separated)

    def test_dataset_initialization_with_numpy_array_single_channel(self):
        data = np.random.rand(3, 10, 10)
        dataset = create_dataset(data=data, format='TXY')
        self.assertEqual(dataset._dtype, 'array')
        self.assertEqual(dataset._format, 'TXY')
        self.assertEqual(dataset._shape, (3, 10, 10))
        self.assertEqual(dataset._ndim, 3)
        self.assertEqual(dataset._nchannels, 1)
        self.assertFalse(dataset._channels_separated)

    def test_dataset_initialization_with_numpy_array_multiple_channels(self):
        data = np.random.rand(3, 10, 10)
        dataset = create_dataset(data=[data]*2, format='TXY')
        self.assertEqual(dataset._dtype, 'array')
        self.assertEqual(dataset._format, 'TXY')
        self.assertEqual(dataset._shape, (3, 10, 10))
        self.assertEqual(dataset._ndim, 3)
        self.assertEqual(dataset._nchannels, 2)
        self.assertTrue(dataset._channels_separated)

    def test_load_dataset_with_monochannel_files(self):
        dataset = create_dataset(data=self.temp_file_monochannel, format='TXY')
        dataset.save(self.temp_dir)
        loaded_dataset = load_dataset(self.temp_dir)
        self.assertEqual(dataset._dtype, loaded_dataset._dtype)
        self.assertEqual(dataset._format, loaded_dataset._format)
        self.assertEqual(dataset._shape, loaded_dataset._shape)
        self.assertEqual(dataset._ndim, loaded_dataset._ndim)
        self.assertEqual(dataset._nchannels, loaded_dataset._nchannels)
        self.assertEqual(dataset._channels_separated, loaded_dataset._channels_separated)

    def test_load_dataset_with_multichannel_files(self):
        dataset = create_dataset(data=self.temp_file_multichannel, format='TCXY')
        dataset.save(self.temp_dir)
        loaded_dataset = load_dataset(self.temp_dir)
        self.assertEqual(dataset._dtype, loaded_dataset._dtype)
        self.assertEqual(dataset._format, loaded_dataset._format)
        self.assertEqual(dataset._shape, loaded_dataset._shape)
        self.assertEqual(dataset._ndim, loaded_dataset._ndim)
        self.assertEqual(dataset._nchannels, loaded_dataset._nchannels)
        self.assertEqual(dataset._channels_separated, loaded_dataset._channels_separated)

    def test_load_dataset_with_multiple_files(self):
        dataset = create_dataset(data=[self.temp_file_monochannel]*2, format='TXY')
        dataset.save(self.temp_dir)
        loaded_dataset = load_dataset(self.temp_dir)
        self.assertEqual(dataset._dtype, loaded_dataset._dtype)
        self.assertEqual(dataset._format, loaded_dataset._format)
        self.assertEqual(dataset._shape, loaded_dataset._shape)
        self.assertEqual(dataset._ndim, loaded_dataset._ndim)
        self.assertEqual(dataset._nchannels, loaded_dataset._nchannels)
        self.assertEqual(dataset._channels_separated, loaded_dataset._channels_separated)

    def test_load_dataset_with_regex_single_channel(self):
        regex_pattern = os.path.join(self.temp_dir, 'image_{:03d}.tif')
        dataset = create_dataset(data=regex_pattern, format='XY', numbers=list(range(3)))
        dataset.save(self.temp_dir)
        loaded_dataset = load_dataset(self.temp_dir)
        self.assertEqual(dataset._dtype, loaded_dataset._dtype)
        self.assertEqual(dataset._format, loaded_dataset._format)
        self.assertEqual(dataset._shape, loaded_dataset._shape)
        self.assertEqual(dataset._ndim, loaded_dataset._ndim)
        self.assertEqual(dataset._nchannels, loaded_dataset._nchannels)
        self.assertEqual(dataset._channels_separated, loaded_dataset._channels_separated)

    def test_load_dataset_with_regex_pattern_multiple_channels(self):
        regex_pattern = os.path.join(self.temp_dir, 'image_{:03d}.tif')
        dataset = create_dataset(data=[regex_pattern]*2, format='XY', numbers=list(range(3)))
        dataset.save(self.temp_dir)
        loaded_dataset = load_dataset(self.temp_dir)
        self.assertEqual(dataset._dtype, loaded_dataset._dtype)
        self.assertEqual(dataset._format, loaded_dataset._format)
        self.assertEqual(dataset._shape, loaded_dataset._shape)
        self.assertEqual(dataset._ndim, loaded_dataset._ndim)
        self.assertEqual(dataset._nchannels, loaded_dataset._nchannels)
        self.assertEqual(dataset._channels_separated, loaded_dataset._channels_separated)

    def test_load_dataset_with_numpy_array_single_channel(self):
        data = np.random.rand(3, 10, 10)
        dataset = create_dataset(data=data, format='TXY')
        dataset.save(self.temp_dir)
        loaded_dataset = load_dataset(self.temp_dir)
        self.assertEqual("file", loaded_dataset._dtype)
        self.assertEqual(dataset._format, loaded_dataset._format)
        self.assertEqual(dataset._shape, loaded_dataset._shape)
        self.assertEqual(dataset._ndim, loaded_dataset._ndim)
        self.assertEqual(dataset._nchannels, loaded_dataset._nchannels)
        self.assertEqual(dataset._channels_separated, loaded_dataset._channels_separated)

    def test_load_dataset_with_numpy_array_multiple_channels(self):
        data = np.random.rand(3, 10, 10)
        dataset = create_dataset(data=[data]*2, format='TXY')
        dataset.save(self.temp_dir)
        loaded_dataset = load_dataset(self.temp_dir)
        self.assertEqual("file", loaded_dataset._dtype)
        self.assertEqual(dataset._format, loaded_dataset._format)
        self.assertEqual(dataset._shape, loaded_dataset._shape)
        self.assertEqual(dataset._ndim, loaded_dataset._ndim)
        self.assertEqual(dataset._nchannels, loaded_dataset._nchannels)
        self.assertEqual(dataset._channels_separated, loaded_dataset._channels_separated)

    def test_get_time_data_with_monochannel_files(self):
        dataset = create_dataset(data=self.temp_file_monochannel, format='TXY')
        time_data = dataset.get_time_data(1)
        expected_data = imread(self.temp_file_monochannel)[1]
        np.testing.assert_array_equal(time_data, expected_data)

    def test_get_time_data_with_multichannel_files(self):
        dataset = create_dataset(data=self.temp_file_multichannel, format='TCXY')
        time_data = dataset.get_time_data(1, channel=2)
        expected_data = imread(self.temp_file_multichannel)[1, 2, :, :]
        np.testing.assert_array_equal(time_data, expected_data)

    def test_get_time_data_with_multiple_files(self):
        dataset = create_dataset(data=[self.temp_file_monochannel]*2, format='TXY')
        time_data = dataset.get_time_data(1, channel=1)
        expected_data = imread(self.temp_file_monochannel)[1]
        np.testing.assert_array_equal(time_data, expected_data)

    def test_get_time_data_with_regex_single_channel(self):
        regex_pattern = os.path.join(self.temp_dir, 'image_{:03d}.tif')
        dataset = create_dataset(data=regex_pattern, format='XY', numbers=list(range(3)))
        time_data = dataset.get_time_data(1)
        expected_data = imread(regex_pattern.format(1))
        np.testing.assert_array_equal(time_data, expected_data)

    def test_get_time_data_with_regex_pattern_multiple_channels(self):
        regex_pattern = os.path.join(self.temp_dir, 'image_{:03d}.tif')
        dataset = create_dataset(data=[regex_pattern]*2, format='XY', numbers=list(range(3)))
        time_data = dataset.get_time_data(1, channel=1)
        expected_data = imread(regex_pattern.format(1))
        np.testing.assert_array_equal(time_data, expected_data)

    def test_get_time_data_with_numpy_array_single_channel(self):
        data = np.random.rand(3, 10, 10)
        dataset = create_dataset(data=data, format='TXY')
        time_data = dataset.get_time_data(1)
        np.testing.assert_array_equal(time_data, data[1])

    def test_get_time_data_with_numpy_array_multiple_channels(self):
        data = np.random.rand(3, 10, 10)
        dataset = create_dataset(data=[data]*2, format='TXY')
        time_data = dataset.get_time_data(1, channel=1)
        np.testing.assert_array_equal(time_data, data[1])

    def test_get_time_data_with_numpy_array_multiple_channels2(self):
        data = np.random.rand(3, 10, 10, 3)  # 3 time points, 10x10 image, 3 channels
        dataset = create_dataset(data=data, format='TXYC')
        time_data = dataset.get_time_data(1, channel=2)
        np.testing.assert_array_equal(time_data, data[1, :, :, 2])

if __name__ == '__main__':
    unittest.main()