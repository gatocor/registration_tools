import os
import unittest
import numpy as np
from skimage.io import imsave, imread
from registration_tools.dataset import Dataset
import shutil
from time import sleep
import zarr
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

        cls.temp_dir2 = 'temp_test_dir2'
        os.makedirs(cls.temp_dir2, exist_ok=True)
        cls.file_paths2 = []
        for i in range(3):
            file_path = os.path.join(cls.temp_dir2, f'image_{i:03d}.tif')
            imsave(file_path, np.random.rand(10, 10))
            cls.file_paths2.append(file_path)

        cls.temp_file_monochannel = 'temp_test_file.tif'
        imsave(cls.temp_file_monochannel, np.random.rand(5, 10, 10))

        cls.temp_file_multichannel = 'temp_test_file2.tif'
        imsave(cls.temp_file_multichannel, np.random.rand(5, 5, 10, 10))

        cls.save_zarr = 'temp_test_file.zarr'

    @classmethod
    def tearDownClass(cls):
        # Remove temporary files after testing
        for file_path in cls.file_paths:
            os.remove(file_path)
        shutil.rmtree(cls.temp_dir)
        shutil.rmtree(cls.temp_dir2)

        os.remove(cls.temp_file_monochannel)

        os.remove(cls.temp_file_multichannel)

        if os.path.exists('tmp.tiff'):
            os.remove('tmp.tiff')

        if os.path.exists(cls.save_zarr):
            shutil.rmtree(cls.save_zarr)

    def test_dataset_initialization_with_multiple_files(self):
        dataset = Dataset(data=[self.temp_file_monochannel]*2, axis_data='C', axis_files='ZYX', scale=(1.,2.,3.))
        self.assertEqual(dataset.shape, (2, 5, 10, 10))
        self.assertEqual(dataset._axis_data, "C")
        self.assertEqual(dataset._n_axis_data, 1)
        self.assertEqual(dataset._axis_files, "ZYX")
        self.assertEqual(dataset._n_axis_files, 3)
        self.assertEqual(dataset._axis_spatial, "ZYX")
        self.assertEqual(dataset._n_axis_spatial, 3)
        self.assertEqual(dataset._axis, "CZYX")
        self.assertEqual(dataset._n_axis, 4)
        self.assertEqual(dataset.dtype, float)
        self.assertEqual(dataset.scale, (1.,2.,3.))

    def test_dataset_initialization_with_regex_single_channel(self):
        regex_pattern = os.path.join(self.temp_dir, 'image_{:03d}.tif')
        dataset = Dataset(data=regex_pattern, axis_data='T', axis_files='YX', scale=(2.,3.))
        self.assertEqual(dataset.shape, (3, 10, 10))
        self.assertEqual(dataset._axis_data, "T")
        self.assertEqual(dataset._n_axis_data, 1)
        self.assertEqual(dataset._axis_files, "YX")
        self.assertEqual(dataset._n_axis_files, 2)
        self.assertEqual(dataset._axis_spatial, "YX")
        self.assertEqual(dataset._n_axis_spatial, 2)
        self.assertEqual(dataset._axis, "TYX")
        self.assertEqual(dataset._n_axis, 3)
        self.assertEqual(dataset.dtype, float)
        self.assertEqual(dataset.scale, (2.,3.))

    def test_dataset_initialization_with_regex_single_channel(self):
        regex_pattern = os.path.join(self.temp_dir, 'image_{:03d}.tif')
        regex_pattern2 = os.path.join(self.temp_dir2, 'image_{:03d}.tif')
        dataset = Dataset(data=[regex_pattern,regex_pattern2], axis_data='CT', axis_files='YX', scale=(2.,3.))
        self.assertEqual(dataset.shape, (2, 3, 10, 10))
        self.assertEqual(dataset._axis_data, "CT")
        self.assertEqual(dataset._n_axis_data, 2)
        self.assertEqual(dataset._axis_files, "YX")
        self.assertEqual(dataset._n_axis_files, 2)
        self.assertEqual(dataset._axis_spatial, "YX")
        self.assertEqual(dataset._n_axis_spatial, 2)
        self.assertEqual(dataset._axis, "CTYX")
        self.assertEqual(dataset._n_axis, 4)
        self.assertEqual(dataset.dtype, float)
        self.assertEqual(dataset.scale, (2.,3.))

    def test_dataset_access(self):
        regex_pattern = os.path.join(self.temp_dir, 'image_{:03d}.tif')
        regex_pattern2 = os.path.join(self.temp_dir2, 'image_{:03d}.tif')
        dataset = Dataset(data=[regex_pattern,regex_pattern2], axis_data='CT', axis_files='YX', scale=(2.,3.))

        self.assertEqual(dataset[0].shape, (3, 10, 10))
        self.assertEqual(dataset[:,0].shape, (2, 10, 10))
        self.assertEqual(dataset[:,:,0,:].shape, (2, 3, 10))
        self.assertEqual(dataset[:1,1:,1:5,:].shape, (1, 2, 4, 10))

    def test_dataset_to_zarr(self):
        regex_pattern = os.path.join(self.temp_dir, 'image_{:03d}.tif')
        regex_pattern2 = os.path.join(self.temp_dir2, 'image_{:03d}.tif')
        dataset = Dataset(data=[regex_pattern,regex_pattern2], axis_data='CT', axis_files='YX', scale=(2.,3.))
        dataset.to_zarr(self.save_zarr)
        data_zarr = zarr.open_array(self.save_zarr, mode='r')
        self.assertEqual(data_zarr.shape, (2, 3, 10, 10))
        self.assertEqual(data_zarr.attrs['scale'], [2.,3.])
        self.assertEqual(data_zarr.attrs['axis'], "CTYX")
        
if __name__ == '__main__':
    unittest.main()