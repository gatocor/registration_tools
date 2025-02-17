import os
import re
import unittest
import shutil
import warnings
from skimage._shared.utils import warn
from registration_tools.dataset.dataset import Dataset
from registration_tools.data import sphere
from registration_tools.utils import project
from skimage.io import imread
import numpy as np
import zarr
class TestRegistration(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        warnings.filterwarnings("ignore", category=UserWarning, message=".*is a low contrast image")
        self.test_folder = os.path.join(os.path.dirname(__file__), 'test_folder')
        self.save_folder = os.path.join(os.path.dirname(__file__), 'save_folder')
        self.zarr_file = os.path.join(os.path.dirname(__file__), 'zarr.zarr')
        self.zarr_file2 = os.path.join(os.path.dirname(__file__), 'zarr2.zarr')
        self.results_folder = os.path.join(os.path.dirname(__file__), 'results_folder')
        self.trnsf_folder = os.path.join(os.path.dirname(__file__), 'trnsf_folder')
        sphere(self.test_folder, num_images=10, image_size=100, num_channels=3, min_radius=10, max_radius=10, jump=3, stride=(1, 2, 3))

    @classmethod
    def tearDownClass(self):
        if os.path.exists(self.test_folder):
            shutil.rmtree(self.test_folder)
        if os.path.exists(self.save_folder):
            shutil.rmtree(self.save_folder)
        if os.path.exists(self.zarr_file):
            shutil.rmtree(self.zarr_file)
        if os.path.exists(self.zarr_file2):
            shutil.rmtree(self.zarr_file2)

    def test_apply_function_dataset(self):
        dataset = Dataset(
            [
                os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"),
                os.path.join(self.test_folder, "channel_1", "sphere_{:02d}.tiff"),
                os.path.join(self.test_folder, "channel_2", "sphere_{:02d}.tiff")             
            ]
            , axis_data="CT", axis_files="XYZ", scale=(3,2,1))

        data_ = dataset[:,:,:,:,:]

        #dataset
        projection = data_.max(axis=2)
        data = project(dataset,"X")
        self.assertTrue(np.all(data==projection), f"Max projections are not the same.")

        #dataset zarr out
        if os.path.exists(self.zarr_file):
            shutil.rmtree(self.zarr_file)
        project(dataset,"X",out=self.zarr_file)
        data = zarr.open_array(self.zarr_file, mode="r")
        self.assertTrue(np.all(data==projection), f"Max projections are not the same.")

        #zarr
        dataset.to_zarr(self.zarr_file2)
        z = zarr.open_array(self.zarr_file2, mode="r")
        projection = data_.max(axis=2)
        data = project(z,"X")
        self.assertTrue(np.all(data==projection), f"Max projections are not the same.")

        #zarr out
        if os.path.exists(self.zarr_file):
            shutil.rmtree(self.zarr_file)
        project(z,"X",out=self.zarr_file)
        data = zarr.open_array(self.zarr_file, mode="r")
        self.assertTrue(np.all(data==projection), f"Max projections are not the same.")

    # def test_register_zarr(self):
    #     dataset = Dataset(
    #         [
    #             os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_1", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_2", "sphere_{:02d}.tiff")             
    #         ]
    #         , axis_data="CT", axis_files="XYZ", scale=(1,2,3))
    #     dataset.to_zarr(self.zarr_file)
    #     dataset = zarr.open_array(self.zarr_file, mode="r")
    #     transformation = Registration(registration_type="translation", registration_direction="backward")
    #     data = transformation.fit_apply(
    #         dataset=dataset,
    #         use_channel=0,
    #         save_behavior="Continue",
    #     )
    #     # Check if the registration files are created
    #     self.assertTrue(len(transformation._transfs) > 0)

    #     # Check if the max from all files is around the center of the image

    #     for channel in range(3):
    #         centers = []
    #         for t in range(10):
    #             image = data[channel][t]
    #             max_pos = np.unravel_index(np.argmax(image), image.shape)
    #             centers.append(max_pos)

    #         centers = np.array(centers)
    #         mean_center = np.mean(centers, axis=0)
    #         expected_center = np.array(image.shape) // 2

    #         self.assertTrue(np.allclose(mean_center, expected_center, atol=3), f"Mean center {mean_center} in channel {channel} is not around the expected center {expected_center}")

    # def test_register_numpy(self):
    #     dataset = Dataset(
    #         [
    #             os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_1", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_2", "sphere_{:02d}.tiff")             
    #         ]
    #         , axis_data="CT", axis_files="XYZ", scale=(3,2,1))
    #     dataset = dataset[:,:,:,:,:]
    #     transformation = Registration(registration_type="translation", registration_direction="backward")
    #     data = transformation.fit_apply(
    #         dataset=dataset,
    #         use_channel=0,
    #         save_behavior="Continue",
    #         axis="CTXYZ",
    #         scale=(3.,2.,1.)
    #     )
    #     # Check if the registration files are created
    #     self.assertTrue(len(transformation._transfs) > 0)

    #     # Check if the max from all files is around the center of the image

    #     for channel in range(3):
    #         centers = []
    #         for t in range(10):
    #             image = data[channel][t]
    #             max_pos = np.unravel_index(np.argmax(image), image.shape)
    #             centers.append(max_pos)

    #         centers = np.array(centers)
    #         mean_center = np.mean(centers, axis=0)
    #         expected_center = np.array(image.shape) // 2

    #         self.assertTrue(np.allclose(mean_center, expected_center, atol=3), f"Mean center {mean_center} in channel {channel} is not around the expected center {expected_center}")

    # def test_register_dataset2(self):
    #     dataset = Dataset(
    #         [
    #             os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_1", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_2", "sphere_{:02d}.tiff")             
    #         ]
    #         , axis_data="CT", axis_files="XYZ", scale=(3,2,1))

    #     transformation = Registration(registration_type="rigid", registration_direction="backward")
    #     data = transformation.fit_apply(
    #         dataset=dataset,
    #         use_channel=0,
    #         save_behavior="Continue",
    #     )
    #     # Check if the registration files are created
    #     self.assertTrue(len(transformation._transfs) > 0)

    #     # Check if the max from all files is around the center of the image

    #     for channel in range(3):
    #         centers = []
    #         for t in range(10):
    #             image = data[channel][t]
    #             max_pos = np.unravel_index(np.argmax(image), image.shape)
    #             centers.append(max_pos)

    #         centers = np.array(centers)
    #         mean_center = np.mean(centers, axis=0)
    #         expected_center = np.array(image.shape) // 2

    #         self.assertTrue(np.allclose(mean_center, expected_center, atol=3), f"Mean center {mean_center} in channel {channel} is not around the expected center {expected_center}")

    # def test_register_zarr2(self):
    #     dataset = Dataset(
    #         [
    #             os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_1", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_2", "sphere_{:02d}.tiff")             
    #         ]
    #         , axis_data="CT", axis_files="XYZ", scale=(1,2,3))
    #     if os.path.exists(self.zarr_file):
    #         shutil.rmtree(self.zarr_file)
    #     dataset.to_zarr(self.zarr_file)
    #     dataset = zarr.open_array(self.zarr_file, mode="r")
    #     transformation = Registration(registration_type="rigid", registration_direction="backward")
    #     data = transformation.fit_apply(
    #         dataset=dataset,
    #         use_channel=0,
    #         save_behavior="Continue",
    #     )
    #     # Check if the registration files are created
    #     self.assertTrue(len(transformation._transfs) > 0)

    #     # Check if the max from all files is around the center of the image

    #     for channel in range(3):
    #         centers = []
    #         for t in range(10):
    #             image = data[channel][t]
    #             max_pos = np.unravel_index(np.argmax(image), image.shape)
    #             centers.append(max_pos)

    #         centers = np.array(centers)
    #         mean_center = np.mean(centers, axis=0)
    #         expected_center = np.array(image.shape) // 2

    #         self.assertTrue(np.allclose(mean_center, expected_center, atol=3), f"Mean center {mean_center} in channel {channel} is not around the expected center {expected_center}")

    # def test_register_numpy2(self):
    #     dataset = Dataset(
    #         [
    #             os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_1", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_2", "sphere_{:02d}.tiff")             
    #         ]
    #         , axis_data="CT", axis_files="XYZ", scale=(3,2,1))
    #     dataset = dataset[:,:,:,:,:]
    #     transformation = Registration(registration_type="rigid", registration_direction="backward")
    #     data = transformation.fit_apply(
    #         dataset=dataset,
    #         use_channel=0,
    #         save_behavior="Continue",
    #         axis="CTXYZ",
    #         scale=(3.,2.,1.)
    #     )
    #     # Check if the registration files are created
    #     self.assertTrue(len(transformation._transfs) > 0)

    #     # Check if the max from all files is around the center of the image

    #     for channel in range(3):
    #         centers = []
    #         for t in range(10):
    #             image = data[channel][t]
    #             max_pos = np.unravel_index(np.argmax(image), image.shape)
    #             centers.append(max_pos)

    #         centers = np.array(centers)
    #         mean_center = np.mean(centers, axis=0)
    #         expected_center = np.array(image.shape) // 2

    #         self.assertTrue(np.allclose(mean_center, expected_center, atol=3), f"Mean center {mean_center} in channel {channel} is not around the expected center {expected_center}")

    # def test_register_dataset3(self):
    #     dataset = Dataset(
    #         [
    #             os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_1", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_2", "sphere_{:02d}.tiff")             
    #         ]
    #         , axis_data="CT", axis_files="XYZ", scale=(3,2,1))

    #     transformation = Registration(registration_type="vectorfield", registration_direction="backward", perfom_global_trnsf=True)
    #     data = transformation.fit_apply(
    #         dataset=dataset,
    #         use_channel=0,
    #         save_behavior="Continue",
    #     )
    #     # Check if the registration files are created
    #     self.assertTrue(len(transformation._transfs) > 0)

    #     # Check if the max from all files is around the center of the image

    #     for channel in range(3):
    #         centers = []
    #         for t in range(10):
    #             image = data[channel][t]
    #             max_pos = np.unravel_index(np.argmax(image), image.shape)
    #             centers.append(max_pos)

    #         centers = np.array(centers)
    #         mean_center = np.mean(centers, axis=0)
    #         expected_center = np.array(image.shape) // 2

    #         self.assertTrue(np.allclose(mean_center, expected_center, atol=3), f"Mean center {mean_center} in channel {channel} is not around the expected center {expected_center}")

    # def test_register_zarr3(self):
    #     dataset = Dataset(
    #         [
    #             os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_1", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_2", "sphere_{:02d}.tiff")             
    #         ]
    #         , axis_data="CT", axis_files="XYZ", scale=(1,2,3))
    #     if os.path.exists(self.zarr_file):
    #         shutil.rmtree(self.zarr_file)
    #     dataset.to_zarr(self.zarr_file)
    #     dataset = zarr.open_array(self.zarr_file, mode="r")
    #     transformation = Registration(registration_type="vectorfield", registration_direction="backward", perfom_global_trnsf=True)
    #     data = transformation.fit_apply(
    #         dataset=dataset,
    #         use_channel=0,
    #         save_behavior="Continue",
    #     )
    #     # Check if the registration files are created
    #     self.assertTrue(len(transformation._transfs) > 0)

    #     # Check if the max from all files is around the center of the image

    #     for channel in range(3):
    #         centers = []
    #         for t in range(10):
    #             image = data[channel][t]
    #             max_pos = np.unravel_index(np.argmax(image), image.shape)
    #             centers.append(max_pos)

    #         centers = np.array(centers)
    #         mean_center = np.mean(centers, axis=0)
    #         expected_center = np.array(image.shape) // 2

    #         self.assertTrue(np.allclose(mean_center, expected_center, atol=3), f"Mean center {mean_center} in channel {channel} is not around the expected center {expected_center}")

    # def test_register_numpy3(self):
    #     dataset = Dataset(
    #         [
    #             os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_1", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_2", "sphere_{:02d}.tiff")             
    #         ]
    #         , axis_data="CT", axis_files="XYZ", scale=(3,2,1))
    #     dataset = dataset[:,:,:,:,:]
    #     transformation = Registration(registration_type="vectorfield", registration_direction="backward", perfom_global_trnsf=True)
    #     data = transformation.fit_apply(
    #         dataset=dataset,
    #         use_channel=0,
    #         save_behavior="Continue",
    #         axis="CTXYZ",
    #         scale=(3.,2.,1.)
    #     )
    #     # Check if the registration files are created
    #     self.assertTrue(len(transformation._transfs) > 0)

    #     # Check if the max from all files is around the center of the image

    #     for channel in range(3):
    #         centers = []
    #         for t in range(10):
    #             image = data[channel][t]
    #             max_pos = np.unravel_index(np.argmax(image), image.shape)
    #             centers.append(max_pos)

    #         centers = np.array(centers)
    #         mean_center = np.mean(centers, axis=0)
    #         expected_center = np.array(image.shape) // 2

    #         self.assertTrue(np.allclose(mean_center, expected_center, atol=3), f"Mean center {mean_center} in channel {channel} is not around the expected center {expected_center}")

    # def test_register_vectorfield(self):
    #     dataset = Dataset(
    #         [
    #             os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_1", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_2", "sphere_{:02d}.tiff")             
    #         ]
    #         , axis_data="CT", axis_files="XYZ", scale=(1,2,3))
    #     transformation = Registration(registration_type="vectorfield", registration_direction="backward")
    #     transformation.fit(
    #         dataset=dataset,
    #         use_channel=0,
    #     )
    #     # Check if the registration files are created
    #     self.assertTrue(len(transformation._transfs) > 0)

    #     vectorfield = transformation.vectorfield()

    # def test_register_continuation(self):
    #     dataset = Dataset(
    #         [
    #             os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_1", "sphere_{:02d}.tiff"),
    #             os.path.join(self.test_folder, "channel_2", "sphere_{:02d}.tiff")             
    #         ]
    #         , "XYZ", numbers=[0,1,2,3,4,5,6,7,8,9], scale=(1,1,1))
    #     transformation = Registration(self.trnsf_folder)
    #     transformation.fit_apply(
    #         dataset=dataset,
    #         save_path=self.save_folder,
    #         use_channel=0,
    #         save_behavior="Continue"
    #     ),
    #     for i in [5,6,7,8,9]:
    #         os.remove(os.path.join(self.save_folder, "files_ch0", f"sphere_{i:02d}.tiff"))
    #     transformation.fit_apply(
    #         dataset=dataset,
    #         save_path=self.save_folder,
    #         use_channel=0,
    #         save_behavior="NotOverwrite"
    #     ),

    # def test_get_pyramid_levels(self):
    #     dataset = Dataset(
    #         os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"),
    #         "XYZ",
    #         numbers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         scale=(1, 1, 1)
    #     )
    #     lowest_level, highest_level = get_pyramid_levels(dataset, maximum_size=50, verbose=False)
    #     self.assertEqual(lowest_level, 1)
    #     self.assertEqual(highest_level, 2)

if __name__ == "__main__":
    unittest.main()
