import os
import re
import unittest
import shutil
import warnings
from skimage._shared.utils import warn
from registration_tools.dataset import create_dataset
from registration_tools.data import sphere
from registration_tools.registration import register, get_pyramid_levels
from skimage.io import imread
import numpy as np

class TestRegistration(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        warnings.filterwarnings("ignore", category=UserWarning, message=".*is a low contrast image")
        self.test_folder = os.path.join(os.path.dirname(__file__), 'test_folder')
        self.save_folder = os.path.join(os.path.dirname(__file__), 'save_folder')
        sphere(self.test_folder, num_images=10, image_size=100, num_channels=3, min_radius=5, max_radius=5, jump=2, stride=(1, 1, 1))

    @classmethod
    def tearDownClass(self):
        if os.path.exists(self.test_folder):
            shutil.rmtree(self.test_folder)
        if os.path.exists(self.save_folder):
            shutil.rmtree(self.save_folder)

    def test_register_invalid_dataset(self):
        with self.assertRaises(ValueError) as context:
            register(
                dataset="invalid_dataset",
                save_path=self.test_folder,
                use_channel=0
            )
        self.assertEqual(str(context.exception), "dataset must be an instance of the Dataset class.")

    def test_register_invalid_save_path(self):
        with self.assertRaises(ValueError) as context:
            path = os.path.join(self.test_folder, "channel_0", 'sphere_{:02d}.tiff')
            register(
                dataset=create_dataset(path, format="XYZ", numbers=[1,2,3,4,6,7,8,9]),
                save_path=self.test_folder,
                use_channel=0
            )
        self.assertEqual(str(context.exception), "save_path must be an empty directory.")

    def test_register_invalid_use_channel(self):
        dataset=create_dataset(os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"), "XYZ", numbers=[1,2,3,4,6,7,8,9])
        with self.assertRaises(ValueError) as context:
            register(
                dataset=dataset,
                save_path=self.save_folder,
                use_channel=3
            )
        self.assertEqual(str(context.exception), "use_channel must be an integer between 0 and 0.")

    def test_register_invalid_numbers(self):
        dataset=create_dataset(os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"), "XYZ", numbers=[1,2,3,4,6,7,8,9])
        # shutil.rmtree(self.save_folder)
        with self.assertRaises(ValueError) as context:
            register(
                dataset=dataset,
                save_path=self.save_folder,
                use_channel=0,
                numbers=[4, 5]
            )
        self.assertEqual(str(context.exception), "All elements in numbers must be present in dataset._numbers.")

    def test_register_invalid_save_behavior(self):
        dataset=create_dataset(os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"), "XYZ", numbers=[1,2,3,4,6,7,8,9])
        shutil.rmtree(self.save_folder)
        with self.assertRaises(ValueError) as context:
            register(
                dataset=dataset,
                save_path=self.save_folder,
                use_channel=0,
                save_behavior="InvalidBehavior"
            )
        self.assertEqual(str(context.exception), "save_behavior should be one of the following: ['NotOverwrite', 'Continue', 'Overwrite'].")

    def test_register_invalid_registration_direction(self):
        dataset=create_dataset(os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"), "XYZ", numbers=[1,2,3,4,6,7,8,9])
        # shutil.rmtree(self.save_folder)
        with self.assertRaises(ValueError) as context:
            register(
                dataset=dataset,
                save_path=self.save_folder,
                use_channel=0,
                registration_direction="invalid_direction"
            )
        self.assertEqual(str(context.exception), "registration_direction must be either None or 'forward' or 'backward'.")

    def test_register_valid(self):
        dataset = create_dataset(
            [
                os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"),
                os.path.join(self.test_folder, "channel_1", "sphere_{:02d}.tiff"),
                os.path.join(self.test_folder, "channel_2", "sphere_{:02d}.tiff")             
            ]
            , "XYZ", numbers=[0,1,2,3,4,5,6,7,8,9], scale=(1,1,1))
        shutil.rmtree(self.save_folder)
        register(
            dataset=dataset,
            save_path=self.save_folder,
            use_channel=0,
            registration_type="translation",
            save_behavior="Overwrite",
            registration_direction="backward",
            make_vectorfield=True,
        )
        # Check if the registration files are created
        registered_files = os.listdir(os.path.join(self.save_folder, "files_ch0"))
        self.assertTrue(len(registered_files) > 0)

        # Check if the max from all files is around the center of the image

        for channel in range(3):
            centers = []
            for file in os.listdir(os.path.join(self.save_folder, f"files_ch{channel}")):
                image = imread(os.path.join(self.save_folder, f"files_ch{channel}", file))
                max_pos = np.unravel_index(np.argmax(image), image.shape)
                centers.append(max_pos)

            centers = np.array(centers)
            mean_center = np.mean(centers, axis=0)
            expected_center = np.array(image.shape) // 2

            self.assertTrue(np.allclose(mean_center, expected_center, atol=2), f"Mean center {mean_center} in channel {channel} is not around the expected center {expected_center}")

    def test_get_pyramid_levels(self):
        dataset = create_dataset(
            os.path.join(self.test_folder, "channel_0", "sphere_{:02d}.tiff"),
            "XYZ",
            numbers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            scale=(1, 1, 1)
        )
        lowest_level, highest_level = get_pyramid_levels(dataset, maximum_size=50, verbose=False)
        self.assertEqual(lowest_level, 1)
        self.assertEqual(highest_level, 2)

if __name__ == "__main__":
    unittest.main()
