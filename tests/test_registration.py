import unittest
import numpy as np
from registration_tools.data import sphere
import registration_tools.registration as rt_reg
import os
import shutil
import napari
import registration_tools.visualization as rt_vis
import registration_tools.utils as rt_utils

class TestExampleData(unittest.TestCase):

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("dataset_registered.zarr", ignore_errors=True)
        shutil.rmtree("trnsf", ignore_errors=True)

    def test_registration(self):
        for dim,channels,dtype,stride,out_dataset,out_trnsf,direction,stepping in [
            (2, 1, np.uint8, (1, 1, 1), None, None, 'backward', 1),
            (2, 1, np.uint8, (1, 1, 1), None, None, 'backward', 2),
            (2, 1, np.uint8, (1, 1, 1), None, None, 'backward', 10),
            (2, 1, np.uint8, (1, 1, 1), None, None, 'forward', 1),
            (2, 1, np.uint8, (1, 1, 1), None, None, 'forward', 2),
            (2, 1, np.uint8, (1, 1, 1), None, None, 'forward', 10), 
            (2, 1, np.uint8, (1, 1, 1), None, 'trnsf', 'backward', 1),
            (2, 1, np.uint8, (1, 1, 1), 'dataset_registered.zarr', None, 'backward', 1),
            (2, 1, np.uint8, (1, 1, 1), 'dataset_registered.zarr', 'trnsf', 'backward', 1),
            (2, 1, np.uint8, (2, 1, 1), None, None, 'backward', 1),
            (2, 1, np.uint8, (1, 2, 1), None, None, 'backward', 1),
            (2, 1, np.uint16, (1, 1, 1), None, None, 'backward', 1),
            (2, 1, np.uint64, (1, 1, 1), None, None, 'backward', 1),
            (2, 1, np.float64, (1, 1, 1), None, None, 'backward', 1),
            (2, 2, np.uint8, (1, 1, 1), None, None, 'backward', 1),
            (2, 2, np.uint8, (1, 1, 1), None, 'trnsf', 'backward', 1),
            (2, 2, np.uint8, (1, 1, 1), 'dataset_registered.zarr', None, 'backward', 1),
            (2, 2, np.uint8, (1, 1, 1), 'dataset_registered.zarr', 'trnsf', 'backward', 1),
            (3, 1, np.uint8, (1, 1, 1), None, None, 'backward', 1),
            (3, 1, np.uint8, (1, 1, 1), None, None, 'backward', 2),
            (3, 1, np.uint8, (1, 1, 1), None, None, 'backward', 10),
            (3, 1, np.uint8, (1, 1, 1), None, None, 'forward', 1),
            (3, 1, np.uint8, (1, 1, 1), None, None, 'forward', 2),
            (3, 1, np.uint8, (1, 1, 1), None, None, 'forward', 10),
            (3, 1, np.uint8, (1, 1, 1), None, 'trnsf', 'backward', 1),
            (3, 1, np.uint8, (1, 1, 1), 'dataset_registered.zarr', None, 'backward', 1),
            (3, 1, np.uint8, (1, 1, 1), 'dataset_registered.zarr', 'trnsf', 'backward', 1),
            (3, 1, np.uint8, (2, 1, 1), None, None, 'backward', 1),
            (3, 1, np.uint8, (1, 2, 1), None, None, 'backward', 1),
            (3, 1, np.uint8, (1, 1, 2), None, None, 'backward', 1),
            (3, 1, np.uint16, (1, 1, 1), None, None, 'backward', 1),
            (3, 1, np.uint64, (1, 1, 1), None, None, 'backward', 1),
            (3, 1, np.float64, (1, 1, 1), None, None, 'backward', 1),
            (3, 2, np.uint8, (1, 1, 1), None, None, 'backward', 1),
            (3, 2, np.uint8, (1, 1, 1), None, 'trnsf', 'backward', 1),
            (3, 2, np.uint8, (1, 1, 1), 'dataset_registered.zarr', None, 'backward', 1),
            (3, 2, np.uint8, (1, 1, 1), 'dataset_registered.zarr', 'trnsf', 'backward', 1),
        ]:
            with self.subTest(dim=dim, channels=channels, dtype=dtype, stride=stride, out_trnsf=out_trnsf, out_dataset=out_dataset, direction=direction, stepping=stepping):
                # Fit_apply
                shutil.rmtree("dataset_registered.zarr", ignore_errors=True)
                shutil.rmtree("trnsf", ignore_errors=True)
                dataset = sphere(
                    num_channels=channels,
                    num_spatial_dims=dim,
                    min_radius=10,
                    max_radius=10,
                    dtype=dtype,
                    stride=stride,
                    jump=4,
                )
                
                model = rt_reg.RegistrationVT(
                    registration_type="rigid3D",
                )
                dataset_registered = model.fit_apply(
                    dataset,
                    out_trnsf=out_trnsf,
                    out_dataset=out_dataset,
                    use_channel=1,
                    direction=direction,
                    perform_global_trnsf=True,
                    stepping=stepping,
                    verbose=False,
                )
                if direction == "backward":
                    pos = 0
                    iter = range(1,dataset.shape[0])
                else:
                    pos = 9
                    iter = range(dataset.shape[0]-2,-1,-1)
                # viewer = napari.Viewer()
                # viewer.add_image(dataset[pos], scale=dataset.attrs["scale"], colormap="red")
                # rt_vis.add_image(viewer, dataset, opacity=0.5)
                # rt_vis.add_image(viewer, dataset_registered, opacity=0.5, colormap="green")
                # viewer.dims.ndisplay = dim
                # viewer.dims.current_step = (0,0,0)
                # napari.run()
                # Check dtype
                self.assertEqual(dataset_registered.dtype, np.dtype(dtype), 
                                f"Expected dtype {dtype}, but got {dataset_registered.dtype}")
                # Check dimensions
                self.assertEqual(dataset.shape, dataset_registered.shape, 
                                f"Expected {dataset.shape} dimensions (including channels), but got {dataset_registered.shape}")
                # Check argmax position for all times
                for t in iter:
                    if channels == 1:
                        shape = dataset[1].shape
                        scale = np.array(dataset.attrs["scale"])[:dim]
                        # print(t, scale, shape)
                        # print(((np.array(np.unravel_index(dataset[pos].argmax(), shape)), np.array(np.unravel_index(dataset[t].argmax(), shape)))*scale))
                        # print(((np.array(np.unravel_index(dataset_registered[pos].argmax(), shape)), np.array(np.unravel_index(dataset_registered[t].argmax(), shape)))*scale))
                        # print()
                        self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos].argmax(), shape)) - np.array(np.unravel_index(dataset[t].argmax(), shape)))*scale) >
                                        np.linalg.norm((np.array(np.unravel_index(dataset[pos].argmax(), shape)) - np.array(np.unravel_index(dataset_registered[t].argmax(), shape)))*scale),
                                        f"Expected center of mass to be close to the same position.")
                    else:
                        shape = dataset[t,0].shape
                        scale = np.array(dataset.attrs["scale"])[:dim]
                        self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset[t,0].argmax(), shape)))*scale) >
                                        np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset_registered[t,0].argmax(), shape)))*scale),
                                        f"Expected center of mass to be close to the same position.")
                        self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset[t,1].argmax(), shape)))*scale) >
                                        np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset_registered[t,1].argmax(), shape)))*scale),
                                        f"Expected center of mass to be close to the same position.")
                        
                # Fit + apply
                shutil.rmtree("dataset_registered.zarr", ignore_errors=True)
                shutil.rmtree("trnsf", ignore_errors=True)
                model = rt_reg.RegistrationVT(
                    registration_type="rigid3D",
                )
                model.fit(
                    dataset,
                    out=out_trnsf,
                    use_channel=1,
                    direction=direction,
                    perform_global_trnsf=True,
                    stepping=stepping,
                    verbose=False,
                )
                dataset_registered = model.apply(
                    dataset,
                    out=out_dataset
                )
                if direction == "backward":
                    pos = 0
                    iter = range(1,dataset.shape[0])
                else:
                    pos = 9
                    iter = range(dataset.shape[0]-2,-1,-1)
                # viewer = napari.Viewer()
                # viewer.add_image(dataset[pos], scale=dataset.attrs["scale"], colormap="red")
                # rt_vis.add_image(viewer, dataset, opacity=0.5)
                # rt_vis.add_image(viewer, dataset_registered, opacity=0.5, colormap="green")
                # viewer.dims.ndisplay = dim
                # viewer.dims.current_step = (0,0,0)
                # napari.run()
                # Check dtype
                self.assertEqual(dataset_registered.dtype, np.dtype(dtype), 
                                f"Expected dtype {dtype}, but got {dataset_registered.dtype}")
                # Check dimensions
                self.assertEqual(dataset.shape, dataset_registered.shape, 
                                f"Expected {dataset.shape} dimensions (including channels), but got {dataset_registered.shape}")
                # Check argmax position for all times
                for t in iter:
                    if channels == 1:
                        shape = dataset[1].shape
                        scale = np.array(dataset.attrs["scale"])[:dim]
                        # print(t, scale, shape)
                        # print(((np.array(np.unravel_index(dataset[pos].argmax(), shape)), np.array(np.unravel_index(dataset[t].argmax(), shape)))*scale))
                        # print(((np.array(np.unravel_index(dataset_registered[pos].argmax(), shape)), np.array(np.unravel_index(dataset_registered[t].argmax(), shape)))*scale))
                        # print()
                        self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos].argmax(), shape)) - np.array(np.unravel_index(dataset[t].argmax(), shape)))*scale) >
                                        np.linalg.norm((np.array(np.unravel_index(dataset[pos].argmax(), shape)) - np.array(np.unravel_index(dataset_registered[t].argmax(), shape)))*scale),
                                        f"Expected center of mass to be close to the same position.")
                    else:
                        shape = dataset[t,0].shape
                        scale = np.array(dataset.attrs["scale"])[:dim]
                        self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset[t,0].argmax(), shape)))*scale) >
                                        np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset_registered[t,0].argmax(), shape)))*scale),
                                        f"Expected center of mass to be close to the same position.")
                        self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset[t,1].argmax(), shape)))*scale) >
                                        np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset_registered[t,1].argmax(), shape)))*scale),
                                        f"Expected center of mass to be close to the same position.")

                # Apply to downsampled dataset
                shutil.rmtree("dataset_registered.zarr", ignore_errors=True)
                dataset_downsampled = rt_utils.downsample(dataset, [0.5 for _ in range(dim)], n_workers=1, n_workers_processing=1)
                dataset_registered = model.apply(
                    dataset_downsampled,
                    out=out_dataset
                )
                if direction == "backward":
                    pos = 0
                    iter = range(1,dataset.shape[0])
                else:
                    pos = 9
                    iter = range(dataset.shape[0]-2,-1,-1)

                # viewer = napari.Viewer()
                # viewer.add_image(dataset[pos], scale=dataset.attrs["scale"], colormap="red", name=f"dataset t={pos}")
                # rt_vis.add_image(viewer, dataset, opacity=0.5, name="dataset")
                # rt_vis.add_image(viewer, dataset_downsampled, opacity=0.5, colormap="blue", name="dataset_downsampled")
                # rt_vis.add_image(viewer, dataset_registered, opacity=0.5, colormap="green", name="dataset_downsampled_registered")
                # viewer.dims.ndisplay = dim
                # viewer.dims.current_step = (0,0,0)
                # napari.run()

                # Check dtype
                self.assertEqual(dataset_registered.dtype, np.dtype(dtype), 
                                f"Expected dtype {dtype}, but got {dataset_registered.dtype}")
                # Check dimensions
                self.assertEqual(dataset_downsampled.shape, dataset_registered.shape, 
                                f"Expected {dataset_downsampled.shape} dimensions (including channels), but got {dataset_registered.shape}")
                # Check argmax position for all times
                for t in iter:
                    if channels == 1:
                        shape = dataset[0].shape
                        shape_downsampled = dataset_downsampled[0].shape
                        scale = np.array(dataset.attrs["scale"])[:dim]
                        scale_downsample = np.array(dataset_downsampled.attrs["scale"])[:dim]
                        # print(t, scale, pos, shape, np.unravel_index(dataset_downsampled[pos].argmax(), shape), shape_downsampled, np.unravel_index(dataset_downsampled[pos].argmax(), shape_downsampled))
                        # print(((np.array(np.unravel_index(dataset_downsampled[pos].argmax(), shape_downsampled)), np.array(np.unravel_index(dataset_downsampled[t].argmax(), shape_downsampled)))*scale))
                        # print(((np.array(np.unravel_index(dataset_registered[pos].argmax(), shape_downsampled)), np.array(np.unravel_index(dataset_registered[t].argmax(), shape_downsampled)))*scale))
                        # print()
                        self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos].argmax(), shape)) - np.array(np.unravel_index(dataset[t].argmax(), shape)))*scale) >
                                        np.linalg.norm((np.array(np.unravel_index(dataset_downsampled[pos].argmax(), shape_downsampled)) - np.array(np.unravel_index(dataset_registered[t].argmax(), shape_downsampled)))*scale_downsample),
                                        f"Expected center of mass to be close to the same position.")
                    else:
                        shape = dataset_downsampled[t,0].shape
                        shape_downsampled = dataset_downsampled[t,0].shape
                        scale = np.array(dataset_downsampled.attrs["scale"])[:dim]
                        scale_downsample = np.array(dataset_downsampled.attrs["scale"])[:dim]
                        self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset[t,0].argmax(), shape)))*scale) >
                                        np.linalg.norm((np.array(np.unravel_index(dataset_downsampled[pos,0].argmax(), shape_downsampled)) - np.array(np.unravel_index(dataset_registered[t,0].argmax(), shape_downsampled)))*scale_downsample),
                                        f"Expected center of mass to be close to the same position.")
                        self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset[t,1].argmax(), shape)))*scale) >
                                        np.linalg.norm((np.array(np.unravel_index(dataset_downsampled[pos,0].argmax(), shape_downsampled)) - np.array(np.unravel_index(dataset_registered[t,1].argmax(), shape_downsampled)))*scale_downsample),
                                        f"Expected center of mass to be close to the same position.")

                # Apply to downsampled dataset
                shutil.rmtree("dataset_registered.zarr", ignore_errors=True)
                dataset_registered = model.apply(
                    dataset,
                    out=out_dataset,
                    padding=[[50,50],[50,50],[50,50]][:dim]
                )
                if direction == "backward":
                    pos = 0
                    iter = range(1,dataset.shape[0])
                else:
                    pos = 9
                    iter = range(dataset.shape[0]-2,-1,-1)

                # viewer = napari.Viewer()
                # viewer.add_image(dataset[pos], scale=dataset.attrs["scale"], colormap="red", name=f"dataset t={pos}")
                # rt_vis.add_image(viewer, dataset, opacity=0.5, name="dataset")
                # rt_vis.add_image(viewer, dataset_registered, opacity=0.5, colormap="green", name="dataset_registered")
                # viewer.dims.ndisplay = dim
                # viewer.dims.current_step = (0,0,0)
                # napari.run()

                # Check dtype
                self.assertEqual(dataset_registered.dtype, np.dtype(dtype), 
                                f"Expected dtype {dtype}, but got {dataset_registered.dtype}")
                # Check dimensions
                new_shape = np.array(dataset.shape); new_shape[-dim:] += 100; new_shape = tuple(new_shape)
                self.assertEqual(new_shape, dataset_registered.shape, 
                                f"Expected {new_shape} dimensions (including channels), but got {dataset_registered.shape}")
                # Check argmax position for all times
                for t in iter:
                    if channels == 1:
                        shape = dataset[0].shape
                        shape_padding = dataset_registered[0].shape
                        scale = np.array(dataset.attrs["scale"])[:dim]
                        # print(t, scale, pos, shape, np.unravel_index(dataset[pos].argmax(), shape), shape_padding, np.unravel_index(dataset_registered[pos].argmax(), shape_padding))
                        # print((np.array(np.unravel_index(dataset[pos].argmax(), shape))*scale, np.array(np.unravel_index(dataset[t].argmax(), shape))*scale))
                        # print((np.array(np.unravel_index(dataset_registered[pos].argmax(), shape_padding))*scale, np.array(np.unravel_index(dataset_registered[t].argmax(), shape_padding))*scale))
                        # print()
                        self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos].argmax(), shape)) - np.array(np.unravel_index(dataset[t].argmax(), shape)))*scale) >
                                        np.linalg.norm((np.array(np.unravel_index(dataset_registered[pos].argmax(), shape_padding)) - np.array(np.unravel_index(dataset_registered[t].argmax(), shape_padding)))*scale),
                                        f"Expected center of mass to be close to the same position.")
                    else:
                        shape = dataset_downsampled[t,0].shape
                        shape_padding = dataset_downsampled[t,0].shape
                        scale = np.array(dataset_downsampled.attrs["scale"])[:dim]
                        self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset[t,0].argmax(), shape)))*scale) >
                                        np.linalg.norm((np.array(np.unravel_index(dataset_registered[pos,0].argmax(), shape_padding)) - np.array(np.unravel_index(dataset_registered[t,0].argmax(), shape_padding)))*scale),
                                        f"Expected center of mass to be close to the same position.")
                        self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset[t,1].argmax(), shape)))*scale) >
                                        np.linalg.norm((np.array(np.unravel_index(dataset_registered[pos,0].argmax(), shape_padding)) - np.array(np.unravel_index(dataset_registered[t,1].argmax(), shape_padding)))*scale),
                                        f"Expected center of mass to be close to the same position.")

if __name__ == "__main__":
    unittest.main()