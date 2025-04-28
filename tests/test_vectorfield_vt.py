import unittest
import numpy as np
from registration_tools.data import sphere
import registration_tools.registration as rt_reg
import os
import shutil
import napari
import registration_tools.visualization as rt_vis
import registration_tools.utils as rt_utils
import pandas as pd

class TestExampleData(unittest.TestCase):

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("dataset_registered.zarr", ignore_errors=True)
        shutil.rmtree("trnsf", ignore_errors=True)

    def test_registration(self):
        for dim,channels,dtype,stride,out_dataset,out_trnsf,direction,stepping in [
            (2, 1, np.uint8, (1, 1, 1), None, None, 'forward', 1),
            # (3, 1, np.uint8, (1, 1, 1), None, None, 'backward', 1),
        ]:
            with self.subTest(dim=dim, channels=channels, dtype=dtype, stride=stride, out_trnsf=out_trnsf, out_dataset=out_dataset, direction=direction, stepping=stepping):
                # Fit_apply
                shutil.rmtree("dataset_registered.zarr", ignore_errors=True)
                shutil.rmtree("trnsf", ignore_errors=True)
                dataset = sphere(
                    num_channels=channels,
                    num_spatial_dims=dim,
                    num_images=2,
                    min_radius=10,
                    max_radius=10,
                    dtype=dtype,
                    stride=stride,
                    jump=20
                )

                model = rt_reg.RegistrationVT(
                    registration_type="vectorfield",
                    pyramid_highest_level=3,
                    pyramid_lowest_level=0,
                )
                img = model.fit_apply(
                    dataset,
                    direction=direction,
                    verbose=False
                )

                # for i in range(10):
                #     model._load_transformation_global()
                # dataset_registered = model.apply(dataset)
                def thresholding(x,id):
                    return x > 0.5
                
                mask = rt_utils.apply_function(
                    dataset,
                    thresholding,
                )
                mask = None
                vectors = model.vectorfield(mask, n_points=20)
                # tracks = model.trajectories(mask, n_points=20)

                if direction == "backward":
                    pos = 0
                    iter = range(1,dataset.shape[0])
                else:
                    pos = 9
                    iter = range(dataset.shape[0]-2,-1,-1)
                pos=0

                viewer = napari.Viewer()
                viewer.add_image(dataset[pos], scale=dataset.attrs["scale"], colormap="red")
                rt_vis.add_image(viewer, dataset, opacity=0.5)
                rt_vis.add_vectors(viewer, vectors)
                # viewer.add_tracks(tracks, name="tracks", opacity=0.5)
                viewer.add_image(img[0], opacity=0.5)
                viewer.dims.ndisplay = dim
                viewer.dims.current_step = (pos,0,0)
                napari.run()

                # # Check dtype
                # self.assertEqual(dataset_registered.dtype, np.dtype(dtype), 
                #                 f"Expected dtype {dtype}, but got {dataset_registered.dtype}")
                # # Check dimensions
                # self.assertEqual(dataset.shape, dataset_registered.shape, 
                #                 f"Expected {dataset.shape} dimensions (including channels), but got {dataset_registered.shape}")
                # # Check argmax position for all times
                # for t in iter:
                #     if channels == 1:
                #         shape = dataset[1].shape
                #         scale = np.array(dataset.attrs["scale"])[:dim]
                #         # print(t, scale, shape)
                #         # print(((np.array(np.unravel_index(dataset[pos].argmax(), shape)), np.array(np.unravel_index(dataset[t].argmax(), shape)))*scale))
                #         # print(((np.array(np.unravel_index(dataset_registered[pos].argmax(), shape)), np.array(np.unravel_index(dataset_registered[t].argmax(), shape)))*scale))
                #         # print()
                #         self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos].argmax(), shape)) - np.array(np.unravel_index(dataset[t].argmax(), shape)))*scale) >
                #                         np.linalg.norm((np.array(np.unravel_index(dataset[pos].argmax(), shape)) - np.array(np.unravel_index(dataset_registered[t].argmax(), shape)))*scale),
                #                         f"Expected center of mass to be close to the same position.")
                #     else:
                #         shape = dataset[t,0].shape
                #         scale = np.array(dataset.attrs["scale"])[:dim]
                #         self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset[t,0].argmax(), shape)))*scale) >
                #                         np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset_registered[t,0].argmax(), shape)))*scale),
                #                         f"Expected center of mass to be close to the same position.")
                #         self.assertTrue(np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset[t,1].argmax(), shape)))*scale) >
                #                         np.linalg.norm((np.array(np.unravel_index(dataset[pos,0].argmax(), shape)) - np.array(np.unravel_index(dataset_registered[t,1].argmax(), shape)))*scale),
                #                         f"Expected center of mass to be close to the same position.")
                        
if __name__ == "__main__":
    unittest.main()