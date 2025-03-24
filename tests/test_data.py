import unittest
import numpy as np
from registration_tools.data import sphere
from registration_tools.visualization import add_image
import os
import shutil

class TestExampleData(unittest.TestCase):
    def test_sphere_generation(self):
        for t in [10,20]:
            for dim in [2, 3]:
                for channels in [1, 2]:
                    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64]:
                        for stride in [(1, 1, 1), (3, 2, 1)]:
                            for out in [None, "dataset.zarr"]:
                                with self.subTest(t=t, dim=dim, channels=channels, dtype=dtype, stride=stride, out=out):
                                    shutil.rmtree("dataset.zarr", ignore_errors=True)
                                    dataset = sphere(
                                        num_images=t,
                                        num_channels=channels,
                                        num_spatial_dims=dim,
                                        min_radius=5,
                                        max_radius=5,
                                        dtype=dtype,
                                        out=out,
                                        stride=stride
                                    )

                                    # Check dtype
                                    self.assertEqual(dataset.dtype, np.dtype(dtype), 
                                                    f"Expected dtype {dtype}, but got {dataset.dtype}")

                                    # Check scale
                                    self.assertEqual(dataset.attrs["scale"], [stride[i] for i in range(dim)], 
                                                    f"Expected scale {[stride[i] for i in range(dim)]}, but got {dataset.attrs['scale']}")

                                    # Check dimensions
                                    self.assertEqual(dataset.ndim, dim + 1 + channels - 1, 
                                                    f"Expected {dim + 1 + channels - 1} dimensions (including channels), but got {dataset.ndim}")

                                    # Check channels
                                    self.assertEqual(dataset.shape[0], t, 
                                                    f"Expected {t} channels, but got {dataset.shape[0]}")
                                    if channels > 1:
                                        self.assertEqual(dataset.shape[1], channels, 
                                                        f"Expected {channels} channels, but got {dataset.shape[1]}")
                                    if dim == 2:
                                        self.assertEqual(dataset.shape[-2], 100//stride[0], 
                                                        f"Expected {100//stride[0]} in x, but got {dataset.shape[-2]}")
                                        self.assertEqual(dataset.shape[-1], 100//stride[1],
                                                        f"Expected {100//stride[1]} in y, but got {dataset.shape[-1]}")
                                    elif dim == 3:
                                        self.assertEqual(dataset.shape[-3], 100//stride[0], 
                                                        f"Expected {100//stride[0]} in x, but got {dataset.shape[-3]}")
                                        self.assertEqual(dataset.shape[-2], 100//stride[1],
                                                        f"Expected {100//stride[1]} in y, but got {dataset.shape[-2]}")
                                        self.assertEqual(dataset.shape[-1], 100//stride[2],
                                                        f"Expected {100//stride[2]} in z, but got {dataset.shape[-1]}")
                                

if __name__ == "__main__":
    unittest.main()