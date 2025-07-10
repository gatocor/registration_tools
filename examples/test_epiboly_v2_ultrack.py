import zarr
import numpy as np
import registration_tools as rt #Main package
import registration_tools.data as rt_data #For generating artificial datasets
import registration_tools.visualization as rt_vis #For visualization
import registration_tools.registration as rt_reg #For registration
import registration_tools.utils as rt_utils #For utilities
import registration_tools.tracking as rt_track
import napari
import matplotlib.pyplot as plt
from copy import deepcopy
from skimage.filters import difference_of_gaussians
import shutil
from ultrack.imgproc import robust_invert, detect_foreground

# from ultrack import MainConfig, load_config, track, to_tracks_layer, tracks_to_zarr
# from ultrack.imgproc import robust_invert, detect_foreground
# from ultrack.utils.array import array_apply, create_zarr

dataset = zarr.open_array("dataset_0_subsampling.zarr")
scale=dataset.attrs['scale']
axis=dataset.attrs['axis']
print(f"Shape {dataset.shape}")
print(f"Scale {dataset.attrs['scale']}")
print(f"Dtype {dataset.dtype}")
dataset = zarr.open_array("dataset_0_subsampling.zarr")#[:,:,:,:,:]
dataset_registered = zarr.open_array("dataset_subsampling_manual_0.zarr")

model = rt_track.TrackingUltrack()

# dataset_monochanel = rt_utils.apply_function_in_time_dask(
#     dataset,
#     lambda x: x[1],
#     new_axis="TZYX"
# )

def f(x):
    y = x[1]
    return difference_of_gaussians(y, 1, 2) #> 500

shutil.rmtree("dataset_0_subsampling_dogs.zarr", ignore_errors=True)
dataset_thresholded = rt_utils.apply_function_in_time_dask(
    dataset_registered,
    f,
    new_axis="TZYX",
    new_scale=scale,
    out="dataset_0_subsampling_dogs.zarr",
)

# dataset_background = rt_utils.apply_function_in_time_dask(
#     dataset_monochanel,
#     detect_foreground,
#     sigma=25
# )

# dataset_borders = rt_utils.apply_function_in_time_dask(
#     dataset_monochanel,
#     robust_invert,
# )

# model.fit(
#     dataset_background,
#     dataset_borders,
# )

viewer = napari.Viewer()
rt_vis.add_image(viewer, dataset_registered)
rt_vis.add_image(viewer, dataset_thresholded)
# rt_vis.add_image(viewer, dataset_background)
# viewer.add_image(
#     dataset_borders,
#     # visible=True,
#     # translate=(start_idx, 0, 0, 0),
#     scale=dataset_registered.attrs['scale'],
# ).contour = 2
# print(model._tracks_df[["track_id","t", "z", "y", "x"]].values)
# viewer.add_tracks(model._tracks_df[["track_id","t", "z", "y", "x"]].values, name="tracks", scale=dataset.attrs['scale'])
napari.run()
