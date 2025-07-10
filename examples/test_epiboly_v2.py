import zarr
import numpy as np
import registration_tools as rt #Main package
import registration_tools.data as rt_data #For generating artificial datasets
import registration_tools.visualization as rt_vis #For visualization
import registration_tools.registration as rt_reg #For registration
import registration_tools.utils as rt_utils #For utilities
import napari
import matplotlib.pyplot as plt
from copy import deepcopy
from skimage.filters import difference_of_gaussians
import shutil

dataset = zarr.open_array("dataset_0_subsampling.zarr")
scale=dataset.attrs['scale']
axis=dataset.attrs['axis']
print(f"Shape {dataset.shape}")
print(f"Scale {dataset.attrs['scale']}")
print(f"Dtype {dataset.dtype}")
dataset = zarr.open_array("dataset_0_subsampling.zarr")[:,:,:,:,:]

# dataset = rt_data.sphere(max_radius=5)

model = rt_reg.RegistrationManual()

# model.fit(
#     dataset,
#     out="registration_subsampling_manual_0",
#     axis=axis,
#     scale=scale,
#     use_channel=1,
#     direction="backward",
#     stepping=1,
#     verbose=False,
# )

model.load(
    "registration_subsampling_manual_0"
)

# for i in range(1, dataset.shape[0]):
#     trnsf = model._load_transformation_global(i, 0)
#     model._save_transformation_global(np.linalg.inv(trnsf), i, 0)

# dataset = rt_utils.downsample(dataset, (.5,.5,.5))

# dataset_registered = model.apply(
#     dataset,
#     out="dataset_subsampling_manual_0.zarr",
#     axis=axis,
#     scale=scale,
# )

dataset_registered = zarr.open_array("dataset_subsampling_manual_0.zarr")

model = rt_reg.RegistrationVT(
    registration_type="vectorfield",
)

# shutil.rmtree("registration_subsampling_vectorfield_0", ignore_errors=True)
# model.fit(
#     dataset_registered,
#     use_channel=1,
#     axis=axis,
#     scale=scale,
#     out="registration_subsampling_vectorfield_0",
#     direction="forward",
#     verbose=False,
# )

model.load("registration_subsampling_vectorfield_0")

# def f(x):
#     y = x[1]
#     return difference_of_gaussians(y, 1, 2) #> 0.5

# shutil.rmtree("dataset_0_subsampling_dogs.zarr", ignore_errors=True)
# dataset_thresholded = rt_utils.apply_function_in_time_dask(
#     dataset_registered,
#     f,
#     new_axis="TZYX",
#     new_scale=scale,
#     out="dataset_0_subsampling_dogs.zarr",
# )

# dataset_dogs = zarr.open_array("dataset_0_subsampling_dogs.zarr")

def f(x):
    y = x[1]
    return y > 500

# shutil.rmtree("dataset_0_subsampling_thresholded.zarr", ignore_errors=True)
# dataset_thresholded = rt_utils.apply_function_in_time_dask(
#     dataset_registered,
#     f,
#     axis="CTZYX",
#     scale=scale,
#     new_axis="TZYX",
#     new_scale=scale,
#     out="dataset_0_subsampling_thresholded.zarr",
# )

dataset_thresholded = zarr.open_array("dataset_0_subsampling_thresholded.zarr")

# print(dataset_registered)

threshold = 1000
viewer = napari.Viewer()
# viewer.add_image(dataset[1,1,:,:,:], scale=scale)
# viewer.add_image(dataset, scale=scale, contrast_limits=(0,threshold))
# viewer.dims.ndisplay = 3
# viewer.dims.current_step = (0,0,0)
# viewer.dims.set_point(0, 1)
# rt_vis.make_video(viewer, "Unregistered.gif", time_channel=1)
# viewer.close()

# viewer = napari.Viewer()
# rt_vis.add_image(viewer, dataset_registered, opacity=0.6, colormap="green")
viewer.add_image(dataset_registered, scale=scale, opacity=0.6, colormap="green")
rt_vis.add_vectors(
    viewer,
    model,
    1,
    mask=dataset_thresholded,
    vector_style="arrow",
    edge_color="red",
    edge_width=0.41,
    length=2.,
    downsample=(100,100,100),
    scale=scale,
)
# rt_vis.add_image(viewer, dataset_dogs, opacity=0.6, colormap="red")
viewer.dims.ndisplay = 3
# viewer.dims.current_step = (0,0,0)
viewer.dims.set_point(0, 1)
# rt_vis.make_video(viewer, "Registered.gif", time_channel=1)
# viewer.close()

napari.run()