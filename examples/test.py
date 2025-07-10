
# %%
import os
import napari
import registration_tools as rt #Main package
import registration_tools.data as rt_data #For generating artificial datasets
import registration_tools.visualization as rt_vis #For visualization
import registration_tools.registration as rt_reg #For registration
import zarr
import tempfile
import shutil
import vt
import numpy as np

# %%
# Create a dataset of spherical images
shutil.rmtree("spheres.zarr", ignore_errors=True)
file_spheres = "spheres.zarr"
dataset = rt_data.sphere(
    out = file_spheres, #If not specified, a new dataset is created and stored in RAM
    num_images=10,
    image_size=150,
    num_channels=3,
    min_radius=5,
    max_radius=5,
    jump=3,
    stride=(1, 1, 1)
)
print("Type: ", type(dataset))
print("Shape: ", dataset.attrs["axis"])
print("Scale: ", dataset.attrs["scale"])
print("Shape: ", dataset.shape)

# %%
# Register the video
params = "-parallelism-type thread"#"-pyramid-gaussian-filtering"

dataset_box = zarr.ones(dataset.shape, dtype='uint8')
dataset_box[:] *= 255
dataset_box.attrs["scale"] = dataset.attrs["scale"]
dataset_box.attrs["axis"] = dataset.attrs["axis"]

with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_file = "./out"
    temp_data = "tmp.zarr"
    if os.path.exists(temp_file):
        shutil.rmtree(temp_file)
    shutil.rmtree("tmp.zarr", ignore_errors=True)
    registration = rt_reg.RegistrationVT()

    registration.fit_manual(
        dataset,
        out=temp_file,
    )