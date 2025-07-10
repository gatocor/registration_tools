
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
dataset = rt_data.hemisphere_spheres(
    out = None, #If not specified, a new dataset is created and stored in RAM
    num_images=150,
    image_size=200,
    num_channels=3,
    min_radius=5,
    max_radius=5,
    translation_speed=0,
    rotation_speed=10*0.1,
    stride=(1, 1, 1),
)
print(dataset)
print("Type: ", type(dataset))
print("Shape: ", dataset.attrs["axis"])
axis = dataset.attrs["axis"]
print("Scale: ", dataset.attrs["scale"])
scale = dataset.attrs["scale"]
print("Shape: ", dataset.shape)
dataset = zarr.array(dataset[:,:,:,20:-20,:])
# dataset2[:] = dataset[:]
dataset.attrs["axis"] = axis
dataset.attrs["scale"] = scale
# dataset = dataset2
# %%
# Register the video
params = ""#"-pyramid-gaussian-filtering"

# dataset_box = zarr.ones(dataset.shape, dtype='uint8')
# dataset_box[:] *= 255
# dataset_box.attrs["scale"] = dataset.attrs["scale"]
# dataset_box.attrs["axis"] = dataset.attrs["axis"]

with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_file = "./out"
    if os.path.exists(temp_file):
        shutil.rmtree(temp_file)
    shutil.rmtree("tmp.zarr", ignore_errors=True)
    registration = rt_reg.RegistrationMoments(
        n_axis=1,
        align_center=False,
        align_rotation=True,
    )

    dataset_registered = registration.fit_apply(
        dataset,
        use_channel=0,
        stepping=1,
        perfom_global_trnsf=True,
        downsample=(1, 2, 1),
        num_loading_threads=3,
        num_preloaded_images=3
    )

    # registration.fit(
    #     dataset,
    #     use_channel=0,
    #     stepping=3,
    #     perfom_global_trnsf=True,
    #     downsample=(1, 1, 1)
    # )

    # dataset_registered = registration.apply(
    #     dataset
    # )

    # dataset_registered = registration.apply(
    #     dataset,
    #     padding=False,
    #     downsample=(1, 1, 1),
    # )

    # dataset_box_registered = registration.apply(
    #     dataset_box,
    #     padding=False,
    #     downsample=(1, 1, 1),
    # )

    viewer = napari.Viewer()
    # d = dataset_box_registered[:]
    # # d[d==0] = 120
    # d[d==255] = 100
    # d[:] += 100*dataset_registered[:]
    # rt_vis.add_image(viewer, dataset_registered, name="Registered", colormap="red")
    rt_vis.add_image(viewer, dataset, name="Dataset")
    rt_vis.add_image(viewer, dataset_registered, name="Dataset Registered", opacity=0.5, colormap="red")
    viewer.dims.ndisplay = 3
    viewer.dims.current_step = (0, 0, 0)
    napari.run()

    # registration.load(temp_file.name)

    # registration.fit_apply(
    #     dataset,
    #     use_channel=0,
    #     downsample=(1, 2, 1),
    #     out="tmp.zarr"
    # )

    # with open(temp_file.name+"/parameters.json", 'r') as file:
    #     registration_data = file.read()
    #     print("Registration Data: ", registration_data)

# img = registration.apply(dataset, downsample=(1,2,2))
# print(img.shape)
# print(img.attrs["axis"])
# print(img.attrs["scale"])