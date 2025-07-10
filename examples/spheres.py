
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
    num_images=150,
    image_size=200,
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
    registration = rt_reg.RegistrationVT(
        pyramid_highest_level=3,           #Higher pyramid level
        pyramid_lowest_level=0,            #Lower pyramid level
        registration_type='rigid',   #Type of registration
        args_registration=params
    )

    # dataset.attrs["scale"] = (1, 1, 1)

    # dataset_registered = registration.fit_apply(
    #     dataset,
    #     out_trnsf=temp_file,
    #     out_dataset=temp_data,
    #     use_channel=0,
    #     perfom_global_trnsf=True,
    #     downsample=(1, 1, 1)
    # )

    registration.fit(
        dataset,
        out=temp_file,
        use_channel=0,
        perfom_global_trnsf=True,
        downsample=(1, 1, 1)
    )
    dataset_registered = registration.apply(
        dataset,
        out=temp_data,
        padding=False,
        downsample=(1, 1, 1),
    )

    # registration.fit(
    #     dataset,
    #     use_channel=0,
    #     )
    
    # print(data[0,0,:,:,:].sum())
    # print(data[0,1,:,:,:].sum())
    # print(data[0,2,:,:,:].sum())
    # print(data[1,0,:,:,:].sum())
    # print(data[1,1,:,:,:].sum())
    # print(data[1,2,:,:,:].sum())

    # dataset_registered = registration.apply(
    #     dataset,
    #     padding=True,
    #     downsample=(1, 1, 1),
    # )

    # dataset_box_registered = registration.apply(
    #     dataset_box,
    #     padding=True,
    #     downsample=(1, 1, 1),
    # )

    viewer = napari.Viewer()
    # d = dataset_box_registered[:]
    # d[d==0] = 120
    # d[d==255] = 100
    # for i in range(10):
    #     print(np.mean(np.argwhere(dataset_box_registered[i,0] == 255), axis=0))
    rt_vis.add_image(viewer, dataset, name="Dataset")
    rt_vis.add_image(viewer, dataset_registered, name="Dataset Registered", colormap="red", opacity=0.5)
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