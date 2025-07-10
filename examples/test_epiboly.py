import zarr
import numpy as np
import registration_tools as rt #Main package
import registration_tools.data as rt_data #For generating artificial datasets
import registration_tools.visualization as rt_vis #For visualization
import registration_tools.registration as rt_reg #For registration
import napari
import matplotlib.pyplot as plt
from copy import deepcopy

dataset = zarr.open_array("dataset_0_subsampling.zarr")
scale=dataset.attrs['scale']
axis=dataset.attrs['axis']
print(f"Shape {dataset.shape}")
print(f"Scale {dataset.attrs['scale']}")
print(f"Dtype {dataset.dtype}")
dataset = zarr.open_array("dataset_0_subsampling.zarr")[:,:,:,:,:]

# plt.hist(dataset[1,0].astype(np.uint16).flatten(), bins=100)
# plt.hist(dataset[1,1].astype(np.uint16).flatten(), bins=100)
# plt.show()

# model = rt_reg.RegistrationVT(
#     pyramid_lowest_level=0,
#     pyramid_highest_level=5,
#     registration_type="rotation",
#     args_registration="-pyramid-gaussian-filtering"# -normalisation -estimator-type lts"
# )
# model = rt_reg.RegistrationMoments(
#     1,
#     align_rotation=True,
#     align_center=False,
# )

dataset_registered = dataset.copy()
d = []
models = [
    rt_reg.RegistrationManual()
    # rt_reg.RegistrationMoments(
    #     1,
    #     align_rotation=True,
    #     align_center=False,
    # ),
    # rt_reg.RegistrationVT(
    #     pyramid_lowest_level=0,
    #     pyramid_highest_level=5,
    #     registration_type="rotation",
    #     args_registration="-pyramid-gaussian-filtering -normalisation"
    # ),
    # rt_reg.RegistrationVT(
    #     pyramid_lowest_level=0,
    #     pyramid_highest_level=5,
    #     registration_type="affine",
    #     args_registration="-pyramid-gaussian-filtering -normalisation -reference-high-threshold 0.5 -floating-high-threshold 0.5"
    # ),
]
for model in models:

    model.fit(
        dataset_registered,
        axis=axis,
        scale=scale,
        use_channel=1,
        perform_global_trnsf=True,
        stepping=1,
        verbose=False,
    )

    dataset_registered = model.apply(
        dataset_registered,
        axis=axis,
        scale=scale,
    )

    d.append(deepcopy(dataset_registered))

viewer = napari.Viewer()
viewer.add_image(dataset[1,1,:,:,:], scale=scale)
viewer.add_image(dataset, scale=scale)
colormap=["blue","red","green"]
for i,c in enumerate(d):
    rt_vis.add_image(viewer, d[i], opacity=0.6, colormap=colormap[i], contrast_limits=(0,500))
viewer.dims.ndisplay = 3
viewer.dims.current_step = (0, 0, 0)
napari.run()