import registration_tools.data as rt_data
import registration_tools.visualization as rt_vis
import napari

for dim in [2,3]:
    for channels in [1,2]:
        for dtype in ["uint8", "uint16", "uint32", "uint64", "float32", "float64"]:
            for out in [None, "dataset.zarr", "dataset"]:
                for stride in [(1,1,1), (3,2,1)]:
                    print(f"Generating dataset with {channels} channels, {dim} spatial dimensions and dtype {dtype}")
                    dataset = rt_data.sphere(
                        num_channels=channels,
                        num_spatial_dims=dim,
                        min_radius=5,
                        max_radius=5,
                        dtype=dtype,
                    )

                    dataset.dtype == dtype

                    viewer = napari.Viewer()
                    rt_vis.add_image(viewer, dataset)
                    viewer.dims.ndisplay = dim
                    viewer.dims.current_step = (0,0,0)
                    napari.run()
