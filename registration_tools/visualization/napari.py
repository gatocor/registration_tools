import os
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
import napari
import imageio
import zarr
import dask.array as da

from skimage.io import imread, imsave
from ..utils.auxiliar import _make_index

def add_image(viewer, dataset, split_channel=None, scale=None, **kwargs):
    """
    Plots the images in the dataset in the viewer.

    Args:
        viewer (napari.Viewer): The napari viewer instance.
        dataset (Dataset): The dataset object.
        channels (list, optional): List of channels to plot. If None, all channels are plotted.
        numbers (list, optional): List of numbers to plot. If None, all numbers are plotted.
        scale (tuple, optional): Scale for the images. Default is (1, 1, 1).
        downsample (tuple, optional): downsample factor for the images. Default is (1, 1, 1).
        verbosity (int, optional): Verbosity level. Default is 1.
    """

    if isinstance(dataset, zarr.Array):
        axis = dataset.attrs["axis"]
        scale = dataset.attrs["scale"]

    if split_channel is not None:
        for i in range(2):
            viewer.add_image(da.from_zarr(dataset)[i,:,:,:,:])
    else:
        viewer.add_image(dataset, scale=scale, **kwargs)

def add_image_difference(viewer, dataset, dt=1, cmap1="red", cmap2="green", opacity1=1, opacity2=0.5, scale=None, **kwargs):

    if isinstance(dataset, zarr.Array):
        axis = dataset.attrs["axis"]
        scale = dataset.attrs["scale"]

        # Convert Zarr dataset to Dask array to avoid memory loading
        dask_dataset = da.from_zarr(dataset)

    else:

        dask_dataset = dataset

    # Lazy slice without loading data into memory
    img1 = dask_dataset[:-dt]
    img2 = dask_dataset[dt:]

    # Add images to Napari without forcing computation
    viewer.add_image(img1, scale=scale[::-1], colormap=cmap1, opacity=opacity1, **kwargs)
    viewer.add_image(img2, scale=scale[::-1], colormap=cmap2, opacity=opacity2, **kwargs)

def add_vectors(viewer, model, time_axis, scale=None, downsample=(1,1,1), **kwargs):
    """
    Plots the vectors in the model in the viewer.

    Args:
        viewer (napari.Viewer): The napari viewer instance.
        model (Registration): The registration object.
        scale (tuple, optional): Scale for the vectors. Default is (1, 1, 1).
        downsample (tuple, optional): downsample factor for the vectors. Default is (1, 1, 1).
        verbosity (int, optional): Verbosity level. Default is 1.
    """

    viewer.add_vectors([], scale=scale, name="vectorfield", **kwargs)
    _update_vectors(viewer, model, time_axis, downsample, None)
    # viewer.dims.events.current_step.connect(_update_vectors)
    viewer.dims.events.current_step.connect(lambda event: _update_vectors(viewer, model, time_axis, event))

def _update_vectors(viewer, model, axis, downsample, event):

    if "vectorfield" in viewer.layers:
        vectors = viewer.layers["vectorfield"]
    else:
        return
            
    t = viewer.dims.current_step[axis]

    for i in range(model._t_max):
        if model._trnsf_exists_relative(t, i):
            trnsf = model._load_transformation_relative(t, i)
            mask = trnsf[:,0,0] % downsample[0] == 0
            trnsf = trnsf[mask,:,:]
            mask = trnsf[:,0,1] % downsample[1] == 0
            trnsf = trnsf[mask,:,:]
            if model._n_spatial == 3:
                mask = trnsf[:,0,2] % downsample[2] == 0
                trnsf = trnsf[mask,:,:]
            vectors.data = trnsf
            break

def make_video(viewer, save_file, time_channel=0, fps=10, zooms=None, angles=None, canvas_only=True):
    """
    Creates a video from the napari viewer by taking screenshots of all time points.

    Args:
        viewer (napari.Viewer): The napari viewer instance.
        save_file (str): The path to save the video. Must have a .gif extension.
        time_channel (int, optional): The dimension index for time. Default is 0.
        fps (int, optional): Frames per second for the video. Default is 10.
        zooms (list, optional): List of zoom levels for each time point. If None, keep fixed.
        angles (list, optional): List of camera angles for each time point. If None, keep fixed.
        canvas_only (bool, optional): Whether to capture only the canvas or the entire viewer. Default is True.

    Returns:
        None
    """

    if not save_file.lower().endswith('.gif'):
        raise ValueError("The save_file must have a .gif extension")

    screenshots = []
    num_timepoints = int(viewer.dims.range[time_channel][1])

    for t in tqdm(range(num_timepoints), desc="Creating video: ", unit="image", total=num_timepoints-1):
        viewer.dims.set_point(time_channel, t)
        
        if zooms is not None and t < len(zooms):
            viewer.camera.zoom = zooms[t]
        
        if angles is not None and t < len(angles):
            viewer.camera.angles = angles[t]
        
        screenshot = viewer.screenshot(canvas_only=canvas_only)
        screenshots.append(screenshot)

    imageio.mimsave(save_file, screenshots, fps=fps)