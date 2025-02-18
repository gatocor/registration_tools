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

def add_image(viewer, dataset, scale=None, **kwargs):
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

def make_video(viewer, save_file, fps=10, zooms=None, angles=None, canvas_only=True):
    """
    Creates a video from the napari viewer by taking screenshots of all time points.

    Args:
        viewer (napari.Viewer): The napari viewer instance.
        save_path (str): The path to save the video.
        fps (int, optional): Frames per second for the video. Default is 10.
        zooms (list, optional): List of zoom levels for each time point. If None, keep fixed.
        angles (list, optional): List of camera angles for each time point. If None, keep fixed.
    """
    screenshots = []
    num_timepoints = int(viewer.dims.range[0][1])

    for t in range(num_timepoints):
        viewer.dims.set_point(0, t)
        
        if zooms is not None and t < len(zooms):
            viewer.camera.zoom = zooms[t]
        
        if angles is not None and t < len(angles):
            viewer.camera.angles = angles[t]
        
        screenshot = viewer.screenshot(canvas_only=canvas_only)
        screenshots.append(screenshot)

    imageio.mimsave(save_file, screenshots, fps=fps)