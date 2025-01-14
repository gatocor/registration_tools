import os
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
import napari
import imageio

from skimage.io import imread, imsave

def plot_images(viewer, dataset, channels=None, numbers=None, downsample=(1, 1, 1), verbosity=1):
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

    cmaps = ['red', 'green', 'blue', 'magenta', 'yellow', 'cyan']

    for ch in range(dataset._nchannels):
        images = []
        for image in dataset.get_data_iterator(channel=ch, downsample=downsample):
            images.append(image[np.newaxis, ...])

        viewer.add_image(np.concatenate(images, axis=0), scale=(dataset._scale), opacity=0.5, blending='additive', colormap=cmaps[ch], name=f'Channel {ch}')

def plot_projections(viewer, dataset, projection, channels=None, old=False):
    """
    Plots the projections of the dataset in the viewer.

    Args:
        viewer (napari.Viewer): The napari viewer instance.
        dataset (Dataset): The dataset object.
        projection (int): The projection axis (e.g., 0 for X, 1 for Y, 2 for Z).
        channels (list, optional): List of channels to plot. If None, all channels are plotted.
        old (bool, optional): Whether to plot old projections. Default is False.
    """
    cmaps = ['red', 'green', 'blue', 'magenta', 'yellow', 'cyan']

    add_old = ""
    if old:
        add_old = "old_"

    if channels is None:
        channels = range(dataset.num_channels)

    scale = list(dataset._scale)
    del scale[projection]

    for pos, ch in enumerate(channels):
        path = os.path.join(dataset._save_folder, "projections", f"{add_old}projections_ch{ch}", f"joint_{add_old}projections_{projection}.tiff")
        if not os.path.exists(path):
            raise ValueError("Projection file does not exist.")

        image = imread(path)

        viewer.add_image(image, scale=scale, opacity=0.5, colormap=cmaps[pos], blending='additive')

def plot_projections_difference(viewer, dataset, projection, channel=0, old=True):
    """
    Plots the difference between current and old projections of the dataset in the viewer.

    Args:
        viewer (napari.Viewer): The napari viewer instance.
        dataset (Dataset): The dataset object.
        projection (int): The projection axis (e.g., 0 for X, 1 for Y, 2 for Z).
        channel (int, optional): The channel to plot. Default is 0.
        old (bool, optional): Whether to plot old projections. Default is True.
    """
    cmaps = ['red', 'green', 'blue']

    scale = list(dataset._scale)
    del scale[projection]

    path = os.path.join(dataset._save_folder, "projections", f"projections_ch{channel}", f"joint_projections_{projection}.tiff")
    image_current_registered = imread(path)[:-1]
    viewer.add_image(image_current_registered, scale=scale, opacity=0.5, colormap=cmaps[0], blending='additive')

    image_next_registered = imread(path)[1:]
    viewer.add_image(image_next_registered, scale=scale, opacity=0.5, colormap=cmaps[1], blending='additive')

    path = os.path.join(dataset._save_folder, "projections", f"old_projections_ch{channel}", f"joint_old_projections_{projection}.tiff")
    if old and not os.path.exists(path):
        print("Warning: Old projection file does not exist.")
        return

    image_next_unregistered = imread(path)[1:]
    viewer.add_image(image_next_unregistered, scale=scale, opacity=0.5, colormap=cmaps[2], blending='additive')

def plot_vectorfield(viewer, dataset):
    """
    Plots the vectorfield of the dataset in the viewer.

    Args:
        viewer (napari.Viewer): The napari viewer instance.
        dataset (Dataset): The dataset object.
    """
    path = os.path.join(dataset._save_folder, "vectorfield", "vectorfield.npy")
    if not os.path.exists(path):
        raise ValueError("Vectorfield file does not exist.")

    vectorfield = np.load(path)

    viewer.add_vectors(vectorfield)

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