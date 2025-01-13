import os
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
import napari

from skimage.io import imread, imsave

def plot_projections(viewer, dataset, projection, channels=None, old=False):
    """
    Plots the projections of the dataset in the viewer.

    Args:
        viewer (napari.Viewer): The napari viewer instance.
        dataset (Dataset): The dataset object.
        projection (str): The projection type (e.g., 'XY', 'XZ', 'YZ').
        channels (list, optional): List of channels to plot. If None, all channels are plotted.
        old (bool, optional): Whether to plot old projections. Default is False.
    """
    cmaps = ['red', 'green', 'blue', 'magenta', 'yellow', 'cyan']

    add_old = ""
    if old:
        add_old = "old_"

    if channels is None:
        channels = range(dataset.num_channels)

    for pos, ch in enumerate(channels):
        path = os.path.join(dataset._save_path, "projections", f"{add_old}projections_ch{ch}", f"joint_{add_old}projections_{projection}.npy")
        if not os.path.exists(path):
            raise ValueError("Projection file does not exist.")

        image = imread(path)

        viewer.add_image(image, opacity=0.5, colormap=cmaps[pos], blending='additive')

def plot_projections_difference(viewer, dataset, projection, channel=0, old=True):
    """
    Plots the difference between current and old projections of the dataset in the viewer.

    Args:
        viewer (napari.Viewer): The napari viewer instance.
        dataset (Dataset): The dataset object.
        projection (str): The projection type (e.g., 'XY', 'XZ', 'YZ').
        channel (int, optional): The channel to plot. Default is 0.
        old (bool, optional): Whether to plot old projections. Default is True.
    """
    cmaps = ['red', 'green', 'blue']

    path = os.path.join(dataset._save_path, "projections", f"projections_ch{ch}", f"joint_projections_{projection}.npy")
    image_current_registered = imread(path).take(slice(None,-1,None), axis=0)
    viewer.add_image(image_current_registered, opacity=0.5, colormap=cmaps[0], blending='additive')

    image_next_registered = imread(path).take(slice(1,None,None), axis=0)
    viewer.add_image(image_next_registered, opacity=0.5, colormap=cmaps[1], blending='additive')

    path = os.path.join(dataset._save_path, "projections", f"old_projections_ch{ch}", f"joint_old_projections_{projection}.npy")
    if old and not os.path.exists(path):
        print("Warning: Old projection file does not exist.")
        return

    image_next_unregistered = imread(path).take(slice(1,None,None), axis=0)
    viewer.add_image(image_next_unregistered, opacity=0.5, colormap=cmaps[2], blending='additive')

def plot_vectorfield(viewer, dataset):
    """
    Plots the vectorfield of the dataset in the viewer.

    Args:
        viewer (napari.Viewer): The napari viewer instance.
        dataset (Dataset): The dataset object.
    """
    path = os.path.join(dataset._save_path, "vectorfield", "vectorfield.npy")
    if not os.path.exists(path):
        raise ValueError("Vectorfield file does not exist.")

    vectorfield = np.load(path)

    viewer.add_vectors(vectorfield)

def visualize_images(viewer, dataset, channels, save_folder, numbers=None, scale=(1, 1, 1), downscale=(1, 1, 1), verbosity=1, view_mode='3D', **kwargs):
    """
    Visualizes the images in the dataset using the napari viewer.

    Args:
        viewer (napari.Viewer): The napari viewer instance.
        dataset (Dataset): The dataset object.
        channels (list): List of channels to visualize.
        save_folder (str): The folder to save the visualizations.
        numbers (list, optional): List of numbers to visualize. If None, all numbers in the dataset are visualized.
        scale (tuple, optional): Scale for the images. Default is (1, 1, 1).
        downscale (tuple, optional): Downscale factor for the images. Default is (1, 1, 1).
        verbosity (int, optional): Verbosity level. Default is 1.
        view_mode (str or tuple, optional): View mode(s) for the images. Default is '3D'.
        **kwargs: Additional keyword arguments.
    """
    valid_view_modes = {'3D', 'projection', 'sections'}
    
    if isinstance(view_mode, str):
        view_mode = (view_mode,)
    elif isinstance(view_mode, tuple):
        if not all(isinstance(vm, str) for vm in view_mode):
            raise ValueError("All elements in view_mode must be strings")
    else:
        raise ValueError("view_mode must be a string or a tuple of strings")
    
    if not all(vm in valid_view_modes for vm in view_mode):
        raise ValueError(f"view_mode can only contain the following values: {valid_view_modes}")

    if numbers is None:
        numbers = dataset.numbers
    if not isinstance(numbers, list) or not all(isinstance(n, int) for n in numbers):
        raise ValueError("numbers must be a list of integers")
    if not all(n in dataset.numbers for n in numbers):
        raise ValueError("All elements in numbers must be present in dataset.numbers")
    
    for pos in dataset.numbers:
        if verbosity > 0:
            print(f"Making images")
        
        vs = ast.literal_eval(dataset.loc[pos, "voxel_size"])
        
        files = []
        for ch in channels:
            fs = [str(i) for i in np.sort(os.listdir(dataset['directory'][pos])) if f"ch{ch}" in i]
            step = int(max(np.floor(len(fs) / kwargs.get('n_plots', 1)), 1))
            fs = fs[::step]
            if len(fs) > 0:
                files.append(fs)
        
        os.makedirs(f"{save_folder}/dataset_{dataset['id'][pos]}", exist_ok=True)
        for chs in tqdm(zip(*files), total=len(files[0])):
            for i in ["Image", "Image [1]", "Image [2]"]:
                try:
                    del viewer.layers[i]
                except:
                    pass
            if not os.path.exists(f"{save_folder}/dataset_{dataset['id'][pos]}/{chs[0].split('_')[0]}.png"):
                img = skimage.io.imread(f"{dataset['directory'][pos]}/{chs[0]}")[np.newaxis, ::downscale[0], ::downscale[1], ::downscale[2]]
                for i in range(len(chs[1:])):
                    img = np.concatenate([img, skimage.io.imread(f"{dataset['directory'][pos]}/{chs[i + 1]}")[np.newaxis, ::downscale[0], ::downscale[1], ::downscale[2]]], axis=0)
                
                for vm in view_mode:
                    if vm == '3D':
                        viewer.add_image(img, scale=[downscale[0] * vs[0], downscale[1] * vs[1], downscale[2] * vs[2]], channel_axis=0)
                    elif vm == 'projection':
                        viewer.add_image(np.max(img, axis=1), scale=[downscale[0] * vs[0], downscale[1] * vs[1]], channel_axis=0)
                    elif vm == 'sections':
                        for z in range(img.shape[1]):
                            viewer.add_image(img[:, z, :, :], scale=[downscale[1] * vs[1], downscale[2] * vs[2]], channel_axis=0)
                
                viewer.screenshot(f"{save_folder}/dataset_{dataset['id'][pos]}/{chs[0].split('_')[0]}.png", canvas_only=True)

