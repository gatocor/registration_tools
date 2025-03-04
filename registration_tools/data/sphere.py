import os
import tifffile
import numpy as np
from ..dataset.dataset import Dataset
import zarr

def sphere(out=None, num_images=10, image_size=100, num_channels=1, min_radius=5, max_radius=20, jump=2, stride=(3, 2, 1), decay_factor=0.5, verbose=False):
    """
    This function creates a series of 3D images of a sphere moving along an L-shaped path while increasing the radius.
    The radius of the spheres increases linearly from min_radius to max_radius across the images.
    The spheres have Gaussian intensity from the center to the border.
    The images are saved in the specified directory with a specified stride.

    Parameters:
        out (str): The directory or zarr file where the images will be saved. If None, the images are stored in memory. Default is None.
        num_images (int): The number of images to generate. Default is 10.
        image_size (int): The size of each dimension of the cubic images. Default is 100.
        num_channels (int): The number of channels for the images. Default is 1.
        min_radius (int): The minimum radius of the spheres. Default is 5.
        max_radius (int): The maximum radius of the spheres. Default is 20.
        jump (int): The step size for the L-shaped path points. Default is 2.
        stride (tuple): The stride to apply when saving the images. Default is (3, 2, 1).
        decay_factor (float): The exponential decay factor for the Gaussian intensity. Default is 0.5.
        verbose (bool): If True, print detailed information. Default is False.

    Returns:
        Dataset or zarr array: The dataset object containing the generated images if saved to a directory, or a zarr array if saved to a zarr file or in memory.
    """
    if out is None:
        type = "zarr"
        store = zarr.storage.MemoryStore()
        zarr_file = zarr.create_array(store=store, shape=(num_images, num_channels, image_size//stride[0], image_size//stride[1], image_size//stride[2]), dtype=np.uint8)
        zarr_file.attrs["axis"] = "TCZYX"
        zarr_file.attrs["scale"] = stride
    elif ( isinstance(out, str) and out.endswith(".zarr") ):
        type = "zarr"
        zarr_file = zarr.create_array(store=out, mode='w', shape=(num_images, num_channels, image_size//stride[0], image_size//stride[1], image_size//stride[2]), dtype=np.uint8)
        zarr_file.attrs["axis"] = "TCZYX"
        zarr_file.attrs["scale"] = stride
    elif isinstance(out, str):
        # Create the directory
        type = "directory"
        if not os.path.exists(out):
            os.makedirs(out)
            if verbose:
                print(f"Directory '{out}' created successfully")
        else:
            raise FileExistsError(f"Directory '{out}' already exists")
    
        # Create subfolders for each channel
        for channel in range(num_channels):
            channel_path = os.path.join(out, f"channel_{channel}")
            os.makedirs(channel_path)

    # Create L-shaped path with jumps
    center = image_size // 2
    path_points = [(center, center, center + i * jump) for i in range(5)] + [(center, center + i * jump, center + 4 * jump) for i in range(1, 6)]

    # Generate and save images
    for i, (x, y, z) in enumerate(path_points):
        radius = min_radius + (i * (max_radius - min_radius) / (num_images - 1))
        image = np.zeros((image_size, image_size, image_size), dtype=np.uint8)
        
        # Apply Gaussian intensity from the center to the border
        X, Y, Z = np.meshgrid(np.arange(image_size), np.arange(image_size), np.arange(image_size), indexing='ij')
        distances = np.sqrt((X - x)**2 + (Y - y)**2 + (Z - z)**2)
        mask = distances < radius
        image[mask] = 255 * np.exp(-decay_factor * (distances[mask] / radius)**2)
        
        for channel in range(num_channels):
            if type == "zarr":
                zarr_file[i, channel] = image[::stride[0],::stride[1],::stride[2]]
            elif type == "directory":
                channel_path = os.path.join(out, f"channel_{channel}")
                tifffile.imwrite(os.path.join(channel_path, f'sphere_{i:02d}.tiff'), image[::stride[0],::stride[1],::stride[2]])

    if type == "zarr" and out is None:
        return zarr_file
    elif type == "zarr":
        zarr_file.close()
        return zarr.open_array(store=store, mode='r')
    else:
        # Create dataset
        dataset = Dataset([os.path.join(out, f"channel_{ch}", "sphere_{:02d}.tiff") for ch in range(num_channels)], "CT", "ZYX", scale=[st for st in stride])
        return dataset

def rotating_spheres(out=None, num_images=10, image_size=100, num_channels=1, min_radius=5, max_radius=20, rotation_speed=0.1, stride=(3, 2, 1), decay_factor=0.5, verbose=False):
    """
    This function creates a series of 3D images of three spheres rotating around the center of the image.
    The radius of the spheres increases linearly from min_radius to max_radius across the images.
    The spheres have Gaussian intensity from the center to the border.
    The images are saved in the specified directory with a specified stride.
    Parameters:
        out (str): The directory or zarr file where the images will be saved. If None, the images are stored in memory. Default is None.
        num_images (int): The number of images to generate. Default is 10.
        image_size (int): The size of each dimension of the cubic images. Default is 100.
        num_channels (int): The number of channels for the images. Default is 1.
        min_radius (int): The minimum radius of the spheres. Default is 5.
        max_radius (int): The maximum radius of the spheres. Default is 20.
        rotation_speed (float): The speed of rotation for the spheres. Default is 0.1.
        stride (tuple): The stride to apply when saving the images. Default is (3, 2, 1).
        decay_factor (float): The exponential decay factor for the Gaussian intensity. Default is 0.5.
        verbose (bool): If True, print detailed information. Default is False.
    Returns:
        Dataset or zarr array: The dataset object containing the generated images if saved to a directory, or a zarr array if saved to a zarr file or in memory.
    """
    if out is None:
        type = "zarr"
        store = zarr.storage.MemoryStore()
        zarr_file = zarr.create_array(store=store, shape=(num_images, num_channels, image_size//stride[0], image_size//stride[1], image_size//stride[2]), dtype=np.uint8)
        zarr_file.attrs["axis"] = "TCZYX"
        zarr_file.attrs["scale"] = stride
    elif ( isinstance(out, str) and out.endswith(".zarr") ):
        type = "zarr"
        zarr_file = zarr.create_array(store=out, mode='w', shape=(num_images, num_channels, image_size//stride[0], image_size//stride[1], image_size//stride[2]), dtype=np.uint8)
        zarr_file.attrs["axis"] = "TCZYX"
        zarr_file.attrs["scale"] = stride
    elif isinstance(out, str):
        # Create the directory
        type = "directory"
        if not os.path.exists(out):
            os.makedirs(out)
            if verbose:
                print(f"Directory '{out}' created successfully")
        else:
            raise FileExistsError(f"Directory '{out}' already exists")
    
        # Create subfolders for each channel
        for channel in range(num_channels):
            channel_path = os.path.join(out, f"channel_{channel}")
            os.makedirs(channel_path)
    # Generate and save images
    center = image_size // 2
    angles = np.linspace(0, 2 * np.pi * rotation_speed, num_images, endpoint=False)
    for i, angle in enumerate(angles):
        radius = min_radius + (i * (max_radius - min_radius) / (num_images - 1))
        image = np.zeros((image_size, image_size, image_size), dtype=np.uint8)
        
        # Calculate sphere positions
        positions = [
            (center, center + int(center/2 * np.cos(angle + 2 * np.pi / 3 * j)), center + int(center/2 * np.sin(angle + 2 * np.pi / 3 * j)))
            for j in range(3)
        ]
        
        for x, y, z in positions:
            # Apply Gaussian intensity from the center to the border
            X, Y, Z = np.meshgrid(np.arange(image_size), np.arange(image_size), np.arange(image_size), indexing='ij')
            distances = np.sqrt((X - x)**2 + (Y - y)**2 + (Z - z)**2)
            mask = distances < radius
            image[mask] = 255 * np.exp(-decay_factor * (distances[mask] / radius)**2)
        
        for channel in range(num_channels):
            if type == "zarr":
                zarr_file[i, channel] = image[::stride[0],::stride[1],::stride[2]]
            elif type == "directory":
                channel_path = os.path.join(out, f"channel_{channel}")
                tifffile.imwrite(os.path.join(channel_path, f'sphere_{i:02d}.tiff'), image[::stride[0],::stride[1],::stride[2]])
    if type == "zarr" and out is None:
        return zarr_file
    elif type == "zarr":
        zarr_file.close()
        return zarr.open_array(store=store, mode='r')
    else:
        # Create dataset
        dataset = Dataset([os.path.join(out, f"channel_{ch}", "sphere_{:02d}.tiff") for ch in range(num_channels)], "CT", "ZYX", scale=[st for st in stride])
        return dataset