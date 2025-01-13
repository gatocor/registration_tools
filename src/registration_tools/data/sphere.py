import os
import tifffile
import numpy as np

def dataset_sphere(path="data", num_images=10, image_size=100, num_channels=1, min_radius=5, max_radius=20, jump=2, stride=(1, 2, 3), verbose=False):
    """
    This function creates a series of 3D images with spheres positioned along an L-shaped path.
    The radius of the spheres increases linearly from min_radius to max_radius across the images.
    The spheres have Gaussian intensity from the center to the border.
    The images are saved in the specified directory with a specified stride.

    Parameters:
        path (str): The directory where the images will be saved. Default is "data".
        num_images (int): The number of images to generate. Default is 10.
        image_size (int): The size of each dimension of the cubic images. Default is 100.
        num_channels (int): The number of channels for the images. Default is 1.
        min_radius (int): The minimum radius of the spheres. Default is 5.
        max_radius (int): The maximum radius of the spheres. Default is 20.
        jump (int): The step size for the L-shaped path points. Default is 2.
        stride (tuple): The stride to apply when saving the images. Default is (1, 2, 3).
        verbose (bool): If True, print detailed information. Default is False.

    Returns:
        None
    """
    # Create the directory
    try:
        os.makedirs(path)
        if verbose:
            print(f"Directory '{path}' created successfully")
    except FileExistsError:
        if verbose:
            print(f"Directory '{path}' already exists")

    # Create subfolders for each channel
    for channel in range(num_channels):
        channel_path = os.path.join(path, f"channel_{channel}")
        try:
            os.makedirs(channel_path)
            if verbose:
                print(f"Directory '{channel_path}' created successfully")
        except FileExistsError:
            if verbose:
                print(f"Directory '{channel_path}' already exists")

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
        image[mask] = 255 * np.exp(-0.5 * (distances[mask] / radius)**2)
        
        for channel in range(num_channels):
            channel_path = os.path.join(path, f"channel_{channel}")
            tifffile.imwrite(os.path.join(channel_path, f'sphere_{i:02d}.tiff'), image[::stride[0],::stride[1],::stride[2]])
            if verbose:
                print(f"Image 'sphere_{i:02d}.tiff' saved successfully in '{channel_path}'")
