import inspect
from skimage import data
import numpy as np
from qtpy.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QComboBox
)
import napari

# Get all skimage.data functions that return images
def get_skimage_dataset_functions():
    data_funcs = {}
    for name, func in inspect.getmembers(data, inspect.isfunction):
        try:
            result = func()
            if isinstance(result, np.ndarray):
                data_funcs[name] = func
        except Exception:
            continue
    return data_funcs

data_funcs = get_skimage_dataset_functions()

def load_data(viewer, name, func):
    image = func()
    viewer.add_image(image, name=name, colormap="gray")
