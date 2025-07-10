from ..registration import RegistrationVT, RegistrationSITK
# from ..utils import project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image

def apply_threshold(image, threshold):
    """
    Apply a threshold to the image.
    
    Parameters:
    - image: The input image.
    - threshold: The threshold value.
    
    Returns:
    - The binary image after applying the threshold.
    """
    return (image > threshold)

def test_scalar(value):
    return value

def project(dataset, axis, style="max"):
    """
    Project the dataset along a specified axis using a specified style.
    
    Parameters:
    - dataset (image): The input image dataset.
    - axis (int or tuple or ints): The axis along which to project.
    - style (str): The projection style ('max', 'min', 'mean', etc.).
    
    Returns:
    - The projected image.
    """
    # Placeholder implementation
    if style == 'max':
        return dataset.max(axis=axis, keepdims=True)  # This should be replaced with actual projection logic
    elif style == 'min':
        return dataset.min(axis=axis, keepdims=True)
    elif style == 'mean':
        return dataset.mean(axis=axis, keepdims=True)
    
def add_images(image1, image2):
    """
    Add two images together.
    
    Parameters:
    - image1: The first image.
    - image2: The second image.
    
    Returns:
    - The resulting image after addition.
    """
    return image1 + image2  # This should be replaced with actual addition logic

def subtract_images(image1, image2):
    """
    Subtract one image from another.
    
    Parameters:
    - image1: The first image.
    - image2: The second image.
    
    Returns:
    - The resulting image after subtraction.
    """
    return image1 - image2  # This should be replaced with actual subtraction logic

def multiply_images(image1, image2):
    """
    Multiply two images together.
    
    Parameters:
    - image1: The first image.
    - image2: The second image.
    
    Returns:
    - The resulting image after multiplication.
    """
    return image1 * image2  # This should be replaced with actual multiplication logic

def divide_images(image1, image2):
    """
    Divide one image by another.
    
    Parameters:
    - image1: The first image.
    - image2: The second image.
    
    Returns:
    - The resulting image after division.
    """
    return image1 / image2  # This should be replaced with actual division logic

def reflect_image(image, axis):
    """
    Reflect the image along a specified axis.
    
    Parameters:
    - image: The input image.
    - axis: The axis along which to reflect the image.
    
    Returns:
    - The reflected image.
    """
    slicing = [slice(None) if i not in axis else slice(None, None, -1) for i in range(image.ndim)]
    return image[slicing]  # This should be replaced with actual reflection logic

def lineplot(dataset, x, hue=None, scale=None, hue_step=1, x_step=1, xmin=None, xmax=None, ymin=None, ymax=None, cmap="viridis", figsize=(10,5), legend=True, legend_location="best"):
    """
    Create a line plot from the dataset.
    Parameters:
    - dataset: The input dataset to plot.
    - x: The x-axis values.
    - hue: The hue variable for color encoding (optional).
    - scale: Scaling factor for the y-axis (optional).
    - hue_step: Step size for the hue variable (default: 1).
    - x_step: Step size for the x-axis values (default: 1).
    - xmin: Minimum x value for the plot (optional).
    - xmax: Maximum x value for the plot (optional).
    - ymin: Minimum y value for the plot (optional).
    - ymax: Maximum y value for the plot (optional).
    - cmap: Colormap for the plot (default: "viridis").
    - figsize: Size of the figure (default: (10, 5)).
    - legend: Whether to show the legend (default: True).
    - legend_location: Location of the legend (default: "best").
    
    Returns:
    - figure_image: Displays the plot.
    """

    if scale is None:
        scale = [1] * dataset.ndim

    slicing = [slice(None) if i not in [x,hue] else (slice(None,None,x_step) if i == x else slice(None, None,hue_step)) for i in range(dataset.ndim)]  # Create a list of slices for all dimensions
    dataset_reduced = dataset[slicing]  # Reduce the dataset by the specified step size for x

    Y = dataset_reduced.flatten()  # Flatten the y values
    
    slicing = [1 if x != i else -1 for i in range(dataset_reduced.ndim)]  # Create a list of slices for all dimensions
    X = (np.ones(dataset_reduced.shape)*np.linspace(0, dataset_reduced.shape[x], dataset_reduced.shape[x]).reshape(slicing)).flatten()*scale[x]* x_step  # Flatten the x values and apply scaling

    fig, ax = plt.subplots(figsize=figsize)
    if hue is not None:
        slicing_hue = [1 if hue != i else -1 for i in range(dataset_reduced.ndim)]  # Create a list of slices for hue dimension
        hue_values = (np.ones(dataset_reduced.shape)*np.linspace(0, dataset_reduced.shape[hue], dataset_reduced.shape[hue]).reshape(slicing_hue)).flatten()*scale[hue]* hue_step  # Flatten the hue values and apply scaling
        sns.lineplot(x=X, y=Y, hue=hue_values, palette=cmap, ax=ax)
    else:
        sns.lineplot(x=X, y=Y, ax=ax)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc=legend_location) if legend else ax.get_legend().remove()

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    image_pil = Image.open(buf).convert("RGBA")
    image_np = np.array(image_pil)

    return image_np

def scatterplot(dataset, x, hue=None, scale=None, hue_step=1, x_step=1, xmin=None, xmax=None, ymin=None, ymax=None, cmap="viridis", figsize=(10,5), legend=True, legend_location="best"):
    """
    Create a line plot from the dataset.
    Parameters:
    - dataset: The input dataset to plot.
    - x: The x-axis values.
    - hue: The hue variable for color encoding (optional).
    - scale: Scaling factor for the y-axis (optional).
    - hue_step: Step size for the hue variable (default: 1).
    - x_step: Step size for the x-axis values (default: 1).
    - xmin: Minimum x value for the plot (optional).
    - xmax: Maximum x value for the plot (optional).
    - ymin: Minimum y value for the plot (optional).
    - ymax: Maximum y value for the plot (optional).
    - cmap: Colormap for the plot (default: "viridis").
    - figsize: Size of the figure (default: (10, 5)).
    - legend: Whether to show the legend (default: True).
    - legend_location: Location of the legend (default: "best").
    
    Returns:
    - figure_image: Displays the plot.
    """

    if scale is None:
        scale = [1] * dataset.ndim

    slicing = [slice(None) if i not in [x,hue] else (slice(None,None,x_step) if i == x else slice(None, None,hue_step)) for i in range(dataset.ndim)]  # Create a list of slices for all dimensions
    dataset_reduced = dataset[slicing]  # Reduce the dataset by the specified step size for x

    Y = dataset_reduced.flatten()  # Flatten the y values
    
    slicing = [1 if x != i else -1 for i in range(dataset_reduced.ndim)]  # Create a list of slices for all dimensions
    X = (np.ones(dataset_reduced.shape)*np.linspace(0, dataset_reduced.shape[x], dataset_reduced.shape[x]).reshape(slicing)).flatten()*scale[x]* x_step  # Flatten the x values and apply scaling

    fig, ax = plt.subplots(figsize=figsize)
    if hue is not None:
        slicing_hue = [1 if hue != i else -1 for i in range(dataset_reduced.ndim)]  # Create a list of slices for hue dimension
        hue_values = (np.ones(dataset_reduced.shape)*np.linspace(0, dataset_reduced.shape[hue], dataset_reduced.shape[hue]).reshape(slicing_hue)).flatten()*scale[hue]* hue_step  # Flatten the hue values and apply scaling
        sns.scatterplot(x=X, y=Y, hue=hue_values, palette=cmap, ax=ax)
    else:
        sns.scatterplot(x=X, y=Y, ax=ax)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc=legend_location) if legend else ax.get_legend().remove()

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    image_pil = Image.open(buf).convert("RGBA")
    image_np = np.array(image_pil)

    return image_np

def heatmap(dataset, scale=None, x_step=1, y_step=1, vmin=None, vmax=None, cmap="viridis", figsize=(10,5)):
    """
    Create a line plot from the dataset.
    Parameters:
    - dataset: The input dataset to plot.
    - scale: Scaling factor for the y-axis (optional).
    - x_step: Step size for the x-axis values (default: 1).
    - y_step: Step size for the y-axis values (default: 1).
    - vmin: Minimum value for the color scale (optional).
    - vmax: Maximum value for the color scale (optional).
    - cmap: Colormap for the plot (default: "viridis").
    - figsize: Size of the figure (default: (10, 5)).
    - legend: Whether to show the legend (default: True).
    - legend_location: Location of the legend (default: "best").
    
    Returns:
    - figure_image: Displays the plot.
    """
    

    slicing = [slice(None) if i != 1 else 0 for i in dataset.shape]  # Create a list of slices for all dimensions
    dataset_reduced = pd.DataFrame(dataset[slicing][::x_step, ::y_step])  # Reduce the dataset by the specified step size for x

    if scale is None:
        scale = [1] * dataset_reduced.ndim

    dataset_reduced.index = dataset_reduced.index * scale[0] * x_step
    dataset_reduced.columns = dataset_reduced.columns * scale[1] * y_step

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dataset_reduced, cmap=cmap, ax=ax, vmin=vmin, vmax=vmax)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    image_pil = Image.open(buf).convert("RGBA")
    image_np = np.array(image_pil)

    return image_np

NODES_REGISTRATION_TOOLS = { 
    "filters" : {
        "rt_tools.filters.apply_filter": {
            "name": "apply_filter",
            "function_name": "apply_filter",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "image",
                    "name": "image"
                }
            },
            "function": apply_threshold,
            "help": "",
            "parameters": {
                "image": {
                    "type": "image",
                    "default": None
                },
                "threshold": {
                    "type": "scalar",
                    "default": None
                }
            }
        },
    },
    "registration" : {
        "rt_tools.registration.RegistrationVT": {
            "name": "RegistrationVT",
            "function_name": "RegistrationVT",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "Registration",
                    "name": "registration_object"
                }
            },
            "function": RegistrationVT,  # Placeholder function
            "help": "",
            "parameters": {
                "registration_type": {
                    "type": str,
                    "default": "rigid"
                },
                "pyramid_lowest_level": {
                    "type": int,
                    "default": 0
                },
                "pyramid_highest_level": {
                    "type": int,
                    "default": 3
                },
                "args_registration": {
                    "type": str,
                    "default": ""
                }
            }
        },
        "rt_tools.registration.RegistrationSITK": {
            "name": "RegistrationSITK",
            "function_name": "RegistrationSITK",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "Registration",
                    "name": "registration_object"
                }
            },
            "function": RegistrationVT,  # Placeholder function
            "help": "",
            "parameters": {
                "registration_type": {
                    "type": str,
                    "default": "rigid"
                },
                "metric": {
                    "type": str,
                    "default": "MeanSquares"
                },
                "optimizer": {
                    "type": str,
                    "default": "RegularStepGradientDescent"
                },
                "sampling": {
                    "type": str,
                    "default": "None"
                },
                "optimizer_learning_rate": {
                    "type": float,
                    "default": 0.1
                },
                "optimizer_minStep": {
                    "type": float,
                    "default": 1e-5
                },
                "optimizer_numberOfIterations": {
                    "type": int,
                    "default": 500
                },
                "optimizer_gradientMagnitudeTolerance": {
                    "type": float,
                    "default": 1e-10
                },
                "optimizer_radius": {
                    "type": int,
                    "default": 1
                },
                "pyramid_shrink_factors": {
                    "type": tuple,
                    "default": (1,)
                },
                "pyramid_smoothing_sigmas": {
                    "type": tuple,
                    "default": (0,)
                },
                "displacement_field_smoothing_sigmas": {
                    "type": float,
                    "default": 0.
                },
            }
        },
        "rt_tools.registration.Registration.fit_apply": {
            "name": "fit_apply",
            "function_name": "fit_apply",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "image",
                    "name": "registered_image"
                }
            },
            "function": lambda registration_object, **kwargs : registration_object.fit_apply(**kwargs),  # Placeholder function
            "help": "",
            "parameters": {
                "registration_object": {
                    "type": "Registration",
                    "default": None
                },
                "dataset": {
                    "type": "image",
                    "default": None
                },
                # "out_trnsf": {
                #     "type": (str, None),
                #     "default": None
                # },
                # "out_dataset": {
                #     "type": (str, None),
                #     "default": None
                # },
                "direction": {
                    "type": str,
                    "default": "backward"
                },
                "use_channel": {
                    "type": (str, None),
                    "default": None
                },
                "axis": {
                    "type": (str, None),
                    "default": None
                },
                "scale": {
                    "type": (str, None),
                    "default": None
                },
                "downsample": {
                    "type": (tuple, None),
                    "default": None
                },
                "stepping": {
                    "type": int,
                    "default": 1
                },
                "save_behavior": {
                    "type": str,
                    "default": "Continue"
                },
                "verbose": {
                    "type": bool,
                    "default": False
                },
            }
        },
        "rt_tools.registration.Registration.fit": {
            "name": "fit",
            "function_name": "fit",
            "type": "function",
            "outputs": {},
            "function": lambda registration_object, **kwargs : registration_object.fit(**kwargs),  # Placeholder function
            "help": "",
            "parameters": {
                "registration_object": {
                    "type": "Registration",
                    "default": None
                },
                "dataset": {
                    "type": "image",
                    "default": None
                },
                # "out": {
                #     "type": (str, None),
                #     "default": None
                # },
                "direction": {
                    "type": str,
                    "default": "backward"
                },
                "perform_global_trnsf": {
                    "type": bool,
                    "default": False
                },
                "use_channel": {
                    "type": (str, None),
                    "default": None
                },
                "axis": {
                    "type": (str, None),
                    "default": None
                },
                "scale": {
                    "type": (tuple, None),
                    "default": None
                },
                "downsample": {
                    "type": (tuple, None),
                    "default": None
                },
                "stepping": {
                    "type": int,
                    "default": 1
                },
                "save_behavior": {
                    "type": str,
                    "default": "Continue"
                },
                "verbose": {
                    "type": bool,
                    "default": False
                },
            }
        },
        "rt_tools.registration.Registration.apply": {
            "name": "apply",
            "function_name": "apply",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "image",
                    "name": "registered_image"
                }
            },
            "function": lambda registration_object, **kwargs : registration_object.apply(**kwargs),  # Placeholder function
            "help": "",
            "parameters": {
                "registration_object": {
                    "type": "Registration",
                    "default": None
                },
                "dataset": {
                    "type": "image",
                    "default": None
                },
                # "out": {
                #     "type": (str, None),
                #     "default": None
                # },
                "axis": {
                    "type": (str, None),
                    "default": None
                },
                "scale": {
                    "type": (tuple, None),
                    "default": None
                },
                "save_behavior": {
                    "type": str,
                    "default": "Continue"
                },
                "transformation": {
                    "type": str,
                    "default": "global"
                },
                "paddding": {
                    "type": (tuple, None),
                    "default": None
                },                
                "verbose": {
                    "type": bool,
                    "default": False
                },
            }
        },
        "rt_tools.registration.Registration.vectorfield": {
            "name": "vectorfield",
            "function_name": "vectorfield",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "vectors",
                    "name": "vector_field"
                }
            },
            "function": lambda registration_object, **kwargs : registration_object.vectorfield(**kwargs),  # Placeholder function
            "help": "",
            "parameters": {
                "registration_object": {
                    "type": "Registration",
                    "default": None
                },
                "mask": {
                    "type": "image",
                    "default": None
                },
                # "out": {
                #     "type": (str, None),
                #     "default": None
                # },
                "axis": {
                    "type": (str, None),
                    "default": None
                },
                "scale": {
                    "type": (tuple, None),
                    "default": None
                },
                "n_points": {
                    "type": int,
                    "default": 20
                },
                "transformation": {
                    "type": str,
                    "default": "relative"
                },
            }
        }
    },
    "transform" : {
        "rt_tools.transform.project": {
            "name": "project",
            "function_name": "project",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "image",
                    "name": "projected_image"
                }
            },
            "function": project,
            "help": "",
            "parameters": {
                "dataset": {
                    "type": "image",
                    "default": None
                },
                "axis": {
                    "type": (tuple, int),
                    "default": None
                },
                "style": {
                    "type": str,
                    "default": "max"
                }
            }
        },
        "rt_tools.transform.add_images": {
            "name": "add_images",
            "function_name": "add_images",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "image",
                    "name": "added_image"
                }
            },
            "function": add_images,
            "help": "",
            "parameters": {
                "image1": {
                    "type": "image",
                    "default": None
                },
                "image2": {
                    "type": "image",
                    "default": None
                }
            }
        },
        "rt_tools.transform.subtract_images": {
            "name": "subtract_images",
            "function_name": "subtract_images",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "image",
                    "name": "subtracted_image"
                }
            },
            "function": subtract_images,
            "help": "",
            "parameters": {
                "image1": {
                    "type": "image",
                    "default": None
                },
                "image2": {
                    "type": "image",
                    "default": None
                }
            }
        },
        "rt_tools.transform.multiply_images": {
            "name": "multiply_images",
            "function_name": "multiply_images",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "image",
                    "name": "multiplied_image"
                }
            },
            "function": multiply_images,
            "help": "",
            "parameters": {
                "image1": {
                    "type": "image",
                    "default": None
                },
                "image2": {
                    "type": "image",
                    "default": None
                }
            }
        },
        "rt_tools.transform.divide_images": {
            "name": "divide_images",
            "function_name": "divide_images",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "image",
                    "name": "divided_image"
                }
            },
            "function": divide_images,
            "help": "",
            "parameters": {
                "image1": {
                    "type": "image",
                    "default": None
                },
                "image2": {
                    "type": "image",
                    "default": None
                }
            }
        },
        "rt_tools.transform.reflect_image": {
            "name": "reflect_image",
            "function_name": "reflect_image",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "image",
                    "name": "reflected_image"
                }
            },
            "function": reflect_image,
            "help": "",
            "parameters": {
                "image": {
                    "type": "image",
                    "default": None
                },
                "axis": {
                    "type": (tuple, int),
                    "default": None
                }
            }
        }
    },
    "plot" : {
        "rt_tools.plot.lineplot": {
            "name": "lineplot",
            "function_name": "lineplot",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "image",
                    "name": "figure_image"
                }
            },
            "function": lineplot,
            "help": "",
            "parameters": {
                "dataset": {
                    "type": "image",
                    "default": None
                },
                "x": {
                    "type": int,
                    "default": 0
                },
                "hue": {
                    "type": (int, None),
                    "default": None
                },
                "scale": {
                    "type": (tuple, None),
                    "default": None
                },
                "hue_step": {
                    "type": int,
                    "default": 1
                },
                "x_step": {
                    "type": int,
                    "default": 1
                },
                "xmin": {
                    "type": (float, None),
                    "default": None
                },
                "xmax": {
                    "type": (float, None),
                    "default": None
                },
                "ymin": {
                    "type": (float, None),
                    "default": None
                },
                "ymax": {
                    "type": (float, None),
                    "default": None
                },
                "cmap" : {
                    "type" : str,
                    "default" : 'viridis'
                },
                "figsize": {
                    "type": (tuple, None),
                    "default": (10, 5)
                },
                'legend': {
                    'type': bool,
                    'default': True
                },
                'legend_location': {
                    'type': str,
                    'default': 'best'
                }
            }
        },
        "rt_tools.plot.scatterplot": {
            "name": "scatterplot",
            "function_name": "scatterplot",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "image",
                    "name": "figure_image"
                }
            },
            "function": scatterplot,
            "help": "",
            "parameters": {
                "dataset": {
                    "type": "image",
                    "default": None
                },
                "x": {
                    "type": int,
                    "default": 0
                },
                "hue": {
                    "type": (int, None),
                    "default": None
                },
                "scale": {
                    "type": (tuple, None),
                    "default": None
                },
                "hue_step": {
                    "type": int,
                    "default": 1
                },
                "x_step": {
                    "type": int,
                    "default": 1
                },
                "xmin": {
                    "type": (float, None),
                    "default": None
                },
                "xmax": {
                    "type": (float, None),
                    "default": None
                },
                "ymin": {
                    "type": (float, None),
                    "default": None
                },
                "ymax": {
                    "type": (float, None),
                    "default": None
                },
                "cmap" : {
                    "type" : str,
                    "default" : 'viridis'
                },
                "figsize": {
                    "type": (tuple, None),
                    "default": (10, 5)
                },
                'legend': {
                    'type': bool,
                    'default': True
                },
                'legend_location': {
                    'type': str,
                    'default': 'best'
                }
            }
        },
        "rt_tools.plot.heatmap": {
            "name": "heatmap",
            "function_name": "heatmap",
            "type": "function",
            "outputs": {
                "0" : {
                    "type": "image",
                    "name": "figure_image"
                }
            },
            "function": heatmap,
            "help": "",
            "parameters": {
                "dataset": {
                    "type": "image",
                    "default": None
                },
                "scale": {
                    "type": (tuple, None),
                    "default": None
                },
                "x_step": {
                    "type": int,
                    "default": 1
                },
                "y_step": {
                    "type": int,
                    "default": 1
                },
                "vmin": {
                    "type": (float, None),
                    "default": None
                },
                "vmax": {
                    "type": (float, None),
                    "default": None
                },
                'cmap' : {
                    'type' : str,
                    'default' : 'viridis'
                },
                "figsize": {
                    "type": (tuple, None),
                    "default": (10, 5)
                }
            }
        }
    }
}