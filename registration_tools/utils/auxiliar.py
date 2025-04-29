import os
from contextlib import contextmanager
import numpy as np
import zarr
from collections import namedtuple
from ..dataset import Dataset

ImgProps = namedtuple("ImgProps", ["shape", "shape_spatial", "axis", "scale", "axis_spatial", "n_spatial"])

def _make_index(t, axis, use_channel=None, downsample=(1,1,1)):
    spatial_order = [i for i in axis if i in "XYZ"]
    index = [slice(None)] * len(axis)
    index[axis.index("T")] = t
    if "C" in axis and use_channel is not None:
        index[axis.index("C")] = use_channel
    for i, ax in enumerate(axis):
        if ax not in ["T", "C"]:
            index[i] = slice(None, None, downsample[spatial_order.index(ax)])
    index = tuple(index)
    return index

def make_index(axis, downsample=(1,1,1), **kwargs):
    spatial_order = [i for i in axis if i in "XYZ"]
    index = [slice(None)] * len(axis)
    for i, ax in enumerate(axis):
        if ax not in kwargs:
            if ax in "XYZ":
                index[i] = slice(None, None, downsample[spatial_order.index(ax)])
        else:
            index[i] = kwargs[ax]

    return tuple(index)

def _get_img_prop(mask, axis, scale, requires="T"):

    if axis is None:
        if isinstance(mask, Dataset):
            axis = mask._axis
        elif isinstance(mask, zarr.Array) and "axis" in mask.attrs:
            axis = mask.attrs["axis"]
        else:
            raise ValueError("The axis cannot be inferred from dataset data and must be specified.")
            
    if scale is None:
        if isinstance(mask, Dataset):
            scale = mask.scale
        elif isinstance(mask, zarr.Array) and "scale" in mask.attrs:
            scale = mask.attrs["scale"]
        else:
            raise ValueError("The scale cannot be inferred from dataset data and must be specified.")

    if len(axis) != len(mask.shape):
        raise ValueError("The axis must have the same length as the dataset shape.")
    
    missing = [i for i in requires if i not in axis]
    if len(missing) > 0:
        raise ValueError(f"The axis must contain the dimensions {requires}. Missing {missing}")

    shape = mask.shape
    axis_spatial = str([i for i in "XYZ" if i in axis])
    n_spatial = len(axis_spatial)
    shape_spatial = tuple([j for i,j in zip(axis,shape) if i in "XYZ"])


    d = ImgProps(
        shape,
        shape_spatial,
        axis,
        scale,
        axis_spatial,
        n_spatial
    )

    return d

def _get_axis_scale(mask, axis, scale):

    if axis is None:
        if isinstance(mask, Dataset):
            axis = mask._axis
        elif isinstance(mask, zarr.Array) and "axis" in mask.attrs:
            axis = mask.attrs["axis"]
        else:
            raise ValueError("The axis cannot be inferred from dataset data and must be specified.")
            
    if scale is None:
        if isinstance(mask, Dataset):
            scale = mask.scale
        elif isinstance(mask, zarr.Array) and "scale" in mask.attrs:
            scale = mask.attrs["scale"]
        else:
            raise ValueError("The scale cannot be inferred from dataset data and must be specified.")

    if len(axis) != len(mask.shape):
        raise ValueError("The axis must have the same length as the dataset shape.")
        
    if "T" not in axis:
        raise ValueError("The axis must contain the time dimension 'T'.")

    return axis, scale

def _shape_padded(shape, axis, padding):
    
    if padding is not None:
        if np.array(padding).shape != (len([i for i in axis if i in "XYZ"]), 2):
            raise ValueError("The padding must have the same length as the number of spatial dimensions.")

        new_shape=[]
        pos = 0
        p = np.array(padding).sum(axis=1)
        for i,(j,k) in enumerate(zip(axis,shape)):
            if j in "XYZ":
                new_shape.append(k+p[pos])
                pos += 1
            else:
                new_shape.append(k)

        return new_shape
    else:
        return shape

def _trnsf_padded(padding, dims):
    if padding is not None:
        padding_trnsf = np.eye(dims+1, dims+1)
        padding_trnsf[:-1, -1] = np.array(padding)[:,0]

        return padding_trnsf
    else:
        return np.eye(dims+1, dims+1)
    
def _image_padded(img, padding):
    if padding is not None:
        if np.array(padding).shape != (img.ndim, 2):
            raise ValueError("The padding must have the same length as the number of spatial dimensions.")
        
        p = np.array(padding).sum(axis=1)
        if np.all(p == 0):
            img_padded = img
        else:
            img_shape = np.array(img.shape) + p
            img_padded = np.zeros(img_shape, dtype=img.dtype)
            img_padded[tuple(slice(i[0], j+i[0]) for i,j in zip(padding, img.shape))] = img

        return img_padded
    else:
        return img

def _shape_downsampled(shape, axis, downsample):

    new_shape=[]
    pos = 0
    for i,(j,k) in enumerate(zip(axis,shape)):
        if j in "XYZ":
            new_shape.append(int(np.ceil(k/downsample[pos])))
            pos += 1
        else:
            new_shape.append(k)

    return new_shape

def _dict_axis_shape(axis, shape):
    return {ax: shape[i] for i, ax in enumerate(axis)}

def _get_spatial_dims(axis):
    return "".join([dim for dim in axis if dim in "XYZ"])

def _create_data(shape, dtype, axis, scale, out=None):
    
    chunks = tuple([j if i in "XYZ" else 1 for i, j in zip(axis, shape)])
    if out is None:
        store = zarr.storage.MemoryStore()
        data = zarr.create_array(store=store, shape=shape, dtype=dtype, chunks=chunks)
        data.attrs["axis"] = axis
        data.attrs["scale"] = scale
    elif isinstance(out, str) and out.endswith('.zarr'):
        data = zarr.open(out, mode='w', shape=shape, dtype=dtype, chunks=chunks)
        data.attrs["axis"] = axis
        data.attrs["scale"] = scale

    return data

@contextmanager
def _suppress_stdout_stderr():
    """Suppresses stdout and stderr, including from C extensions."""
    # Open null files
    with open(os.devnull, 'w') as devnull:
        # Save the actual stdout (1) and stderr (2) file descriptors
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        try:
            # Redirect stdout and stderr to devnull
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            # Restore stdout and stderr
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
