import numpy as np
import zarr
from ..dataset import Dataset

def _shape_downsampled(shape, axis, downsample=(1,1,1)):
    spatial_order = [i for i in axis if i in "XYZ"]
    shape = np.array(shape)
    for i, ax in enumerate(axis):
        if ax in "XYZ":
            shape[i] = shape[i] // downsample[spatial_order.index(ax)]
    return tuple(shape)

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
        
    return axis, scale

def _dict_axis_shape(axis, shape):
    return {ax: shape[i] for i, ax in enumerate(axis)}


def _get_spatial_dims(axis):
    return "".join([dim for dim in axis if dim in "XYZ"])

def _create_data(shape, dtype, axis, scale, out=None):
    
    if out is None:
        store = zarr.storage.MemoryStore()
        data = zarr.create_array(store=store, shape=shape, dtype=dtype)
        data.attrs["axis"] = axis
        data.attrs["scale"] = scale
    elif isinstance(out, str) and out.endswith('.zarr'):
        data = zarr.open(out, mode='w', shape=shape, dtype=dtype)
        data.attrs["axis"] = axis
        data.attrs["scale"] = scale

    return data
