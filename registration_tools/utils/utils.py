from tqdm import tqdm
import numpy as np
import os
from skimage.io import imsave
from .auxiliar import _get_axis_scale, _make_index, _get_spatial_dims, _dict_axis_shape, _create_data
import zarr

def apply_function_in_time(dataset, function, axis=None, scale=None, out=None, new_axis=None, new_scale=None, **kwargs):

    axis, scale = _get_axis_scale(dataset, axis, scale)

    if "T" not in axis:
        raise ValueError("The axis must contain the time dimension 'T'.")
    if new_axis is not None:
        if "T" not in axis:
            raise ValueError("The new_axis must contain the time dimension 'T'.")

    t = _dict_axis_shape(axis, dataset.shape)["T"]

    index = _make_index(0, axis)
    data = dataset[index]
    modified = function(data, **kwargs)

    if len(data.shape) != len(modified.shape) and (new_axis is None):
        raise ValueError("The shape of the modified data is different from the original data. Please specify the new axis.")
    elif new_axis is None:
        new_axis = axis
    
    if new_scale is None:
        spatial_dims = _get_spatial_dims(axis)
        original_spatial_shape = [dataset.shape[axis.index(dim)] for dim in spatial_dims]
        spatial_dims = _get_spatial_dims(new_axis)
        modified_spatial_shape = [modified.shape[new_axis.replace("T","").index(dim)] for dim in spatial_dims]
        
        if original_spatial_shape == modified_spatial_shape:
            new_scale = scale
        else:
            raise ValueError("The spatial dimensions of the modified data do not match the original data. Please specify the new scale.")

    #Update shape
    new_shape = []
    d = _dict_axis_shape(axis, dataset.shape)
    d = _dict_axis_shape(new_axis.replace("T",""), modified.shape)
    for i,a in enumerate(new_axis):
        if a in d.keys():
            new_shape.append(d[a])
        else:
            new_shape.append(t)
    
    data = _create_data(new_shape, modified.dtype, new_axis, new_scale, out)
    for t in tqdm(range(dataset.shape[axis.index("T")]), desc="Applying function"):
        index = _make_index(t, axis)
        new_index = _make_index(t, new_axis)
        data[new_index] = function(dataset[index], **kwargs)

    if out is None:
        return data
    else:
        return

def project(dataset, projection_axis, axis=None, scale=None, out=None, style="max"):

    axis, scale = _get_axis_scale(dataset, axis, scale)

    def _max(data, pos=0):
        return data.max(axis=pos)

    def _mean(data, pos=0):
        return data.mean(axis=pos)

    if projection_axis not in "XYZ":
        raise ValueError("The projection axis must be one of 'X', 'Y', or 'Z'.")
    
    if projection_axis not in axis: 
        raise ValueError("The projection axis must be present in the dataset axis.")

    pos = axis.replace("T","").index(projection_axis)
    new_axis = axis.replace(projection_axis, "")
    new_scale = tuple([i for i,ax in zip(scale, _get_spatial_dims(axis)) if ax != projection_axis])

    if style == "max":
        return apply_function_in_time(dataset, _max, axis, scale, out, new_axis, new_scale, pos=pos)
    elif style == "mean":
        return apply_function_in_time(dataset, _mean, axis, scale, out, new_axis, new_scale, pos=pos)
    else:
        raise ValueError("The style must be either 'max' or 'mean'.")
