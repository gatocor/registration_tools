from tqdm import tqdm
import numpy as np
import scipy.ndimage as ndi
import os
from .auxiliar import _get_axis_scale, _make_index, _get_spatial_dims, _dict_axis_shape, _create_data, make_index
import zarr
from copy import deepcopy
import dask.array as da
from dask.diagnostics import ResourceProfiler
from dask.distributed import LocalCluster
from .progressbar import TqdmProgressBar
import psutil

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cndi
    import cucim.skimage.measure as cskm
    GPU_AVAILABLE = True
    print("GPU_AVAILABLE!, Steps that can be accelerated with CUDA will be passed to the GPU.")
except ImportError:
    GPU_AVAILABLE = False

def apply_function_in_time_dask(dataset, function, axis=None, scale=None, out=None, new_axis=None, new_scale=None, cluster=LocalCluster, cluster_kwargs={}, infer_n_workers=True, verbose=True, **kwargs):
    """
    Apply a given function to a dataset along the time dimension.

    Parameters:

        dataset (array-like) : The input dataset to which the function will be applied.
        function (callable) : The function to apply to the dataset. It should accept an array-like object as its first argument.
        axis (str, optional) : The axis labels of the dataset. Must contain the time dimension 'T'.
        scale (list or tuple, optional) : The scale of the dataset along each axis.
        out (out, optional) : An optional file to store the results. If not provided, a new array will be created.
        new_axis (str, optional) : The axis labels of the modified data. Must contain the time dimension 'T'.
        new_scale (list or tuple, optional) : The scale of the modified data along each axis.
        **kwargs (dict) : Additional keyword arguments to pass to the function.

    Returns:
        array-like : The modified dataset with the function applied along the time dimension. If `out` is provided, the function returns None.
    """

    axis, scale = _get_axis_scale(dataset, axis, scale)

    if "T" not in axis:
        raise ValueError("The axis must contain the time dimension 'T'.")
    if new_axis is not None:
        if "T" not in axis:
            raise ValueError("The new_axis must contain the time dimension 'T'.")

    t = _dict_axis_shape(axis, dataset.shape)["T"]

    # Create delayed arrays for each image, applying the custom function
    try:
        dataset_dask = da.from_zarr(dataset)
    except:
        dataset_dask = da.from_array(dataset)
    dataset_dask = dataset_dask.rechunk([1 if j == "T" else i for i,j in zip(dataset.shape, axis)])

    drop_axis_pos = [i for i,j in enumerate(axis) if j not in new_axis]
    new_shape = [dataset.shape[i] for i,j in enumerate(axis) if j in new_axis]
    new_axis_pos = [i for i,j in enumerate(new_axis) if j not in axis]
    slices = [j if i != "T" else 1 for i,j in zip(new_axis, new_shape)]

    available_memory = psutil.virtual_memory().available / 1e9  # Convert bytes to GB
    cluster_kwargs_ = deepcopy(cluster_kwargs)
    cluster_kwargs_['n_workers'] = 1
    cluster_ = cluster(**cluster_kwargs_)
    client = cluster_.get_client()

    index = make_index(axis, T=slice(None,1,None))
    with ResourceProfiler(dt=0.01) as rprof:
        data = dataset_dask[index]
        modified = data.map_blocks(
                        function,        
                        dtype=dataset_dask.dtype,   
                        chunks=slices,
                        drop_axis=drop_axis_pos,
                        new_axis=new_axis_pos
                    )
        modified = modified.compute()

    client.close()

    peak_memory_gb = max([entry[1]/1e3 for entry in rprof.results])
    n_workers = max(1,int(np.floor(available_memory / (np.ceil(peak_memory_gb)))))  # Add some buffer

    if verbose and infer_n_workers:
        print(f"Available memory: {available_memory} GB")
        print(f"Peak memory: {peak_memory_gb} GB")
        print(f"Maximum number of workers: {n_workers}")

    if len(axis) != len(modified.shape) and (new_axis is None):
        raise ValueError("The shape of the modified data is different from the original data. Please specify the new axis.")
    elif len(new_axis) != len(modified.shape):
        raise ValueError(f"The new_axis must have the same length as the modified data shape. Provided axis: {new_axis}, new_axis: {modified.shape} of length {len(modified.shape)}.")
    elif new_axis is None:
        new_axis = axis
    
    if new_scale is None:
        spatial_dims = _get_spatial_dims(axis)
        original_spatial_shape = [dataset.shape[axis.index(dim)] for dim in spatial_dims]
        spatial_dims = _get_spatial_dims(new_axis)
        new_axis_ = [i if i in "XYZ" else "" for i in new_axis]
        modified_spatial_shape = [modified.shape[new_axis_.index(dim)] for dim in spatial_dims]

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
    
    if GPU_AVAILABLE:
        lock = Lock()
        def _apply_function(image, function, lock, **kwargs):
            """
            Apply a custom function to an image.
            """
            with lock:
                img = function(image, **kwargs)
            # img = function(image, **kwargs)
            return img
    else:
        lock = None
        def _apply_function(image, function, lock=None, **kwargs):
            """
            Apply a custom function to an image.
            """
            return function(image, **kwargs)

    new_array = dataset_dask.map_blocks(
        _apply_function, 
        function=function,
        lock=lock, 
        dtype=dataset_dask.dtype,
        chunks=slices,
        drop_axis=drop_axis_pos,
        new_axis=new_axis_pos
    )
    
    if infer_n_workers and "n_workers" not in cluster_kwargs:
        cluster_kwargs = deepcopy(cluster_kwargs)
        cluster_kwargs['n_workers'] = n_workers

    print(cluster_kwargs)
    cluster_ = cluster(**cluster_kwargs)
    client = cluster_.get_client()

    if out is None:
        future = client.compute(new_array)
        pb = TqdmProgressBar([future], total=t)
    else:
        arr = new_array.to_zarr(out, compute=False)
        future = client.compute(arr)
        pb = TqdmProgressBar([future], total=t)    

    client.close()

    # Add attributes to the Zarr file (such as axis and scale)
    if out is None:
        data = zarr.array(data)
    else:
        data = zarr.open_array(out, mode="r+")
    
    data.attrs['axis'] = new_axis
    data.attrs['scale'] = new_scale
    
    return data
    
def apply_function_in_time(dataset, function, axis=None, scale=None, out=None, new_axis=None, new_scale=None, **kwargs):
    """
    Apply a given function to a dataset along the time dimension.

    Parameters:

        dataset (array-like) : The input dataset to which the function will be applied.
        function (callable) : The function to apply to the dataset. It should accept an array-like object as its first argument.
        axis (str, optional) : The axis labels of the dataset. Must contain the time dimension 'T'.
        scale (list or tuple, optional) : The scale of the dataset along each axis.
        out (out, optional) : An optional file to store the results. If not provided, a new array will be created.
        new_axis (str, optional) : The axis labels of the modified data. Must contain the time dimension 'T'.
        new_scale (list or tuple, optional) : The scale of the modified data along each axis.
        **kwargs (dict) : Additional keyword arguments to pass to the function.

    Returns:
        array-like : The modified dataset with the function applied along the time dimension. If `out` is provided, the function returns None.
    """

    nx = cp if GPU_AVAILABLE else np 

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
        m = function(nx.array(dataset[index]), **kwargs)
        data[new_index] = m.get() if GPU_AVAILABLE else m

    return data

def downsample(dataset, factor, axis=None, scale=None, out=None, style="zoom", order=1):
    """
    Downsample a dataset by a given factor along each spatial dimension.

    Parameters:

        dataset (array-like) : The input dataset to be downsampled.
        factor (tuple) : The downsampling factor along each spatial dimension.
        scale (tuple, optional) : The scale of the dataset. If not provided, it will be inferred.
        out (str, optional) : The output file to store the result. If not provided, a new array will be created.
        style (str, optional) : The downsampling style to use. Must be either 'max' for maximum downsampling or 'mean' for mean downsampling. Default is 'mean'.

    Returns:
        array-like : The downsampled dataset.
    """

    nx = cp if GPU_AVAILABLE else np 
    ndix = cndi if GPU_AVAILABLE else ndi

    axis, scale = _get_axis_scale(dataset, axis, scale)

    new_scale = tuple([s/f for s,f in zip(scale, factor)])

    if "C" in axis:
        factor = tuple([1] + list(factor))

    return apply_function_in_time_dask(
                    dataset, 
                    ndix.zoom, 
                    axis=axis,
                    scale=scale, 
                    out=out, 
                    new_axis=axis, 
                    new_scale=new_scale, 
                    zoom=factor, 
                    order=order)

def project(dataset, projection_axis, axis=None, scale=None, out=None, style="max"):
    """
    Project a dataset along a specified axis using either maximum or mean projection.
    
    Parameters:

        dataset (array-like) : The input dataset to be projected.
        projection_axis (str) : The axis along which to project the dataset. Must be one of 'X', 'Y', or 'Z'.
        axis (str, optional) : The axis labels of the dataset. If not provided, it will be inferred.
        scale (tuple, optional) : The scale of each axis in the dataset. If not provided, it will be inferred.
        out (str, optional) : The output file to store the result. If not provided, a new array will be created.
        style (str, optional) : The projection style to use. Must be either 'max' for maximum projection or 'mean' for mean projection. Default is 'max'.

    Returns:
        array-like : The projected dataset.
    """

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
        return apply_function_in_time_dask(dataset, _max, axis, scale, out, new_axis, new_scale, pos=pos)
    elif style == "mean":
        return apply_function_in_time_dask(dataset, _mean, axis, scale, out, new_axis, new_scale, pos=pos)
    else:
        raise ValueError("The style must be either 'max' or 'mean'.")
