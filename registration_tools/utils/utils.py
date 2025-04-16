from tqdm import tqdm
import numpy as np
import scipy.ndimage as ndi
import os
from .auxiliar import _get_axis_scale, _make_index, _get_spatial_dims, _dict_axis_shape, _create_data, make_index
import zarr
from copy import deepcopy
import dask
import dask.array as da
from dask.diagnostics import ResourceProfiler
from dask.distributed import LocalCluster, Lock, Worker, get_worker
from .progressbar import TqdmProgressBar
import psutil
import threading
import multiprocessing
import time
import gc
import logging
from .gpuprofiler import GpuProfiler
import tempfile

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cndi
    import cucim.skimage.measure as cskm
    GPU_AVAILABLE = True
    print("GPU_AVAILABLE!, Steps that can be accelerated with CUDA will be passed to the GPU.")
except ImportError:
    GPU_AVAILABLE = False

def apply_function(dataset, function, axis=None, axis_slicing="T", scale=None, out=None, new_axis=None, new_scale=None, cluster=None, cluster_kwargs={}, n_workers=None, n_workers_processing=None, buffer=1.1, verbose=True, **kwargs):
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

    logging.getLogger("distributed").setLevel(logging.ERROR)

    def load(ids, data, axis, axis_slicing):
        # print(f"loading {i} {get_worker().name}")
        # time.sleep(1)
        d = {j:i for i,j in zip(ids, axis_slicing)}
        index = make_index(axis, **d)
        data = data[index]
        return data
    
    def img_reshape(img, new_axis, axis_slicing):
        # print(f"Reshaping {t} {get_worker().name}")
        # time.sleep(1)
        d = {i:np.newaxis for i in axis_slicing}
        index = make_index(new_axis, **d)
        img = img[index]
        return img

    def save(ids, data, axis, axis_slicing, zarr_array):
        # print(f"Saving {t} {get_worker().name}")
        # time.sleep(10)
        d = {j:slice(i,i+1,None) for i,j in zip(ids, axis_slicing)}
        index = make_index(axis, **d)
        zarr_array[index] = data
        return

    axis, scale = _get_axis_scale(dataset, axis, scale)
    s = [j for i,j in zip(axis,dataset.shape) if i in axis_slicing]
    if new_axis is None:
        new_axis = axis
    if scale is None:
        new_scale = scale

    # Create delayed arrays for each image, applying the custom function
    try:
        dataset_dask = da.from_zarr(dataset)
    except:
        dataset_dask = da.from_array(dataset)
    dataset_dask = dataset_dask.rechunk([1 if j in axis_slicing else i for i,j in zip(dataset.shape, axis)])

    if cluster is None:
        cluster_ = LocalCluster(n_workers=1, threads_per_worker=1)
    else:
        cluster_ = cluster

    client = cluster_.get_client()

    try:
        id = (0,)*len(axis_slicing)
        with ResourceProfiler(dt=0.01) as rprof, GpuProfiler(dt=0.01) as gprof:
            image = client.submit(load, id, dataset_dask, axis, axis_slicing)#, workers=workers_load)
            modified = client.submit(function, image, id)
            modified = client.submit(img_reshape, modified, new_axis, axis_slicing)
            modified = modified.result()
    except Exception as e:
        raise e
    finally:
        client.close()
        cluster_.close()

    available_memory = psutil.virtual_memory().available / 1e9  # Convert bytes to GB
    if verbose:
        print(f"Available memory: {available_memory} GB")
    
    if cluster is None:
        n_max_workers = psutil.cpu_count(logical=True)
    else:
        n_max_workers = len(cluster_.workers)
    if verbose:
        print(f"Max number of workers: {n_max_workers}")

    if n_workers is None:
        peak_memory_gb = max([entry[1]/1e3 for entry in rprof.results])*buffer
        if verbose:
            print(f"Peak memory: {peak_memory_gb/buffer} GB {peak_memory_gb} GB (buffered) {peak_memory_gb / available_memory} % of available memory (buffered)")
        n_workers_ = max(1,int(np.floor(available_memory / peak_memory_gb)))  # Add some buffer
    else:
        n_workers_ = n_workers
    n_workers_ = min(n_workers_, n_max_workers)
    if verbose:
        print(f"Number of workers: {n_workers_}")

    if n_workers_processing is None and GPU_AVAILABLE:
        total_gpu_memory_gb = gprof.total_memory_mb / 1e3
        peak_gpu_memory_gb = gprof.peak_memory_mb / 1e3 * buffer
        if verbose:
            print(f"Peak GPU memory: {peak_gpu_memory_gb/buffer} GB {peak_gpu_memory_gb} GB (buffered) {peak_gpu_memory_gb / total_gpu_memory_gb} % of total GPU memory")
        n_workers_processing_ = max(1,int(np.floor(total_gpu_memory_gb / peak_gpu_memory_gb)))  # Add some buffer
    elif n_workers_processing is None:
        n_workers_processing_ = n_workers_
    else:
        n_workers_processing_ = n_workers_processing
    n_workers_processing_ = min(n_workers_processing_, n_workers_)
    if verbose:
        print(f"Number of workers processing: {n_workers_processing_}")

    if len(axis) != len(modified.shape) and (new_axis is None):
        raise ValueError("The shape of the modified data is different from the original data. Please specify the new axis.")
    elif new_axis is None:
        new_axis = axis
    elif len(new_axis) != len(modified.shape):
        raise ValueError(f"The new_axis must have the same length as the modified data shape. Provided axis: {new_axis}, new_axis: {modified.shape} of length {len(modified.shape)}.")
    
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
    new_shape = tuple([modified.shape[new_axis.index(i)] if i not in axis_slicing else dataset.shape[axis.index(i)] for i in new_axis])

    if out is None:
        tmpdir = tempfile.mkdtemp()
        store = os.path.join(tmpdir, "temp.zarr")
        data = zarr.create_array(store, shape=new_shape, dtype=modified.dtype, chunks=modified.shape)
    else:
        data = zarr.create_array(out, shape=new_shape, dtype=modified.dtype, chunks=modified.shape)
        # data = zarr.open_array(out, mode="r+")

    if cluster is None:
        cluster_ = LocalCluster(n_workers=n_workers_, threads_per_worker=1)

    client = cluster_.get_client()

    # workers_load = [cluster_.workers[i].worker_address for i in range(min(n_process_images, n_workers), n_workers)]
    if isinstance(n_workers_processing_, int):
        workers_process = [cluster_.workers[i].worker_address for i in range(n_workers_processing_)]
    else:
        workers_process = [cluster_.workers[i].worker_address for i in n_workers_processing_]
        
    try:
        futures = []
        l = [i.flatten() for i in np.meshgrid(*[range(j) for j in s])]
        pbar = tqdm(total=len(l[0]), desc="Processing time frames", position=0, leave=True)
        for i in zip(*l):
            while len([i for i in futures if i.status == "pending"]) >= n_workers_:
                # print(len([i for i in futures if i.status == "pending"]))
                time.sleep(0.1)
            # print(len([i for i in futures if i.status == "pending"]))
            image = client.submit(load, i, dataset_dask, axis, axis_slicing)#, workers=workers_load)
            processed = client.submit(function, image, i, workers=workers_process)
            processed = client.submit(img_reshape, processed, new_axis, axis_slicing)
            processed_zarr = client.submit(save, i, processed, new_axis, axis_slicing, data)
            futures.append(processed_zarr)

            if len([i for i in futures if i.status == "finished"]) > pbar.n:
                pbar.update(len([i for i in futures if i.status == "finished"]) - pbar.n)

        while len([i for i in futures if i.status == "pending"]) > 0:
            time.sleep(0.1)
            if len([i for i in futures if i.status == "finished"]) > pbar.n:
                pbar.update(len([i for i in futures if i.status == "finished"]) - pbar.n)

        client.gather(futures)

    except Exception as e:
        raise e
    finally:
        client.close()
        cluster_.close()

    # Add attributes to the Zarr file (such as axis and scale)
    if out is not None:
        data = zarr.open_array(out, mode="r+")
    
    data.attrs['axis'] = new_axis
    data.attrs['scale'] = new_scale
    
    return data

def apply_function_in_time_dask(dataset, function, axis=None, scale=None, out=None, new_axis=None, new_scale=None, cluster=None, cluster_kwargs={}, n_workers=None, n_workers_processing=None, buffer=1.1, verbose=True, **kwargs):
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

    logging.getLogger("distributed").setLevel(logging.ERROR)

    axis, scale = _get_axis_scale(dataset, axis, scale)

    if "T" not in axis:
        raise ValueError("The axis must contain the time dimension 'T'.")
    if new_axis is not None:
        if "T" not in axis:
            raise ValueError("The new_axis must contain the time dimension 'T'.")
    else:
        new_axis = axis

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

    if cluster is None:
        cluster_ = LocalCluster(n_workers=1, threads_per_worker=1)
    else:
        cluster_ = cluster

    client = cluster_.get_client()

    index = make_index(axis, T=slice(None,1,None))
    with ResourceProfiler(dt=0.01) as rprof, GpuProfiler(dt=0.01) as gprof:
        data = dataset_dask[index]
        modified = data.map_blocks(
                        function,        
                        dtype=dataset_dask.dtype,   
                        chunks=slices,
                        drop_axis=drop_axis_pos,
                        new_axis=new_axis_pos,
                        **kwargs
                    )
    modified = modified.compute()

    client.gater(modified)
    client.close()

    available_memory = psutil.virtual_memory().available / 1e9  # Convert bytes to GB
    if verbose:
        print(f"Available memory: {available_memory} GB")
    
    if cluster is None:
        n_max_workers = psutil.cpu_count(logical=True)
    else:
        n_max_workers = len(cluster_.workers)
    if verbose:
        print(f"Max number of workers: {n_max_workers}")

    if n_workers is None:
        peak_memory_gb = max([entry[1]/1e3 for entry in rprof.results])*buffer
        if verbose:
            print(f"Peak memory: {peak_memory_gb/buffer} GB {peak_memory_gb} GB (buffered) {peak_memory_gb / available_memory} % of available memory (buffered)")
        n_workers_ = max(1,int(np.floor(available_memory / peak_memory_gb)))  # Add some buffer
    else:
        n_workers_ = n_workers
    n_workers_ = min(n_workers_, n_max_workers)
    if verbose:
        print(f"Number of workers: {n_workers_}")

    if n_workers_processing is None and GPU_AVAILABLE:
        total_gpu_memory_gb = gprof.total_memory_mb / 1e3
        peak_gpu_memory_gb = gprof.peak_memory_mb / 1e3 * buffer
        if verbose:
            print(f"Peak GPU memory: {peak_gpu_memory_gb/buffer} GB {peak_gpu_memory_gb} GB (buffered) {peak_gpu_memory_gb / total_gpu_memory_gb} % of total GPU memory")
        n_workers_processing_ = max(1,int(np.floor(total_gpu_memory_gb / peak_gpu_memory_gb)))  # Add some buffer
    elif n_workers_processing is None:
        n_workers_processing_ = n_workers_
    else:
        n_workers_processing_ = n_workers_processing
    n_workers_processing_ = min(n_workers_processing_, n_workers_)
    if verbose:
        print(f"Number of workers processing: {n_workers_processing_}")

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
    d = _dict_axis_shape(axis, modified.shape)
    # d = _dict_axis_shape(new_axis.replace("T",""), modified.shape)
    for i,a in enumerate(new_axis):
        if a in d.keys():
            new_shape.append(d[a])
        else:
            new_shape.append(t)
        
    if out is None:
        store = zarr.storage.MemoryStore()
        data = zarr.create_array(store, shape=new_shape, dtype=modified.dtype, chunks=tuple(slices))
    else:
        data = zarr.create_array(out, shape=new_shape, dtype=modified.dtype, chunks=tuple(slices))

    def load(i,data,axis):
        # print(f"loading {i} {get_worker().name}")
        # time.sleep(1)
        index = make_index(axis, T=slice(i,i+1,None))
        data = data[index]
        # print("Loaded ",i)
        return data

    def save(t, data, new_axis, zarr_array):
        # print(f"Saving {t} {get_worker().name}")
        # time.sleep(10)
        index = make_index(new_axis, T=slice(t,t+1,None))
        zarr_array[index] = data
        return

    if cluster is None:
        cluster_ = LocalCluster(n_workers=n_workers_, threads_per_worker=1)

    client = cluster_.get_client()

    # workers_load = [cluster_.workers[i].worker_address for i in range(min(n_process_images, n_workers), n_workers)]
    if isinstance(n_workers_processing_, int):
        workers_process = [cluster_.workers[i].worker_address for i in range(n_workers_processing_)]
    else:
        workers_process = [cluster_.workers[i].worker_address for i in n_workers_processing_]
        
    try:
        futures = []
        pbar = tqdm(total=t, desc="Processing time frames", position=0, leave=True)
        for i in range(t):
            while len([i for i in futures if i.status == "pending"]) >= n_workers_:
                # print(len([i for i in futures if i.status == "pending"]))
                time.sleep(0.1)
            # print(len([i for i in futures if i.status == "pending"]))
            image = client.submit(load, i, dataset_dask, axis)#, workers=workers_load)
            processed = client.submit(function, image, i, workers=workers_process)
            processed_zarr = client.submit(save, i, processed, new_axis, data)
            futures.append(processed_zarr)

            if len([i for i in futures if i.status == "finished"]) > pbar.n:
                pbar.update(len([i for i in futures if i.status == "finished"]) - pbar.n)

        while len([i for i in futures if i.status == "pending"]) > 0:
            time.sleep(0.1)
            if len([i for i in futures if i.status == "finished"]) > pbar.n:
                pbar.update(len([i for i in futures if i.status == "finished"]) - pbar.n)

    except Exception as e:
        raise e
    finally:
        client.gather(futures)
        client.close()

    # Add attributes to the Zarr file (such as axis and scale)
    if out is not None:
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

def downsample(dataset, factor, axis=None, scale=None, out=None, style="zoom", order=1, verbose=False, **kwargs):
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

    zoom_factor = [factor["XYZ".index(f)] for i, f in enumerate(axis) if f in "XYZ"]

    axis_slicing = [i for i in axis if i not in "XYZ"]

    def zoom(data, zoom=factor, order=1):
        return ndix.zoom(data, zoom_factor, order=order)

    return apply_function(
                    dataset, 
                    zoom, 
                    axis=axis,
                    axis_slicing=axis_slicing,
                    scale=scale, 
                    out=out, 
                    new_axis=axis, 
                    new_scale=new_scale, 
                    order=order,
                    verbose=verbose,
                    **kwargs
                    )

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
