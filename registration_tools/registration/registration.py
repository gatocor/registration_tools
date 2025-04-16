import os
import re
import numpy as np
from skimage.io import imread, imsave
import skimage.measure as skm
import scipy.ndimage as ndi
import shutil
import vt
from ..dataset.dataset import Dataset
import json
from tqdm import tqdm
import time
import warnings  # Add this import
from copy import deepcopy
import zarr
from ..utils.auxiliar import _get_axis_scale, _make_index, make_index, _dict_axis_shape, _suppress_stdout_stderr, _shape_downsampled, _shape_padded, _trnsf_padded, _image_padded
from ..utils.utils import apply_function
import tempfile
import time
from copy import deepcopy
import threading
import queue
import dask.array as da
# from dask.diagnostics import ProgressBar
from dask.callbacks import Callback
from dask.distributed import Client, Lock, LocalCluster

class ProgressBar(Callback):
    def __init__(self, total_tasks):
        self.total_tasks = total_tasks  # Pass the total number of blocks as a parameter

    def _start_state(self, dsk, state):
        # Initialize tqdm with the number of blocks (tasks) to be processed
        self._tqdm = tqdm(total=self.total_tasks, desc="Dask Progress")

    def _posttask(self, key, result, dsk, state, worker_id):
        # Update the progress bar after each task completes
        self._tqdm.update(1)

    def _finish(self, dsk, state, errored):
        # Close the progress bar when the computation finishes
        self._tqdm.close()

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cndi
    import cucim.skimage.measure as cskm
    GPU_AVAILABLE = True
    print("GPU_AVAILABLE!, Steps that can be accelerated with CUDA will be passed to the GPU.")
except ImportError:
    GPU_AVAILABLE = False

import napari
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QVBoxLayout, QWidget, QSlider, QLabel, QPushButton, QLineEdit, QHBoxLayout, QGridLayout
from qtpy.QtCore import Qt
from napari.utils.events import Event
from vispy.util.keys import ALT, CONTROL

class Registration:
    """
    A class to perform image registration.

    Parameters:
        out (str, optional): Output path to save the registration results. Default is None.
        registration_type (str, optional): Type of registration to perform. Default is "rigid".
        perform_global_trnsf (bool, optional): Whether to perform global transformation. Default is None.
        pyramid_lowest_level (int, optional): Lowest level of the pyramid for multi-resolution registration. Default is 0.
        pyramid_highest_level (int, optional): Highest level of the pyramid for multi-resolution registration. Default is 3.
        registration_direction (str, optional): Direction of registration, either "forward" or "backward". Default is "backward".
        args_registration (str, optional): Additional arguments for the registration process. Default is an empty string.

    Attributes:
        registration_type (str): The type of registration to perform.
        perform_global_trnsf (bool): Whether to perform global transformation.
        pyramid_lowest_level (int): The lowest level of the pyramid.
        pyramid_highest_level (int): The highest level of the pyramid.
        registration_direction (str): The direction of registration.
        args_registration (str): Additional arguments for registration.
        _n_spatial (None): Placeholder for number of spatial dimensions.
        _fitted (bool): Whether the registration object has been fitted.
        _box (None): Placeholder for the bounding box.
        _t_max (None): Placeholder for the maximum time point.
        _origin (None): Placeholder for the origin.
        _transfs (dict): Dictionary to store transformations.
    Methods:
        save(out=None, overwrite=False):
        load(out):
        _create_folder():
            Creates the necessary folders for saving the registration object.
        _save_transformation_relative(trnsf, pos_float, pos_ref):
            Saves the relative transformation.
        _save_transformation_global(trnsf, pos_float, pos_ref):
            Saves the global transformation.
        _load_transformation_relative(pos_float, pos_ref):
            Loads the relative transformation.
        _load_transformation_global(pos_float, pos_ref):
            Loads the global transformation.
        _trnsf_exists_relative(pos_float, pos_ref):
            Checks if the relative transformation exists.
        _trnsf_exists_global(pos_float, pos_ref):
            Checks if the global transformation exists.
        fit(dataset, use_channel=None, axis=None, scale=None, downsample=None, save_behavior="Continue", verbose=False):
        apply(dataset, out=None, axis=None, scale=None, downsample=None, save_behavior="Continue", transformation="global", verbose=False, **kwargs):
            Applies the registration to a dataset and saves the results to the specified path.
        fit_apply(dataset, out=None, use_channel=None, axis=None, scale=None, downsample=None, save_behavior="Continue", transformation="global", verbose=False):
            Fits and applies the registration to a dataset and saves the results to the specified path.
        vectorfield(mask=None, axis=None, scale=None, out=None, n_points=20, transformation="relative", **kwargs):
            Computes the vector field of the registration.
        trajectories(mask=None, axis=None, scale=None, out=None, n_points=20, transformation="relative", **kwargs):
            Computes the trajectories of the registration.
    """

    def __init__(self):
        """
        Initialize the registration object with the specified parameters.
        """

        self._perform_global_trnsf = None
        self._registration_direction = None
        self._registration_type = None
        self._padding_box = None
        self._out = None
        self._n_spatial = None
        self._fitted = False
        self._box = None
        self._t_max = None
        self._origin = None
        self._stepping = 1
        self._transfs = {}
        self._failed = []
        
    def save(self, out=None, overwrite=False):
        """
        Saves the registration object to the specified path.

        Args:
            out (str): The path to save the registration object.
        """
        self._create_folder()

        if self._out is not None and out is None:
            metadata = deepcopy(self.__dict__)
            del metadata["_out"]
            del metadata["_transfs"]
            metadata_path = os.path.join(self._out, "parameters.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
        elif self._out is not None and out is not None:
            raise ValueError("The save path is already defined. Please specify a new path or set out to None.")
        elif self._out is None and out is None:
            raise ValueError("The save path must be specified.")
        else:
            self._out = out

        metadata = deepcopy(self.__dict__)
        del metadata["_out"]
        del metadata["_transfs"]
        metadata_path = os.path.join(self._out, "parameters.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        for name, trnsf in self._transfs.items():
            if "global" in name:
                self.write_trnsf(trnsf, f"{self._out}/trnsf_global/{name}.trnsf")
            elif "relative" in name:
                self.write_trnsf(trnsf, f"{self._out}/trnsf/{name}.trnsf")

    def load(self, out):
        """
        Loads a registration object from a save path.

        Args:
            out (str): The path to the registration object.

        Returns:
            Registration: The registration object.
        """

        if not os.path.exists(f"{out}/parameters.json"):
            raise ValueError("The registration object does not exist.")

        with open(f"{out}/parameters.json", "r") as f:
            parameters = json.load(f)

        for i,j in parameters.items():
            setattr(self, i, j)
        self._out = out

        return

    def _get_image(self, dataset, pos, axis, use_channel, downsample):
        return dataset[_make_index(pos, axis, use_channel, downsample)]

    def _create_folder(self):

        os.makedirs(f"{self._out}", exist_ok=True)
        os.makedirs(f"{self._out}/trnsf_relative", exist_ok=True)
        if self._perform_global_trnsf:
            os.makedirs(f"{self._out}/trnsf_global", exist_ok=True)

    def _save_transformation_relative(self, trnsf, pos_float, pos_ref):
        if self._out is None:
            self._transfs[f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf"] = trnsf
        else:
            self.write_trnsf(trnsf, f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf")

    def _save_transformation_global(self, trnsf, pos_float, pos_ref):
        if self._out is None:
            self._transfs[f"trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf"] = trnsf
        else:
            self.write_trnsf(trnsf, f"{self._out}/trnsf_global/trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf")

    def _load_transformation_relative(self, pos_float, pos_ref):
        if self._out is None:
            if f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf" in self._transfs:
                return self._transfs[f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf"]
            elif f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf.npy" in self._transfs:
                return self._transfs[f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf"]
            else:
                raise ValueError(f"The transformation between {pos_float:04d} and {pos_ref:04d} the specified positions does not exist.")
        else:
            if os.path.exists(f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf"):
                return self.read_trnsf(f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf")
            elif os.path.exists(f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf.npy"):
                return self.read_trnsf(f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf")
            else:
                raise ValueError("The transformation between the specified positions does not exist.")

    def _load_transformation_global(self, pos_float, pos_ref):
        if self._out is None:
            if f"trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf" not in self._transfs:
                raise ValueError(f"The transformation between {pos_float:04d} and {pos_ref:04d} the specified positions does not exist.")
            else:
                return self._transfs[f"trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf"]
        else:
            if not os.path.exists(f"{self._out}/trnsf_global/trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf"):
                raise ValueError(f"The transformation between {pos_float:04d} and {pos_ref:04d} the specified positions does not exist.")
            else:
                return self.read_trnsf(f"{self._out}/trnsf_global/trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf")

    def _trnsf_exists_relative(self, pos_float, pos_ref):
        if self._out is None:
            return f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf" in self._transfs.keys() or f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf.npy" in self._transfs.keys()   
        else:
            return os.path.exists(f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf") or os.path.exists(f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf.npy")
    
    def _trnsf_exists_global(self, pos_float, pos_ref):
        if self._out is None:
            return f"trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf" in self._transfs.keys() or f"trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf.npy" in self._transfs.keys()
        else:
            return os.path.exists(f"{self._out}/trnsf_global/trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf") or os.path.exists(f"{self._out}/trnsf_global/trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf.npy")
        
    def _padding_box_to_points(self, padding_reference, scale):
        return np.array(np.meshgrid(*[np.array(i) for i in padding_reference])).T.reshape(-1, self._n_spatial) * np.array(scale)

    def _points_to_padding_box(self, points, scale):
        points_scaled = (points / np.array(scale)).astype(int)
        box = []
        for i in range(points_scaled.shape[1]):
            box.append((points_scaled[:, i].min(), points_scaled[:, i].max()))
        return box

    def _preload_images(self, dataset, positions, axis, scale, use_channel, downsample, buffer_queue, stop_event):
        """Background thread function to preload images into a queue."""
        for pos in positions:
            if stop_event.is_set():
                break  # Stop preloading if main thread signals exit

            image = self.get_image(dataset, pos, scale, axis, use_channel, downsample)
            buffer_queue.put((pos, image))  # Store preloaded image

    def _preload_images_in_thread(self, id, events, dataset, axis, scale, use_channel, downsample, task_queue, buffer_queue, num_events, assignement):
        """Background thread function to preload images into a queue."""
        while not task_queue.empty():
            ord, pos = task_queue.get()

            # print(f"Worker loading pos {id} {pos}")
            assignement[ord] = (id, False)
            image = self._get_image(dataset, pos, axis, use_channel, downsample)
            # time.sleep(np.random.randint(1, 3))
            # print(f"Worker finishing pos {id} {pos}")

            while ord-1 not in assignement.keys() and ord > 0:
                time.sleep(.1)

            if ord > 0:
                while not assignement[ord-1][1]:
                    time.sleep(.1)

            # print(f"Worker putting pos {id} {pos}")
            buffer_queue.put((ord, image))  # Store preloaded image
            assignement[ord] = (id, True)

            # while ord-1 not in assignement.keys() and ord > 0:
            #     time.sleep(1)
            
            # if ord > 0: 
            #     if not assignement[ord-1][1]:
            #         print(f"{id} waiting for {assignement[ord-1][0]}")
            #         events[id].wait()  # Wait for previous event to be completed

            # # Process the task
            # # print(f"Worker processing pos {pos}")
            # print(f"Putting {ord}")
            # buffer_queue.put((ord, image))  # Store preloaded image
            # assignement[ord] = (id, True)

            # # Signal the next task in sequence
            # while ord+1 not in assignement.keys(): 
            #     time.sleep(1)

            # print(f"Setting {assignement[ord+1]} {pos}")
            # events[assignement[ord+1][0]].set()
            # task_queue.task_done()
        
    def apply_trnsf(self, image, trnsf, scale, padding):

        nx = cp if GPU_AVAILABLE else np 
        ndix = cndi if GPU_AVAILABLE else ndi

        # Extract rotation and translation
        trnsf = nx.array(trnsf)

        # Scaling matrices (fixing order of operations)
        scale_aff = nx.eye(image.ndim+1)
        scale_aff[:-1,:-1] = nx.diag(scale)
        scale_inv_aff = nx.eye(image.ndim+1)
        scale_inv_aff[:-1,:-1] = nx.diag(1/nx.array(scale))

        image_padded = _image_padded(image, padding)
        padding_trnsf = _trnsf_padded(padding, self._n_spatial)
        padding_trnsf_inv = np.eye(image.ndim+1)
        padding_trnsf_inv[:-1, -1] = -padding_trnsf[:-1, -1]
        padding_trnsf = nx.array(padding_trnsf)
        padding_trnsf_inv = nx.array(padding_trnsf_inv)
        image_ = nx.array(image_padded)

        trnsf_ = padding_trnsf @ scale_inv_aff @ trnsf @ scale_aff @ padding_trnsf_inv

        # Apply the affine transformation to align img1 to img2
        rotation_matrix = nx.array(trnsf_[:-1, :-1])
        translation = nx.array(trnsf_[:-1, -1])

        transformed_img = ndix.affine_transform(image_, rotation_matrix, offset=translation)

        return transformed_img.get() if GPU_AVAILABLE else transformed_img

    def write_trnsf(self, trnsf, out):
        if trnsf.ndim == 2:
            np.savetxt(out, trnsf, fmt='%.6f')
        else:
            np.save(out, trnsf)

    def read_trnsf(self, out):
        if self._registration_type == "vectorfield":
            try:
                return np.load(out+".npy", allow_pickle=True)
            except:
                return np.zeros((1,2,3))
        else:
            return np.loadtxt(out)

    def image2array(self, img):
        return img
    
    def compose_trnsf(self, trnsfs):
        return np.linalg.multi_dot(trnsfs)

    def fit(self, dataset, out=None, direction="backward", perform_global_trnsf=False, use_channel=None, axis=None, scale=None, downsample=None, stepping=1, save_behavior="Continue", verbose=False):
        """
        Registers a dataset and saves the results to the specified path.
        """

        self._out = out
        self._registration_direction = direction
        self._perform_global_trnsf = perform_global_trnsf
        if self._out is not None:
            self.save()
        if direction not in ["backward", "forward"]:
            raise ValueError("The direction must be either 'backward' or 'forward'.")

        save_behaviors = ["NotOverwrite", "Continue"]

        # Check inputs
        axis, scale = _get_axis_scale(dataset, axis, scale)
        
        if len(axis) != len(dataset.shape):
            raise ValueError("The axis must have the same length as the dataset shape.")
        
        if "T" not in axis:
            raise ValueError("The axis must contain the time dimension 'T'.")
        
        use_channel = use_channel
        if "C" in axis and use_channel is None:
            raise ValueError("The axis contains the channel dimension 'C'. The channel to use must be specified.")
        elif "C" in axis and dataset.shape[np.where([i == "C" for i in axis])[0][0]] < use_channel:
            raise ValueError("The specified channel does not exist in the dataset.")
                
        if save_behavior not in save_behaviors:
            raise ValueError(f"save_behavior should be one of the following: {save_behaviors}.")

        # Check downsample
        self._n_spatial = len([i for i in axis if i in "XYZ"])
        if downsample is None:
            downsample = (1,) * self._n_spatial
        else:
            downsample = downsample

        if not all(isinstance(d, int) for d in downsample):
            raise ValueError("All elements in downsample must be integers.")
        elif len(downsample) != self._n_spatial:
            raise ValueError(f"downsample must have the same length as the number of spatial dimensions ({self._n_spatial}).")

        # Scale
        new_scale = tuple([i * j for i, j in zip(scale, downsample)])

        # Compute box
        self._box = tuple([int(i) for i in np.array(dataset.shape)[[axis.index(ax) for ax in axis if ax in "XYZ"]] * np.array(scale)])

        # Compute padding box
        padding_reference = [(0,j) for i,j in _dict_axis_shape(axis, dataset.shape).items() if i in "XYZ"][::-1]
        self._padding_box = [(0,j) for i,j in _dict_axis_shape(axis, dataset.shape).items() if i in "XYZ"][::-1]

        # Stepping
        self._stepping = stepping

        # Set registration direction
        self._t_max = _dict_axis_shape(axis, dataset.shape)["T"]
        if self._registration_direction == "backward":
            origin = 0
            iterations = []
            iterations_next = []
            for i in range(stepping):
                if i != origin:
                    iterations.append(origin)
                    iterations_next.append(i)
                for j in range(i, self._t_max, stepping):
                    if self._t_max <= stepping+j:
                        break
                    iterations.append(j)
                    iterations_next.append(j+stepping)

            # print([(i,j) for i,j in zip(iterations, iterations_next)])
            iterator = zip(
                iterations,
                iterations_next
            )
        else:
            origin = self._t_max - 1
            iterations = []
            iterations_next = []
            for i in range(self._t_max-1, self._t_max-stepping-1, -1):
                if i != origin:
                    iterations.append(origin)
                    iterations_next.append(i)
                for j in range(i, 0, -stepping):
                    if j-stepping < 0:
                        break
                    iterations.append(j)
                    iterations_next.append(j-stepping)

            # print([(i,j) for i,j in zip(iterations, iterations_next)])
            iterator = zip(
                iterations,
                iterations_next
            )

        self._origin = origin

        # Save initial transformation
        self._save_transformation_relative(np.eye(self._n_spatial+1,self._n_spatial+1), origin, origin)
        if self._perform_global_trnsf:
            self._save_transformation_global(np.eye(self._n_spatial+1,self._n_spatial+1), origin, origin)

        if self._out is not None:
            self.save()

        img_ref = None
        trnsf_global = None
        pos_ref_current = None
        for pos_ref, pos_float in tqdm(iterator, desc=f"Registering images using channel {use_channel}", unit="", total=self._t_max-1):

            if self._trnsf_exists_relative(pos_float, pos_ref):
                if save_behavior == "NotOverwrite":
                    raise ValueError("The relative transformation already exists. If you want to overwrite it, set save_behavior='Overwrite'.")
            else:
                if img_ref is None or pos_ref_current != pos_ref:
                    img_ref = self._get_image(dataset, pos_ref, axis, use_channel, downsample)
                else:
                    img_ref = img_float

                img_float = self._get_image(dataset, pos_float, axis, use_channel, downsample)

                trnsf = self.register(img_float, img_ref, scale, verbose=verbose)
                # if failed:
                #     self._failed.append(f"{pos_float:04d}_{pos_ref:04d}")

                self._save_transformation_relative(trnsf, pos_float, pos_ref)

            if self._perform_global_trnsf:

                if self._trnsf_exists_global(pos_float, self._origin):
                    if save_behavior == "NotOverwrite":
                        raise ValueError("The global transformation already exists. If you want to overwrite it, set save_behavior='Overwrite'.")
                    elif save_behavior == "Continue":
                        None
                else:
                    if trnsf_global is None and pos_ref != self._origin:
                        trnsf_global = self._load_transformation_global(pos_ref, self._origin)
                    elif pos_ref == self._origin:
                        trnsf_global = trnsf
                        self._save_transformation_global(trnsf_global, pos_float, self._origin)
                    else:
                        trnsf_global = self.compose_trnsf([
                            trnsf_global,
                            trnsf
                        ])
                        self._save_transformation_global(trnsf_global, pos_float, self._origin)

                # padding_reference = [(0,j) for i,j in _dict_axis_shape(axis, dataset.shape).items() if i in "XYZ"][::-1]
                # padding_box = self._points_to_padding_box(self.apply_trnsf_to_points(self._padding_box_to_points(padding_reference, scale[::-1]), trnsf_global), scale)
                # self._padding_box = [(int(min(i[0], j[0])), int(max(i[1], j[1]))) for i, j in zip(self._padding_box, padding_box)]

            if self._out is not None:
                self._padding_box = self._padding_box[::-1]
                self.save()
                self._padding_box = self._padding_box[::-1]

            pos_ref_current = pos_float
        
        self._padding_box = self._padding_box[::-1]
        self._fitted = True
        if self._out is not None:
            self.save()

    # def apply_old(self, dataset, out=None, axis=None, scale=None, downsample=None, save_behavior="Continue", transformation="global", padding=False, verbose=False, **kwargs):
    #     """
    #     Registers a dataset and saves the results to the specified path.
    #     """

    #     save_behaviors = ["NotOverwrite", "Overwrite", "Continue"]

    #     if not self._fitted:
    #         raise ValueError("The registration object has not been fitted. Please fit the registration object before applying it.")

    #     # Check inputs
    #     axis, scale = _get_axis_scale(dataset, axis, scale)
        
    #     if len(axis) != len(dataset.shape):
    #         raise ValueError("The axis must have the same length as the dataset shape.")
        
    #     if "T" not in axis:
    #         raise ValueError("The axis must contain the time dimension 'T'.")
                        
    #     if save_behavior not in save_behaviors:
    #         raise ValueError(f"save_behavior should be one of the following: {save_behaviors}.")

    #     # Check downsample
    #     self._n_spatial = len([i for i in axis if i in "XYZ"])
    #     if downsample is None:
    #         downsample = (1,) * self._n_spatial
    #     else:
    #         downsample = downsample

    #     if not all(isinstance(d, int) for d in downsample):
    #         raise ValueError("All elements in downsample must be integers.")
    #     elif len(downsample) != self._n_spatial:
    #         raise ValueError(f"downsample must have the same length as the number of spatial dimensions ({self._n_spatial}).")

    #     # Scale
    #     new_scale = tuple([i * j for i, j in zip(scale, downsample)])

    #     # Shape
    #     if padding:
    #         padded_shape = _shape_padded(dataset.shape, axis, self._padding_box)
    #     else:
    #         padded_shape = dataset.shape
    #     new_shape = _shape_downsampled(padded_shape, axis, downsample)
        
    #     # Setup output
    #     if out is None:
    #         store = zarr.storage.MemoryStore()
    #         data = zarr.create_array(
    #             store=store,
    #             shape=new_shape,
    #             dtype=dataset.dtype,
    #             **kwargs
    #         )
    #         data.attrs["axis"] = axis
    #         data.attrs["scale"] = new_scale
    #         data.attrs["computed"] = []
    #     elif isinstance(out, str):# and out.endswith(".zarr"):
    #         if os.path.exists(out) and save_behavior == "NotOverwrite":
    #             raise ValueError("The output file already exists. If you want to overwrite it, set save_behavior='Overwrite'.")
    #         elif os.path.exists(out) and save_behavior == "Continue":
    #             out_path, out_file = os.path.split(out)
    #             data = zarr.open_array(out_file, path=out_path)
    #             if "computed" not in data.attrs:
    #                 data.attrs["computed"] = []
    #         elif os.path.exists(out) and save_behavior == "Overwrite":
    #             shutil.rmtree(out)
    #             data = zarr.create_array(
    #                 store=out,
    #                 shape=new_shape,
    #                 dtype=dataset.dtype,
    #                 **kwargs
    #             )
    #             data.attrs["axis"] = axis
    #             data.attrs["scale"] = new_scale
    #             data.attrs["computed"] = []
    #         else:
    #             data = zarr.create_array(
    #                 store=out,
    #                 shape=new_shape,
    #                 dtype=dataset.dtype,
    #                 **kwargs
    #             )
    #             data.attrs["axis"] = axis
    #             data.attrs["scale"] = new_scale
    #             data.attrs["computed"] = []
    #     else:
    #         raise ValueError("The output must be None or a the name to a zarr file.")

    #     mt = np.eye(4)
    #     # if padding:
    #     #     padding_drift = [i[0]*j for i,j in zip(self._padding_box, scale)][::-1]
    #     #     mt[:3,3] = np.array(padding_drift)
    #     #     padding_trnsf = vt.vtTransformation(mt)

    #     if self._registration_direction == "backward":
    #         origin = 0
    #         iterator = range(1, new_shape[np.where([i == "T" for i in axis])[0][0]])
    #     else:
    #         origin = new_shape[np.where([i == "T" for i in axis])[0][0]] - 1
    #         iterator = range(new_shape[np.where([i == "T" for i in axis])[0][0]] - 2, -1, -1)

    #     if "C" in axis:
    #         for ch in range(new_shape[np.where([i == "C" for i in axis])[0][0]]):
    #             img = self._get_image(dataset, origin, axis, ch, downsample)
    #             if padding:
    #                 im = self.apply_trnsf(img, padding_trnsf, new_scale)
    #             else:
    #                 im = img
    #             data[_make_index(origin, axis, ch)] = self.image2array(im)
    #     else:
    #         data[_make_index(origin, axis, None)] = dataset[_make_index(dataset.shape[np.where([i == "T" for i in axis])[0][0]] - 1, axis, None, downsample)]

    #     for t in tqdm(iterator, desc=f"Applying registration to images", unit="", total=self._t_max-1):
    #         #Skip is computed
    #         if t in data.attrs["computed"]:
    #                 continue

    #         if transformation == "global":
    #             if not self._trnsf_exists_global(t, origin):
    #                 raise ValueError("The global transformation does not exist.")
    #             trnsf = self._load_transformation_global(t, origin)
    #         else:
    #             if not self._trnsf_exists_relative(t, origin):
    #                 raise ValueError("The relative transformation does not exist.")
    #             trnsf = self._load_transformation_relative(t, origin)

    #         if "C" in axis:
    #             for ch in range(new_shape[np.where([i == "C" for i in axis])[0][0]]):
    #                 img = self._get_image(dataset, t, axis, ch, downsample)
    #                 if padding:
    #                     joint_trnsf = self.compose_trnsf([trnsf,padding_trnsf])
    #                 else:
    #                     joint_trnsf = trnsf
    #                 im_trnsf = self.apply_trnsf(img, joint_trnsf, new_scale)
    #                 data[_make_index(t, axis, ch)] = self.image2array(im_trnsf)
    #         else:
    #             img = self._get_image(dataset, t, axis, None, downsample)
    #             if padding:
    #                 joint_trnsf = self.compose_trnsf([trnsf,padding_trnsf])
    #             else:
    #                 joint_trnsf = trnsf
    #             im = self.apply_trnsf(img, joint_trnsf, new_scale)
    #             data[_make_index(t, axis, None)] = self.image2array(im)

    #         data.attrs["computed"].append(t)
        
    #     return data

    # def apply_old(self, dataset, out=None, axis=None, scale=None, downsample=None, save_behavior="Continue", transformation="global", padding=False, verbose=False, num_loading_threads=1, num_preloaded_images=1, **kwargs):
    #     """
    #     Registers a dataset and saves the results to the specified path.
    #     """

    #     save_behaviors = ["NotOverwrite", "Overwrite", "Continue"]

    #     if not self._fitted:
    #         raise ValueError("The registration object has not been fitted. Please fit the registration object before applying it.")

    #     # Check inputs
    #     axis, scale = _get_axis_scale(dataset, axis, scale)
        
    #     if len(axis) != len(dataset.shape):
    #         raise ValueError("The axis must have the same length as the dataset shape.")
        
    #     if "T" not in axis:
    #         raise ValueError("The axis must contain the time dimension 'T'.")
                        
    #     if save_behavior not in save_behaviors:
    #         raise ValueError(f"save_behavior should be one of the following: {save_behaviors}.")

    #     # Check downsample
    #     self._n_spatial = len([i for i in axis if i in "XYZ"])
    #     if downsample is None:
    #         downsample = (1,) * self._n_spatial
    #     else:
    #         downsample = downsample

    #     if not all(isinstance(d, int) for d in downsample):
    #         raise ValueError("All elements in downsample must be integers.")
    #     elif len(downsample) != self._n_spatial:
    #         raise ValueError(f"downsample must have the same length as the number of spatial dimensions ({self._n_spatial}).")

    #     # Scale
    #     new_scale = tuple([i * j for i, j in zip(scale, downsample)])

    #     # Shape
    #     if padding:
    #         padded_shape = _shape_padded(dataset.shape, axis, self._padding_box)
    #     else:
    #         padded_shape = dataset.shape
    #     new_shape = _shape_downsampled(padded_shape, axis, downsample)
        
    #     # Setup output
    #     if out is None:
    #         store = zarr.storage.MemoryStore()
    #         data = zarr.create_array(
    #             store=store,
    #             shape=new_shape,
    #             dtype=dataset.dtype,
    #             **kwargs
    #         )
    #         data.attrs["axis"] = axis
    #         data.attrs["scale"] = new_scale
    #         data.attrs["computed"] = []
    #     elif isinstance(out, str):# and out.endswith(".zarr"):
    #         if os.path.exists(out) and save_behavior == "NotOverwrite":
    #             raise ValueError("The output file already exists. If you want to overwrite it, set save_behavior='Overwrite'.")
    #         elif os.path.exists(out) and save_behavior == "Continue":
    #             out_path, out_file = os.path.split(out)
    #             data = zarr.open_array(out_file, path=out_path)
    #             if "computed" not in data.attrs:
    #                 data.attrs["computed"] = []
    #         elif os.path.exists(out) and save_behavior == "Overwrite":
    #             shutil.rmtree(out)
    #             data = zarr.create_array(
    #                 store=out,
    #                 shape=new_shape,
    #                 dtype=dataset.dtype,
    #                 **kwargs
    #             )
    #             data.attrs["axis"] = axis
    #             data.attrs["scale"] = new_scale
    #             data.attrs["computed"] = []
    #         else:
    #             data = zarr.create_array(
    #                 store=out,
    #                 shape=new_shape,
    #                 dtype=dataset.dtype,
    #                 **kwargs
    #             )
    #             data.attrs["axis"] = axis
    #             data.attrs["scale"] = new_scale
    #             data.attrs["computed"] = []
    #     else:
    #         raise ValueError("The output must be None or a the name to a zarr file.")

    #     mt = np.eye(4)
    #     # if padding:
    #     #     padding_drift = [i[0]*j for i,j in zip(self._padding_box, scale)][::-1]
    #     #     mt[:3,3] = np.array(padding_drift)
    #     #     padding_trnsf = vt.vtTransformation(mt)

    #     if self._registration_direction == "backward":
    #         origin = 0
    #         iterator = range(0, new_shape[np.where([i == "T" for i in axis])[0][0]])
    #     else:
    #         origin = new_shape[np.where([i == "T" for i in axis])[0][0]] - 1
    #         iterator = range(new_shape[np.where([i == "T" for i in axis])[0][0]] - 1, -1, -1)

    #     # Create a preloading buffer and threading event
    #     buffer_queue = queue.Queue(maxsize=num_preloaded_images)  # Buffer with 2 preloaded images
    #     # A list of events to synchronize threads
    #     events = [threading.Event() for _ in range(num_loading_threads)]
    #     task_queue = queue.Queue()
    #     for ord, task in enumerate(deepcopy(iterator)):
    #         task_queue.put((ord, task))  # Include index to preserve task order

    #     # Launch threads
    #     # Create a ThreadPoolExecutor to manage the workers
    #     task_worker_threads = []
    #     assignement = {}
    #     for _ in range(num_loading_threads):
    #         thread = threading.Thread(target=self._preload_images_in_thread, args=(_, events, dataset, axis, scale, None, downsample, task_queue, buffer_queue, num_loading_threads, assignement))
    #         task_worker_threads.append(thread)
    #         thread.start()

    #     # # start = time.time() 
    #     # if "C" in axis:
    #     #     for ch in range(new_shape[np.where([i == "C" for i in axis])[0][0]]):
    #     #         img = self._get_image(dataset, origin, axis, ch, downsample)
    #     #         if padding:
    #     #             im = self.apply_trnsf(img, padding_trnsf, new_scale)
    #     #         else:
    #     #             im = img
    #     #         data[_make_index(origin, axis, ch)] = im
    #     # else:
    #     #     data[_make_index(origin, axis, None)] = dataset[_make_index(dataset.shape[np.where([i == "T" for i in axis])[0][0]] - 1, axis, None, downsample)]
    #     # # print(f"Time to get image: {time.time()-start}")

    #     data_ = np.zeros([i for i,j in zip(new_shape, axis) if j != "T"])
    #     for t in tqdm(iterator, desc=f"Applying registration to images", unit="", total=self._t_max-1):

    #         _, image = buffer_queue.get()

    #         # print(f"Loading {t}")
    #         # total_time = time.time() 
    #         #Skip is computed
    #         if t in data.attrs["computed"]:
    #                 continue

    #         if transformation == "global":
    #             if not self._trnsf_exists_global(t, origin):
    #                 raise ValueError(f"The global transformation between {t} and {origin} does not exist.")
    #             trnsf = self._load_transformation_global(t, origin)
    #         else:
    #             if not self._trnsf_exists_relative(t, origin):
    #                 raise ValueError(f"The relative transformation between {t} and {origin} does not exist.")
    #             trnsf = self._load_transformation_relative(t, origin)

    #         if "C" in axis:
    #             for ch in range(new_shape[np.where([i == "C" for i in axis])[0][0]]):
    #                 start = time.time()
    #                 img = image[ch]
    #                 if padding:
    #                     joint_trnsf = self.compose_trnsf([trnsf,padding_trnsf])
    #                 else:
    #                     joint_trnsf = trnsf
    #                 # print(f"Time to get image: {time.time()-start}")
    #                 # start = time.time()
    #                 im_trnsf = self.apply_trnsf(img, joint_trnsf, new_scale)
    #                 # print(f"Time to apply trnsf: {time.time()-start}")
    #                 # start = time.time()
    #                 data_[ch] = im_trnsf
    #                 # print(f"Time to save partial image: {time.time()-start}")

    #             # start = time.time()
    #             data[_make_index(t, axis, None)] = data_
    #             # print(f"Time to save image: {time.time()-start}")
    #         else:
    #             img = image
    #             if padding:
    #                 joint_trnsf = self.compose_trnsf([trnsf,padding_trnsf])
    #             else:
    #                 joint_trnsf = trnsf
    #             im = self.apply_trnsf(img, joint_trnsf, new_scale)
    #             data[_make_index(t, axis, None)] = im

    #         data.attrs["computed"].append(t)

    #         # print(f"Total time: {time.time()-total_time}")
        
    #     return data

    def apply(self, dataset, out=None, axis=None, scale=None, 
                save_behavior="Continue", transformation="global", padding=None,
                verbose=False, **kwargs):
        """
        Registers a dataset using Dask for parallel processing and saves the results in a Zarr store.
        """
        save_behaviors = ["NotOverwrite", "Overwrite", "Continue"]

        # Checks
        if not self._fitted:
            raise ValueError("The registration object has not been fitted. Please fit it before applying.")
        if save_behavior not in save_behaviors:
            raise ValueError(f"save_behavior should be one of the following: {save_behaviors}.")

        # Check inputs
        axis, scale = _get_axis_scale(dataset, axis, scale)                                
        # Check padding
        new_shape = _shape_padded(dataset.shape, axis, padding)

        # Determine time axis index
        time_idx = axis.index("T")
        origin = 0 if self._registration_direction == "backward" else new_shape[time_idx] - 1

        def process_image(image, id, self=self, axis=axis, origin=origin, transformation=transformation, scale=scale, padding=padding):
            t = id[axis.index("T")]
            if transformation == "global":
                trnsf = self._load_transformation_global(t, origin)
            else:
                trnsf = self._load_transformation_relative(t, origin)
            
            return self.apply_trnsf(image, trnsf, scale, padding)

        data = apply_function(
            dataset,
            process_image,
            axis_slicing=[i for i in axis if i not in "XYZ"],
            out=out,
            verbose=verbose,
            new_scale=scale,
        )
        data.attrs["padding"] = np.linalg.inv(_trnsf_padded(padding, self._n_spatial)).tolist()

        return data

    def fit_apply(self, 
                  dataset, out_trnsf=None, out_dataset=None, direction="backward", 
                  use_channel=None, axis=None, scale=None, downsample=None, stepping=1, 
                  perform_global_trnsf=False, save_behavior="Continue", verbose=False, padding=None, 
                  num_loading_threads=1, num_preloaded_images=1, **kwargs):
        """
        Registers a dataset and saves the results to the specified path.
        """

        self._out = out_trnsf
        self._registration_direction = direction
        self._perform_global_trnsf = perform_global_trnsf
        if self._out is not None:
            self.save()
        if direction not in ["backward", "forward"]:
            raise ValueError("The direction must be either 'backward' or 'forward'.")

        save_behaviors = ["NotOverwrite", "Overwrite", "Continue"]

        # Check inputs
        axis, scale = _get_axis_scale(dataset, axis, scale)

        if len(axis) != len(dataset.shape):
            raise ValueError("The axis must have the same length as the dataset shape.")
        
        if "T" not in axis:
            raise ValueError("The axis must contain the time dimension 'T'.")
        
        use_channel = use_channel
        if "C" in axis and use_channel is None:
            raise ValueError("The axis contains the channel dimension 'C'. The channel to use must be specified.")
        elif "C" in axis and dataset.shape[np.where([i == "C" for i in axis])[0][0]] < use_channel:
            raise ValueError("The specified channel does not exist in the dataset.")
                
        if save_behavior not in save_behaviors:
            raise ValueError(f"save_behavior should be one of the following: {save_behaviors}.")

        # Check downsample
        self._n_spatial = len([i for i in axis if i in "XYZ"])
        if downsample is None:
            downsample = (1,) * self._n_spatial
        else:
            downsample = downsample

        if not all(isinstance(d, int) for d in downsample):
            raise ValueError("All elements in downsample must be integers.")
        elif len(downsample) != self._n_spatial:
            raise ValueError(f"downsample must have the same length as the number of spatial dimensions ({self._n_spatial}).")

        # Scale
        new_scale = tuple([i * j for i, j in zip(scale, downsample)])

        # Compute box
        self._box = tuple([int(i) for i in np.array(dataset.shape)[[axis.index(ax) for ax in axis if ax in "XYZ"]] * np.array(scale)])

        # Compute padding box
        padding_reference = [(0,j) for i,j in _dict_axis_shape(axis, dataset.shape).items() if i in "XYZ"][::-1]
        self._padding_box = [(0,j) for i,j in _dict_axis_shape(axis, dataset.shape).items() if i in "XYZ"][::-1]

        # New shape
        new_shape = _shape_padded(dataset.shape, axis, padding)

        # Suppress skimage warnings for low contrast images
        warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image.*")

        # Stepping
        self._stepping = stepping

        # Set registration direction
        self._t_max = _dict_axis_shape(axis, dataset.shape)["T"]
        if self._registration_direction == "backward":
            self._origin = 0
            iterations = []
            iterations_next = []
            for i in range(stepping):
                if i != self._origin:
                    iterations.append(self._origin)
                    iterations_next.append(i)
                for j in range(i, self._t_max, stepping):
                    if self._t_max <= stepping+j:
                        break
                    iterations.append(j)
                    iterations_next.append(j+stepping)

            iterator = zip(
                iterations,
                iterations_next
            )
        else:
            self._origin = self._t_max - 1
            iterations = []
            iterations_next = []
            for i in range(self._t_max-1, self._t_max-stepping-1, -1):
                if i != self._origin:
                    iterations.append(self._origin)
                    iterations_next.append(i)
                for j in range(i, 0, -stepping):
                    if j-stepping < 0:
                        break
                    iterations.append(j)
                    iterations_next.append(j-stepping)

            # print([(i,j) for i,j in zip(iterations, iterations_next)])
            iterator = zip(
                iterations,
                iterations_next
            )

        # Setup output
        if out_dataset is None:
            store = zarr.storage.MemoryStore()
            data = zarr.create_array(
                store=store,
                shape=new_shape,
                dtype=dataset.dtype,
                **kwargs
            )
            data.attrs["axis"] = axis
            data.attrs["scale"] = new_scale
            data.attrs["computed"] = []
        elif isinstance(out_dataset, str):# and out.endswith(".zarr"):
            if os.path.exists(out_dataset) and save_behavior == "NotOverwrite":
                raise ValueError("The output file already exists. If you want to overwrite it, set save_behavior='Overwrite'.")
            elif os.path.exists(out_dataset) and save_behavior == "Continue":
                out_path, out_file = os.path.split(out_dataset)
                data = zarr.open_array(out_file, path=out_path)
                if "computed" not in data.attrs:
                    data.attrs["computed"] = []
            elif os.path.exists(out_dataset) and save_behavior == "Overwrite":
                shutil.rmtree(out_dataset)
                data = zarr.create_array(
                    store=out_dataset,
                    shape=new_shape,
                    dtype=dataset.dtype,
                    **kwargs
                )
                data.attrs["axis"] = axis
                data.attrs["scale"] = new_scale
                data.attrs["computed"] = []
            else:
                data = zarr.create_array(
                    store=out_dataset,
                    shape=new_shape,
                    dtype=dataset.dtype,
                    **kwargs
                )
                # data[:] = 0.
                data.attrs["axis"] = axis
                data.attrs["scale"] = new_scale
                data.attrs["computed"] = []
        else:
            raise ValueError("The output must be None or a the name to a zarr file.")
        
        # Save initial image
        if "C" in axis:
            for ch in range(new_shape[np.where([i == "C" for i in axis])[0][0]]):
                data[_make_index(self._origin, axis, ch)] = _image_padded(dataset[_make_index(self._origin, axis, ch, downsample=downsample)], padding)
        else:
            data[_make_index(self._origin, axis, None)] = _image_padded(dataset[_make_index(self._origin, axis, None, downsample=downsample)], padding)

        # Save parameters in the registration folder
        if self._out is not None:
            self.save()

        # Create a preloading buffer and threading event
        buffer_queue = queue.Queue(maxsize=num_preloaded_images)  # Buffer with 2 preloaded images
        # preloader_thread = threading.Thread(target=self._preload_images, 
        #                                     args=(dataset, [p[1] for p in deepcopy(iterator)], axis, scale, None, downsample, buffer_queue, stop_event))
        # preloader_thread.start()

        # Divide the positions across multiple threads

        # A list of events to synchronize threads
        events = [threading.Event() for _ in range(num_loading_threads)]

        task_queue = queue.Queue()
        for ord, task in enumerate(deepcopy(iterator)):
            task_queue.put((ord, task[1]))  # Include index to preserve task order

        # Launch threads
        # Create a ThreadPoolExecutor to manage the workers
        task_worker_threads = []
        assignement = {}
        for _ in range(num_loading_threads):
            thread = threading.Thread(target=self._preload_images_in_thread, args=(_, events, dataset, axis, scale, None, downsample, task_queue, buffer_queue, num_loading_threads, assignement))
            task_worker_threads.append(thread)
            thread.start()

        # Save initial transformation
        self._save_transformation_relative(np.eye(self._n_spatial+1,self._n_spatial+1), self._origin, self._origin)
        if self._perform_global_trnsf:
            self._save_transformation_global(np.eye(self._n_spatial+1,self._n_spatial+1), self._origin, self._origin)

        # Loop over the images for registration
        img_ref = None
        trnsf_global = None
        pos_ref_current = None
        data_ = np.zeros([i for i,j in zip(new_shape, axis) if j != "T"])
        trnsf = np.eye(self._n_spatial+1)
        for pos_ref, pos_float in tqdm(iterator, desc=f"Registering images using channel {use_channel}", unit="", total=self._t_max-1):
            # start_total = time.time()
            if self._trnsf_exists_relative(pos_float, pos_ref) and save_behavior != "Overwrite":
                if save_behavior == "NotOverwrite":
                    raise ValueError("The relative transformation already exists. If you want to overwrite it, set save_behavior='Overwrite'.")
                elif save_behavior == "Continue":
                    None
            else:
                # start = time.time()
                if img_ref is None or pos_ref_current != pos_ref:
                    img_ref = self._get_image(dataset, pos_ref, axis, use_channel, downsample)
                else:
                    img_ref = img_float
                _, img_total = buffer_queue.get()
                # img_total = self.get_image(dataset, pos_float, scale, axis, None, downsample)
                if "C" in axis:
                    img_float = img_total[_make_index(use_channel, axis.replace("T","").replace("C","T"))]
                else:
                    img_float = img_total
                # print(f"Getting images: {time.time()-start:.2f}")
                # start = time.time()
                trnsf = self.register(img_float, img_ref, new_scale, verbose=verbose)
                # print(f"Registration: {time.time()-start:.2f}")
                # start = time.time()
                self._save_transformation_relative(trnsf, pos_float, pos_ref)
                # print(f"Saving transformation: {time.time()-start:.2f}")
            # start = time.time()
            if self._perform_global_trnsf:
                if self._trnsf_exists_global(pos_float, self._origin):
                    if save_behavior == "NotOverwrite":
                        raise ValueError("The global transformation already exists. If you want to overwrite it, set save_behavior='Overwrite'.")
                    elif save_behavior == "Continue":
                        None
                else:
                    if trnsf_global is None and pos_ref != self._origin:
                        trnsf_global = self._load_transformation_global(pos_ref, self._origin)
                    elif pos_ref == self._origin:
                        trnsf_global = trnsf
                        self._save_transformation_global(trnsf_global, pos_float, self._origin)
                    else:
                        trnsf_global = self.compose_trnsf([
                            trnsf_global,
                            trnsf
                        ])
                        self._save_transformation_global(trnsf_global, pos_float, self._origin)
            # print(f"Global transformation: {time.time()-start:.2f}")
                #check data extremes
                # padding_reference = [(0,j) for i,j in _dict_axis_shape(axis, dataset.shape).items() if i in "XYZ"][::-1]
                # points_vt = vt.vtPointList(self._padding_box_to_points(padding_reference, scale[::-1]))
                # points_vt_out = vt.apply_trsf_to_points(points_vt, vt.inv_trsf(trnsf_global))
                # padding_box = self._points_to_padding_box(points_vt_out.copy_to_array(), scale[::-1])
                # self._padding_box = [(int(min(i[0], j[0])), int(max(i[1], j[1]))) for i, j in zip(self._padding_box, padding_box)]
            if self._out is not None:
                self._padding_box = self._padding_box[::-1]
                self.save()
                self._padding_box = self._padding_box[::-1]
            if pos_float in data.attrs["computed"]:
                continue
            if perform_global_trnsf:
                trnsf_img = trnsf_global
            else:
                trnsf_img = trnsf
            if "C" in axis:
                for ch in range(new_shape[np.where([i == "C" for i in axis])[0][0]]):
                    # start = time.time()
                    img_ = img_total[ch]
                    # print(f"Getting image: {time.time()-start:.2f}")
                    # start = time.time()
                    im_trnsf = self.apply_trnsf(img_, trnsf_img, new_scale)
                    # print(f"Applying transformation: {time.time()-start:.2f}")
                    # start = time.time()
                    data_[ch] = im_trnsf
                    # print(f"Saving image: {time.time()-start:.2f}")
                # start = time.time()
                data[_make_index(pos_float, axis, None)] = data_
                # print(f"Saving image: {time.time()-start:.2f}")
            else:
                im = self.apply_trnsf(img_float, trnsf_img, new_scale, padding)
                data[_make_index(pos_float, axis, None)] = im
            pos_ref_current = pos_float

            # print(f"Total time: {time.time()-start_total:.2f}")

            # Wait for all tasks to be processed
        for thread in task_worker_threads:
            thread.join()

        self._padding_box = self._padding_box[::-1]
        self._fitted = True
        if self._out is not None:
            self.save()

        if padding:
            print("Padding data")
            return self.apply(data, axis=axis, scale=scale, downsample=downsample, save_behavior="Overwrite", perform_global_trnsf=perform_global_trnsf, padding=padding, verbose=verbose, **kwargs)
        else:
            return data

    def vectorfield(self, mask=None, axis=None, scale=None, out=None, n_points=20, transformation="relative", **kwargs):

        if not self._fitted:
            raise ValueError("The registration object has not been fitted. Please fit the registration object before applying it.")

        # Check inputs
        mesh = np.meshgrid(*[np.linspace(0, i, n_points) for i in self._box])
        points = np.vstack([i.ravel() for i in mesh]).T

        if mask is not None:
            axis, scale = _get_axis_scale(mask, axis, scale)
            d = _dict_axis_shape(axis, mask.shape)
            mesh_mask = np.meshgrid(*[np.round(np.linspace(0, j-1, n_points)).astype(int) for i, j in d.items() if i in "XYZ"])
            # mesh_mask = np.concatenate([i.reshape(-1,1) for i in mesh_mask],axis=1)
            if "T" not in axis:
                keep_points = points[mask[*mesh_mask].flatten()][:,::-1]
        else:
            keep_points = points[:,::-1]

        n_spatial = len([i for i in axis if i in "XYZ"])
        if out is None:
            store = zarr.storage.MemoryStore()
            data = zarr.create_array(store=store, shape=(0,2,n_spatial+1), dtype=float, **kwargs)
        elif isinstance(out, str) and out.endswith(".zarr"):
            data = zarr.create_array(
                store=out,
                shape=(0,2,n_spatial+1),
                dtype=float,
                **kwargs
            )
        else:
            raise ValueError("The output must be None or a the name to a zarr file.")

        if self._registration_direction == "backward":
            iterator = range(1, self._t_max)
            iterator_next = range(0, self._t_max-1)
        else:
            iterator = range(self._t_max-2, -1, -1)
            iterator_next = range(self._t_max-1, 0, -1)

        for t, t_next in tqdm(zip(iterator, iterator_next), desc=f"Computing vectorfield", unit="", total=self._t_max-1):
            if mask is not None and "T" in axis:
                keep_points = points[mask[_make_index(t,axis)][*mesh_mask].flatten()][:,::-1]

            if transformation == "global":
                trnsf = self._load_transformation_global(t, self._origin)
            elif transformation == "relative":
                trnsf = self._load_transformation_relative(t, t_next)
             
            points_vt = vt.vtPointList(keep_points)
            points_vt_out = vt.apply_trsf_to_points(points_vt, trnsf)
            vectorfield = points_vt_out.copy_to_array() - keep_points
            l = vectorfield.shape[0]
            data_add = np.zeros((l,2,n_spatial+1))
            data_add[:,0,0] = t
            data_add[:,1,0] = 0
            data_add[:,0,1:] = keep_points[:,::-1]
            data_add[:,1,1:] = -vectorfield[:,::-1]

            data.append(data_add, axis=0)

        # Return data
        if out is None:
            return data

    def trajectories(self, mask=None, axis=None, scale=None, out=None, n_points=20, transformation="relative", **kwargs):

        if not self._fitted:
            raise ValueError("The registration object has not been fitted. Please fit the registration object before applying it.")

        # Check inputs
        mesh = np.meshgrid(*[np.linspace(0, i, n_points) for i in self._box])
        points = np.vstack([i.ravel() for i in mesh]).T

        if mask is not None:
            axis, scale = _get_axis_scale(mask, axis, scale)
            d = _dict_axis_shape(axis, mask.shape)
            mesh_mask = np.meshgrid(*[np.round(np.linspace(0, j-1, n_points)).astype(int) for i, j in d.items() if i in "XYZ"])
            # mesh_mask = np.concatenate([i.reshape(-1,1) for i in mesh_mask],axis=1)
            if "T" not in axis:
                keep_points = points[mask[*mesh_mask].flatten()][:,::-1]
            else:
                raise ValueError("The mask cannot contain the time dimension 'T'.")
        else:
            keep_points = points[:,::-1]

        n_spatial = len([i for i in axis if i in "XYZ"])
        if out is None:
            store = zarr.storage.MemoryStore()
            data = zarr.create_array(store=store, shape=(0,n_spatial+2), dtype=float, **kwargs)
        elif isinstance(out, str) and out.endswith(".zarr"):
            data = zarr.create_array(
                store=out,
                shape=(0,n_spatial+2),
                dtype=float,
                **kwargs
            )
        else:
            raise ValueError("The output must be None or a the name to a zarr file.")

        if self._registration_direction == "backward":
            iterator = range(self._t_max-1, 0, -1)
            iterator_next = range(self._t_max-2, -1, -1)
            t = self._t_max-1
        else:
            iterator = range(0, self._t_max-1)
            iterator_next = range(1, self._t_max)
            t = 0
                    
        l = keep_points.shape[0]
        data_add = np.zeros((l,n_spatial+2))
        data_add[:,0] = range(len(keep_points))
        data_add[:,1] = t
        data_add[:,2:] = keep_points[:,::-1]
        data.append(data_add, axis=0)

        for t, t_next in tqdm(zip(iterator, iterator_next), desc=f"Computing trajectories", unit="", total=self._t_max-1):

            if transformation == "global":
                trnsf = self._load_transformation_global(t, self._origin)
            elif transformation == "relative":
                trnsf = self._load_transformation_relative(t, t_next)
             
            points_vt = vt.vtPointList(keep_points)
            points_vt_out = vt.apply_trsf_to_points(points_vt, trnsf)
            keep_points = keep_points - (points_vt_out.copy_to_array()-keep_points)
            l = keep_points.shape[0]
            data_add = np.zeros((l,n_spatial+2))
            data_add[:,0] = range(len(keep_points))
            data_add[:,1] = t_next
            data_add[:,2:] = keep_points[:,::-1]

            data.append(data_add, axis=0)

        # Return data
        if out is None:
            return data
    
    def make_axis(self, origin, scale):
        """
        Returns the axis of the registration object.
        """

        if self._registration_direction == "backward":
            iterator = range(0, self._t_max)
            iterator2 = [0] + [i for i in range(1, self._t_max)]
        else:
            iterator = range(self._t_max-1, -1, -1)
            iterator2 = [self._t_max-1] + [i for i in range(self._t_max-2, -1, -1)]

        axis1 = np.zeros((self._t_max, 2, self._n_spatial))
        axis2 = np.zeros((self._t_max, 2, self._n_spatial))
        for i,j in zip(iterator, iterator2):
            if self._perform_global_trnsf and self._trnsf_exists_global(i, self._origin):
                transformation = self._load_transformation_global(i, self._origin)
            elif not self._perform_global_trnsf and self._trnsf_exists_relative(i, j):
                transformation = self._load_transformation_relative(i, j)

            pos = transformation @ np.array([0,0,1,0])
            pos2 = transformation @ np.array([0,1,0,0]) 
            axis1[i,0,:] = origin
            axis1[i,1,:] = pos[:3]*np.sqrt(np.sum(scale**2))/4
            axis2[i,0,:] = origin
            axis2[i,1,:] = pos2[:3]*np.sqrt(np.sum(scale**2))/4

        return axis1, axis2

def get_pyramid_levels(dataset, maximum_size = 100, verbose = True):
    """
    Returns the lowest and highest pyramid levels for the dataset.

    Args:
        dataset (Dataset): The dataset object.
        maximum_size (int, optional): The maximum size for the pyramid levels. Default is 100.
        verbose (bool, optional): If True, print the pyramid levels. Default is True.

    Returns:
        tuple: The lowest and highest pyramid levels.
    """

    shape = np.array(dataset.get_spatial_shape())
    n = int(np.ceil(np.max(np.log2(shape))))
    ll_threshold = n
    for level, n in enumerate(range(n, -1, -1)):
        if 2**n < 32:
            print(f"Level {level} is below 32 per dimension. Registration will use up to level {level-1} when computing.")
            break
        new_shape = np.minimum(2**n, shape)
        if verbose:
            print(f"Level {level}: {new_shape}")
        if np.any(2**n > maximum_size):
            ll_threshold = level

    return ll_threshold, level - 1

class RegistrationVT(Registration):
    """
    A class to perform image registration.

    Parameters:
        out (str, optional): Output path to save the registration results. Default is None.
        registration_type (str, optional): Type of registration to perform. Default is "rigid".
        perform_global_trnsf (bool, optional): Whether to perform global transformation. Default is None.
        pyramid_lowest_level (int, optional): Lowest level of the pyramid for multi-resolution registration. Default is 0.
        pyramid_highest_level (int, optional): Highest level of the pyramid for multi-resolution registration. Default is 3.
        registration_direction (str, optional): Direction of registration, either "forward" or "backward". Default is "backward".
        args_registration (str, optional): Additional arguments for the registration process. Default is an empty string.

    Attributes:
        registration_type (str): The type of registration to perform.
        perform_global_trnsf (bool): Whether to perform global transformation.
        pyramid_lowest_level (int): The lowest level of the pyramid.
        pyramid_highest_level (int): The highest level of the pyramid.
        registration_direction (str): The direction of registration.
        args_registration (str): Additional arguments for registration.
        _n_spatial (None): Placeholder for number of spatial dimensions.
        _fitted (bool): Whether the registration object has been fitted.
        _box (None): Placeholder for the bounding box.
        _t_max (None): Placeholder for the maximum time point.
        _origin (None): Placeholder for the origin.
        _transfs (dict): Dictionary to store transformations.
    Methods:
        save(out=None, overwrite=False):
        load(out):
        _create_folder():
            Creates the necessary folders for saving the registration object.
        _save_transformation_relative(trnsf, pos_float, pos_ref):
            Saves the relative transformation.
        _save_transformation_global(trnsf, pos_float, pos_ref):
            Saves the global transformation.
        _load_transformation_relative(pos_float, pos_ref):
            Loads the relative transformation.
        _load_transformation_global(pos_float, pos_ref):
            Loads the global transformation.
        _trnsf_exists_relative(pos_float, pos_ref):
            Checks if the relative transformation exists.
        _trnsf_exists_global(pos_float, pos_ref):
            Checks if the global transformation exists.
        fit(dataset, use_channel=None, axis=None, scale=None, downsample=None, save_behavior="Continue", verbose=False):
        apply(dataset, out=None, axis=None, scale=None, downsample=None, save_behavior="Continue", transformation="global", verbose=False, **kwargs):
            Applies the registration to a dataset and saves the results to the specified path.
        fit_apply(dataset, out=None, use_channel=None, axis=None, scale=None, downsample=None, save_behavior="Continue", transformation="global", verbose=False):
            Fits and applies the registration to a dataset and saves the results to the specified path.
        vectorfield(mask=None, axis=None, scale=None, out=None, n_points=20, transformation="relative", **kwargs):
            Computes the vector field of the registration.
        trajectories(mask=None, axis=None, scale=None, out=None, n_points=20, transformation="relative", **kwargs):
            Computes the trajectories of the registration.
    """

    def __init__(self, 
            registration_type="rigid", 
            pyramid_lowest_level=0, 
            pyramid_highest_level=3, 
            args_registration=""
        ):
        """
        Initialize the registration object with the specified parameters.

        Parameters:
        registration_type (str, optional): Type of registration to perform. Default is "rigid".
        pyramid_lowest_level (int, optional): Lowest level of the pyramid for multi-resolution registration. Default is 0.
        pyramid_highest_level (int, optional): Highest level of the pyramid for multi-resolution registration. Default is 3.
        args_registration (str, optional): Additional arguments for the registration process. Default is an empty string.
        """

        super().__init__()

        rigid_transformations = ["translation2D", "translation3D", "translation", "rigid2D", "rigid3D", "rigid", "rotation2D", "rotation3D", "rotation", "affine2D", "affine3D", "affine", "vectorfield"]

        if registration_type not in rigid_transformations:
            raise ValueError(f"Invalid registration type: {registration_type}. Supported types are: {rigid_transformations}")
        self._registration_type = registration_type
        self._pyramid_lowest_level = pyramid_lowest_level
        self._pyramid_highest_level = pyramid_highest_level
        self._args_registration = args_registration
    
    def _array2image(self, img, scale):
        if img.dtype not in (np.uint8, np.uint16):
            img = np.interp(img, (img.min(), img.max()), (0, np.iinfo(np.uint16).max)).astype(np.uint16)
        # if padding is not None:
        #     padding_ = [(-p[0]//d, (p[1]-s)//d) for p,s,d in zip(padding, img.shape, downsample)]
        #     # padding_ = [(0, (p[1]-s)//d-p[0]//d) for p,s,d in zip(padding, img.shape, downsample)]
        #     img = np.pad(img, padding_, mode="constant")

        # Suppress skimage warnings for low contrast images
        warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image.*")

        scale_ = list(scale)[::-1] + [1]
        scale_ = tuple(scale_[:3])

        try:
            with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as temp_file:
                imsave(temp_file.name, img)
                im = vt.vtImage(temp_file.name)
                # img = vt.vtImage(img.copy()) #Ideal but runs all the time in problems
                im.setSpacing(scale_)
        except: #In case you do not have access permissions
            temp_file = f"temp_{np.random.randint(0,1E6)}.tiff"
            imsave(temp_file, img)
            im = vt.vtImage(temp_file)
            # img = vt.vtImage(img.copy()) #Ideal but runs all the time in problems
            im.setSpacing(scale_)
            os.remove(temp_file)

        # im = vt.vtImage(img)
        # img = vt.vtImage(img.copy()) #Ideal but runs all the time in problems
        # im.setSpacing(scale_)

        # Activate warnings again
        warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image.*")

        return im

    def _image2array(self, img):
        return img.copy_to_array()

    # def write_trnsf(self, trnsf, out):
    #     trnsf.write(out)

    # def read_trnsf(self, out):
    #     return vt.vtTransformation(out)
    
    # def apply_trnsf(self, img, trnsf, scale):
    #     img = self._array2image(img, scale)
    #     return vt.apply_trsf(img, trnsf).copy_to_array()
    
    # def apply_trnsf_to_points(self, points, trnsf):
    #     points = vt.vtPointList(points)
    #     return vt.apply_trsf_to_points(points, trnsf).copy_to_array()
    
    # def compose_trnsf(self, trnsfs):
    #     return vt.compose_trsf(trnsfs)
    
    def _vectorfield2array(self, trnsf, scale, stride=5):

        points = np.meshgrid(*[np.arange(0, i, j) for i,j in zip(self._box, scale)])
        points = np.vstack([i.ravel() for i in points]).T
        points = points[:,::-1]
        points_vt = vt.vtPointList(points)
        points_vt_out = vt.apply_trsf_to_points(points_vt, trnsf)
        vectorfield = points_vt_out.copy_to_array()[:,:self._n_spatial] - points
        p,d = points.shape

        points = np.floor(points/scale[::-1])
        vectorfield /= scale[::-1]

        return np.concatenate([points[:,::-1].reshape(p,1,d), -vectorfield[:,::-1].reshape(p,1,d)], axis=1)

    def _make_registration_args(self, registration_type=None, pyramid_lowest_level=None, pyramid_highest_level=None, args_registration=None, verbose=False):
        if registration_type is None:
            registration_type = self._registration_type
        if pyramid_lowest_level is None:
            pyramid_lowest_level = self._pyramid_lowest_level
        if pyramid_highest_level is None:
            pyramid_highest_level = self._pyramid_highest_level
        if args_registration is None:
            args_registration = self._args_registration

        registration_args = ""
        if not verbose:
            registration_args = " -no-verbose"
        registration_args += f" -transformation-type {registration_type} -pyramid-lowest-level {pyramid_lowest_level} -pyramid-highest-level {pyramid_highest_level} "
        registration_args += args_registration
        registration_args = registration_args.replace("rotation", "rigid")

        return registration_args
    
    def register(self, img_float, img_ref, scale, verbose=False):

        img_float_vt = self._array2image(img_float, scale)
        img_ref_vt = self._array2image(img_ref, scale)

        keep = True
        counter = 0
        registration_args = self._make_registration_args(verbose=verbose)
        while keep and counter < 10:
            # Usage
            if verbose:
                trnsf = vt.blockmatching(img_float_vt, image_ref=img_ref_vt, params=registration_args)
            else:
                with _suppress_stdout_stderr():
                    trnsf = vt.blockmatching(img_float_vt, image_ref=img_ref_vt, params=registration_args)
            if trnsf is None:
                # print("Failed")
                trnsf = vt.vtTransformation(np.eye(4))
                failed = True
                counter += 1
                if self._pyramid_highest_level  - counter > 0:
                    if verbose:
                        print("Registration failed. Trying with a lower highest pyramid level.")
                    pyramid_highest_level = self._pyramid_highest_level - counter - 1
                    registration_args = self._make_registration_args(pyramid_highest_level=pyramid_highest_level, verbose=verbose)
                else:
                    if "-pyramid-gaussian-filtering" not in self._args_registration and "-py-gf" not in self._args_registration:
                        if verbose:
                            print("Registration failed. Trying with pyramid gaussian filtering.")
                        args_registration = self._args_registration + " -pyramid-gaussian-filtering"
                        registration_args = self._make_registration_args(args_registration=args_registration, verbose=verbose)
            else:
                registration_args = self._make_registration_args(verbose=verbose)
                keep = False

        if "rotation" in self._registration_type:
            center = vt.vtPointList([[i/2 for i in self._box[::-1]]])
            center_moved = vt.apply_trsf_to_points(center, trnsf)
            compensation = np.eye(self._n_spatial+1)
            compensation[:self._n_spatial, -1] = center.copy_to_array()[0] - center_moved.copy_to_array()[0]
            trnsf = vt.compose_trsf([vt.vtTransformation(compensation), trnsf])

        if self._registration_type == "vectorfield":
            return self._vectorfield2array(trnsf, scale)

        #Invert axis order to make it consistent with input order
        if self._n_spatial == 2:
            v = [0,1,-1]
            v_ = [1,0,-1]
        else:
            v = [0,1,2,-1]
            v_ = [2,1,0,-1]
        trnsf = trnsf.copy_to_array()
        trnsf = trnsf[:,v_]
        trnsf = trnsf[v_,:]

        return trnsf
    
#     Parameters for the blockmatching algorithm:
#         ### image geometry ### 
#         [-reference-voxel %lf %lf [%lf]]:
#         changes/sets the voxel sizes of the reference image
#         [-floating-voxel %lf %lf [%lf]]:
#         changes/sets the voxel sizes of the floating image
#         ### pre-processing ###
#         [-normalisation|-norma|-rescale] # input images are normalized on one byte
#         before matching (this may be the default behavior)
#         [-no-normalisation|-no-norma|-no-rescale] # input images are not normalized on
#         one byte before matching
#         ### post-processing ###
#         [-no-composition-with-left] # the written result transformation is only the
#         computed one, ie it is not composed with the left/initial one (thus does not allow
#         to resample the floating image if an left/initial transformation is given) [default]
#         [-composition-with-left] # the written result transformation is the
#         computed one composed with the left/initial one (thus allows to resample the
#         floating image if an left/initial transformation is given) 
#         ### pyramid building ###
#         [-pyramid-gaussian-filtering | -py-gf] # before subsampling, the images 
#         are filtered (ie smoothed) by a gaussian kernel.
#         ### block geometry (floating image) ###
#         -block-size|-bl-size %d %d %d       # size of the block along X, Y, Z
#         -block-spacing|-bl-space %d %d %d   # block spacing in the floating image
#         -block-border|-bl-border %d %d %d   # block borders: to be added twice at
#         each dimension for statistics computation
#         ### block selection ###
#         [-floating-low-threshold | -flo-lt %d]     # values <= low threshold are not
#         considered
#         [-floating-high-threshold | -flo-ht %d]    # values >= high threshold are not
#         considered
#         [-floating-removed-fraction | -flo-rf %f]  # maximal fraction of removed points
#         because of the threshold. If too many points are removed, the block is
#         discarded
#         [-reference-low-threshold | -ref-lt %d]    # values <= low threshold are not
#         considered
#         [-reference-high-threshold | -ref-ht %d]   # values >= high threshold are not
#         considered
#         [-reference-removed-fraction | -ref-rf %f] # maximal fraction of removed points
#         because of the threshold. If too many points are removed, the block is
#         discarded
#         [-floating-selection-fraction[-ll|-hl] | -flo-frac[-ll|-hl] %lf] # fraction of
#         blocks from the floating image kept at a pyramid level, the blocks being
#         sorted w.r.t their variance (see note (1) for [-ll|-hl])
#         ### pairing ###
#         [-search-neighborhood-half-size | -se-hsize %d %d %d] # half size of the search
#         neighborhood in the reference when looking for similar blocks
#         [-search-neighborhood-step | -se-step %d %d %d] # step between blocks to be
#         tested in the search neighborhood
#         [-similarity-measure | -similarity | -si [cc|ecc|ssd|sad]]  # similarity measure
#         cc: correlation coefficient
#         ecc: extended correlation coefficient
#         [-similarity-measure-threshold | -si-th %lf]    # threshold on the similarity
#         measure: pairings below that threshold are discarded
#         ### transformation regularization ###
#         [-elastic-regularization-sigma[-ll|-hl] | -elastic-sigma[-ll|-hl]  %lf %lf %lf]
#         # sigma for elastic regularization (only for vector field) (see note (1) for
#         [-ll|-hl])
#         ### transformation estimation ###
#         [-estimator-type|-estimator|-es-type %s] # transformation estimator
#         wlts: weighted least trimmed squares
#         lts: least trimmed squares
#         wls: weighted least squares
#         ls: least squares
#         [-lts-cut|-lts-fraction %lf] # for trimmed estimations, fraction of pairs that are kept
#         [-lts-deviation %lf] # for trimmed estimations, defines the threshold to discard
#         pairings, ie 'average + this_value * standard_deviation'
#         [-lts-iterations %d] # for trimmed estimations, the maximal number of iterations
#         [-fluid-sigma|-lts-sigma[-ll|-hl] %lf %lf %lf] # sigma for fluid regularization,
#         ie field interpolation and regularization for pairings (only for vector field)
#         (see note (1) for [-ll|-hl])
#         [-vector-propagation-distance|-propagation-distance|-pdistance %f] # 
#         distance propagation of initial pairings (ie displacements)
#         this implies the same displacement for the spanned sphere
#         (only for vectorfield)
#         [-vector-fading-distance|-fading-distance|-fdistance %f] # 
#         area of fading for initial pairings (ie displacements)
#         this allows progressive transition towards null displacements
#         and thus avoid discontinuites
#         ### end conditions for matching loop ###
#         [-max-iteration[-ll|-hl]|-max-iterations[-ll|-hl]|-max-iter[-ll|-hl] %d]|...
#         ...|-iterations[-ll|-hl] %d]   # maximal number of iteration
#         (see note (1) for [-ll|-hl])
#         [-corner-ending-condition|-rms] # evolution of image corners
#         ### filter type ###
#         [-gaussian-filter-type|-filter-type deriche|fidrich|young-1995|young-2002|...
#         ...|gabor-young-2002|convolution] # type of filter for image/vector field
#         smoothing
#         ### misc writing stuff ###
#         [-default-filenames|-df]     # use default filename names
#         [-no-default-filenames|-ndf] # do not use default filename names
#         [-command-line %s]           # write the command line
#         [-logfile %s]                # write some output in this logfile
#         [-vischeck]  # write an image with 'active' blocks
#         [-write_def] # id. 
#         ### parallelism ###
#         [-parallel|-no-parallel] # use parallelism (or not)
#         [-parallelism-type|-parallel-type default|none|openmp|omp|pthread|thread]
#         [-max-chunks %d] # maximal number of chunks
#         [-parallel-scheduling|-ps default|static|dynamic-one|dynamic|guided] # type
#         of scheduling for open mp
#         ### general parameters ###
#         -verbose|-v: increase verboseness parameters being read several time, use '-nv -v -v ...' to set the verboseness level
#         -debug|-D: increase debug level
#         -no-debug|-nodebug: no debug indication
#         -trace:
#         -no-trace:
#         -print-parameters|-param:
#         -print-time|-time:
#         -no-time|-notime:
#         -trace-memory|-memory: keep trace of allocated pointers (in instrumented procedures)
#         display some information about memory consumption
#         Attention: it disables the parallel mode, because of concurrent access to memory parallel mode may be restored by specifying '-parallel' after '-memory' but unexpected crashes may be experienced
#         -no-memory|-nomemory:
#         -h: print option list
#         -help: print option list + details
        
#         Notes
#         (1) If -ll or -hl are respectively added to the option, this specifies only the
#         value for respectively the lowest or the highest level of the pyramid (recall
#         that the most lowest level, ie #0, refers to the original image). For
#         intermediary levels, values are linearly interpolated.

class RegistrationMoments(Registration):
    def __init__(self, n_axis, align_center=True, align_rotation=True):
        super().__init__()
        self.align_center = align_center
        self.align_rotation = align_rotation
        self._n_spatial = None
        self._n_axis = n_axis

    def register(self, img_float, img_ref, scale, verbose=False):
        """Compute the transformation (translation + rotation) to align img2 to img1, considering anisotropy."""

        nx = cp if GPU_AVAILABLE else np 
        ndix = cndi if GPU_AVAILABLE else ndi
        skmx = cskm if GPU_AVAILABLE else skm
        img_float_ = nx.array(img_float)
        img_ref_ = nx.array(img_ref)
        
        # Compute center of the images in real-world coordinates
        center_image = (nx.array(img_float_.shape) - 1) / 2 * nx.array(scale)

        center1 = skmx.centroid(img_ref_, spacing=scale)  # Center of img1 in real-world space
        center2 = skmx.centroid(img_float_, spacing=scale)  # Center of img2 in real-world space

        # Compute translation to move img2 to img1
        translation = center2 - center1 if self.align_center else nx.zeros(3)

        # Compute rotation
        if self.align_rotation:
            selected_axes1 = (center1 - center_image)  # Strongest principal axes of img1
            selected_axes2 = (center2 - center_image)  # Strongest principal axes of img2

            # Normalize axes
            selected_axes1 /= nx.linalg.norm(selected_axes1, axis=0)
            selected_axes2 /= nx.linalg.norm(selected_axes2, axis=0)

            # Compute cross product (normal vector)
            normal = nx.cross(selected_axes1, selected_axes2)
            normal_norm = nx.linalg.norm(normal)

            if normal_norm < 1e-6:  # If vectors are nearly parallel, set identity rotation
                rotation_matrix = nx.eye(3)
            else:
                normal /= normal_norm
                angle = nx.arccos(nx.clip(nx.dot(selected_axes1, selected_axes2), -1.0, 1.0))

                K = nx.zeros((3, 3))
                K[0, 1], K[0, 2] = -normal[2], normal[1]
                K[1, 0], K[1, 2] = normal[2], -normal[0]
                K[2, 0], K[2, 1] = -normal[1], normal[0]
                rotation_matrix = nx.eye(3) + nx.sin(angle) * K + (1 - nx.cos(angle)) * (K @ K)
        else:
            rotation_matrix = nx.eye(3)

        # Convert to 4x4 affine transformation matrix
        rotation_matrix_aff = nx.eye(4)
        rotation_matrix_aff[:3,:3] = rotation_matrix

        # Translation matrices
        translation_aff = nx.eye(4)
        translation_aff[:3,3] = -center_image
        translation_inv_aff = nx.eye(4)
        translation_inv_aff[:3,3] = center_image

        translation_aff_center = nx.eye(4)
        translation_aff_center[:3,3] = -translation

        # Apply transformation in the correct order
        trnsf = translation_aff_center @ translation_inv_aff @ rotation_matrix_aff @ translation_aff

        return trnsf.get() if GPU_AVAILABLE else trnsf
    
class RegistrationManual(Registration):
    def __init__(self, n_axis=1):
        super().__init__()
    
    def fit(self, dataset, out=None, axis=None, scale=None, use_channel=None, stepping=None, direction="backward", verbose=False):

        self._perform_global_trnsf = True
        self._out = out
        if self._out is not None:
            if os.path.exists(self._out):
                self.load(out)
            else:
                self.save()
    
        if direction not in ["backward", "forward"]:
            raise ValueError("The direction must be either 'backward' or 'forward'.")
        self._registration_direction = direction

        # Check inputs
        axis, scale = _get_axis_scale(dataset, axis, scale)

        self._scale = scale
        self._axis = axis

        self._t_max = _dict_axis_shape(axis, dataset.shape)["T"]

        self._n_spatial = sum([i in "XYZ" for i in axis])

        self._spatial_shape = tuple([j for i,j in _dict_axis_shape(axis, dataset.shape).items() if i in "XYZ"])

        self._arrow_scale = np.sqrt(np.sum(np.array([j for i, j in _dict_axis_shape(axis, dataset.shape).items() if i in "XYZ"])**2)) / 4

        ax_ = [j for i,j in _dict_axis_shape(axis, dataset.shape).items() if i in "XYZ"]

        viewer = napari.Viewer()
        viewer.add_image(dataset, scale=scale, name="Dataset", opacity=0.5)
        viewer.add_image(dataset, scale=scale, name="Dataset (corrected)", opacity=0.5, colormap="green")
        dataset_dask = da.from_array(dataset, chunks=dataset.shape)
        if self._registration_direction == "backward":
            slicing = make_index(self._axis, T=slice(1,None,1))
            print(self._axis, slicing)
            viewer.add_image(dataset_dask[slicing], scale=scale, name="Dataset t+1", opacity=0.5, colormap="red")
        else:
            slicing = make_index(self._axis, T=slice(None,-1,None))
            viewer.add_image(dataset_dask[slicing], scale=scale, name="Dataset t+1", opacity=0.5, colormap="red")

        viewer.add_shapes(
            data=np.array([]),
            shape_type='polygon',
            edge_color='yellow',
            edge_width=1,
            face_color='transparent',
            opacity=0.1,
            name='Original Bounding Box',
            blending='translucent_no_depth'
        )

        # Add to napari as polygons
        viewer.add_shapes(
            data=np.array([]),
            shape_type='polygon',
            edge_color='white',
            edge_width=1,
            face_color='transparent',
            opacity=0.1,
            name='Bounding Box',
            blending='translucent_no_depth'
        )

        if direction == "backward":
            self._origin = 0
        else:
            self._origin = self._t_max-1

        if self._out is None:
            for t in range(self._t_max):
                self._save_transformation_global(np.eye(4), t, self._origin)
        elif len(os.listdir(os.path.join(self._out, "trnsf_global"))) == 0:
            for t in range(self._t_max):
                self._save_transformation_global(np.eye(4), t, self._origin)

        if self._n_spatial == 2:
            viewer.dims.ndisplay = 2
        else:
            viewer.dims.ndisplay = 3
        viewer.dims.current_step = (0, 0, 0)
        widget = AffineRegistrationWidget(self, viewer, axis)
        viewer.window.add_dock_widget(widget, area='right')
        napari.run()

        self._fitted = True
        if self._out is not None:
            self.save()

class AffineRegistrationWidget(QWidget):
    
    def __init__(self, model, viewer, axis):
        super().__init__()

        self.viewer = viewer
        self.model = model
        self.axis = axis

        self._registration_model = RegistrationVT()
        self._registration_model._n_spatial = model._n_spatial
        self._registration_model._spatial_shape = model._spatial_shape
        self._last_t = viewer.dims.current_step[axis.index("T")]
        self.setWindowTitle("Rotation Widget")
        self.layout = QVBoxLayout()
        self.explanation = self.create_explanation()
        self.register_button = self.create_button("Register", self.register)
        self.rotation_slider = self.create_slider("Rotation step", 1, 360, 1)
        self.translation_slider = self.create_slider("Translation Step", 1, np.min(model._spatial_shape), min(1,np.min(model._spatial_shape)//100))
        self.propagate_backwards_button = self.create_button("Propagate backwards", self.propagate_backwards)
        self.propagate_forwards_button = self.create_button("Propagate forwards", self.propagate_forwards)
        self.make_limits()
        self.setLayout(self.layout)

        faces = self.bounding_box(0, self.model._spatial_shape[0], 0, self.model._spatial_shape[1], 0, self.model._spatial_shape[2])
        self.viewer.layers["Original Bounding Box"].data = faces

        # viewer.mouse_drag_callbacks.append(self.on_mouse_drag)
        self.viewer.bind_key("Shift+W",self.on_keyboard_up_translate)
        self.viewer.bind_key("Shift+S",self.on_keyboard_down_translate)
        self.viewer.bind_key("Shift+A",self.on_keyboard_left_translate)
        self.viewer.bind_key("Shift+D",self.on_keyboard_right_translate)
        self.viewer.bind_key("Shift+Q",self.on_keyboard_counterclockwise_rotate)
        self.viewer.bind_key("Shift+E",self.on_keyboard_clockwise_rotate)
        self.viewer.bind_key("Shift+R",self.register)
        self.viewer.dims.events.current_step.connect(self.update_images)
        # self.viewer.window._qt_viewer.canvas.setFocus()

    def create_explanation(self):
        widget = QWidget()
        layout = QVBoxLayout()
        label = QLabel(
            """
            Shift + W: Translate up.
            Shift + S: Translate down.
            Shift + A: Translate left.
            Shift + D: Translate right.
            Shift + Q: Rotate counterclockwise.
            Shift + E: Rotate clockwise.

            Shift + R: Register the transformation.
            """
        )
        layout.addWidget(label)
        widget.setLayout(layout)
        self.layout.addWidget(widget)
        return widget

    def create_button(self, label_text, callback):
        button = QPushButton(label_text)
        button.clicked.connect(callback)
        self.layout.addWidget(button)
        return button

    def create_slider(self, label_text, min_value, max_value, set_value):
        widget = QWidget()
        layout = QVBoxLayout()
        label = QLabel(f"{label_text}: {set_value}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(set_value)
        slider.valueChanged.connect(lambda value, l=label: l.setText(f"{label_text}: {value}"))
        layout.addWidget(label)
        layout.addWidget(slider)
        widget.setLayout(layout)
        self.layout.addWidget(widget)
        return widget

    def make_limits(self):
        coord_widget = QWidget()
        coord_layout = QGridLayout()
        coord_layout.setContentsMargins(0, 0, 0, 0)
        coord_layout.setSpacing(5)

        # Row 0: Min values
        self.minX_input, self.minX_field = self.create_labeled_input("Min X", 0)
        self.minY_input, self.minY_field = self.create_labeled_input("Min Y", 0)
        if self._registration_model._n_spatial == 3:
            self.minZ_input, self.minZ_field = self.create_labeled_input("Min Z", 0)

        coord_layout.addWidget(self.minX_input, 0, 0)
        coord_layout.addWidget(self.minY_input, 0, 1)
        if self._registration_model._n_spatial == 3:
            coord_layout.addWidget(self.minZ_input, 0, 2)

        # Row 1: Max values
        self.maxX_input, self.maxX_field = self.create_labeled_input("Max X", self._registration_model._spatial_shape[0])
        self.maxY_input, self.maxY_field = self.create_labeled_input("Max Y", self._registration_model._spatial_shape[1])
        if self._registration_model._n_spatial == 3:
            self.maxZ_input, self.maxZ_field = self.create_labeled_input("Max Z", self._registration_model._spatial_shape[2])

        coord_layout.addWidget(self.maxX_input, 1, 0)
        coord_layout.addWidget(self.maxY_input, 1, 1)
        if self._registration_model._n_spatial == 3:
            coord_layout.addWidget(self.maxZ_input, 1, 2)

        coord_widget.setLayout(coord_layout)
        self.layout.addWidget(coord_widget)

        # Connect input changes to bounding_box
        self.minX_field.textChanged.connect(self.update_bounding_box)
        self.minY_field.textChanged.connect(self.update_bounding_box)
        self.maxX_field.textChanged.connect(self.update_bounding_box)
        self.maxY_field.textChanged.connect(self.update_bounding_box)
        if self._registration_model._n_spatial == 3:
            self.minZ_field.textChanged.connect(self.update_bounding_box)
            self.maxZ_field.textChanged.connect(self.update_bounding_box)

        # Initial box
        self.update_bounding_box()

    def create_labeled_input(self, label_text, default_value):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        label = QLabel(label_text)
        label.setFixedWidth(45)
        input_field = QLineEdit()
        input_field.setText(str(default_value))
        input_field.setFixedWidth(60)

        layout.addWidget(label)
        layout.addWidget(input_field)
        widget.setLayout(layout)

        return widget, input_field  # Return both

    def bounding_box(self, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None):

        # Define the 8 corners of the box
        corners = np.array([
            [z_min, y_min, x_min],
            [z_max, y_min, x_min],
            [z_max, y_max, x_min],
            [z_min, y_max, x_min],
            [z_min, y_min, x_max],
            [z_max, y_min, x_max],
            [z_max, y_max, x_max],
            [z_min, y_max, x_max],
        ])

        # Define the 6 faces using 4 vertices each (polygons)
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],  # Bottom
            [corners[4], corners[5], corners[6], corners[7]],  # Top
            [corners[0], corners[1], corners[5], corners[4]],  # Front
            [corners[2], corners[3], corners[7], corners[6]],  # Back
            [corners[1], corners[2], corners[6], corners[5]],  # Right
            [corners[3], corners[0], corners[4], corners[7]],  # Left
        ]

        return np.array(faces)
    
    def update_bounding_box(self):

        # Get coordinates from UI
        x_min = float(self.minX_input.findChild(QLineEdit).text())
        x_max = float(self.maxX_input.findChild(QLineEdit).text())
        y_min = float(self.minY_input.findChild(QLineEdit).text())
        y_max = float(self.maxY_input.findChild(QLineEdit).text())
        z_min = float(self.minZ_input.findChild(QLineEdit).text()) if self._registration_model._n_spatial == 3 else 0
        z_max = float(self.maxZ_input.findChild(QLineEdit).text()) if self._registration_model._n_spatial == 3 else 0

        faces = self.bounding_box(x_min, x_max, y_min, y_max, z_min, z_max)

        self.viewer.layers["Bounding Box"].data = faces

    def register(self,*args):
        t = self.viewer.dims.current_step[self.axis.index("T")]
        if "C" in self.axis:
            slicing = make_index(self.axis, 
                                 T = self.viewer.dims.current_step[self.axis.index("T")],
                                 C = self.viewer.dims.current_step[self.axis.index("C")]
            )
            if t == self.model._t_max-1:
                slicing_ = make_index(self.axis, 
                                     T = self.viewer.dims.current_step[self.axis.index("T")-1],
                                     C = self.viewer.dims.current_step[self.axis.index("C")]
                )
            else:
                slicing_ = slicing
        else:
            slicing = make_index(self.axis, 
                                 T = self.viewer.dims.current_step[self.axis.index("T")]
            )
            if t == self.model._t_max-1:
                slicing_ = make_index(self.axis, 
                                     T = self.viewer.dims.current_step[self.axis.index("T")-1],
                )
            else:
                slicing_ = slicing

        layer = self.viewer.layers["Dataset t+1"]
        # translation = np.eye(4)
        # translation[:3, 3] = np.array([0, 0, 20])  # Replace with desired translation values (x, y, z)
        # trnsf = translation
        # self.viewer.layers["Dataset t+1"].data[0,t] = ndi.affine_transform(self.viewer.layers["Dataset"].data[0,t], trnsf, order=1)
        img_ref = self.viewer.layers["Dataset"].data[slicing]
        img_float = self.viewer.layers["Dataset t+1"].data[slicing_]
        trnsf_ref = self.model._load_transformation_global(t, self.model._origin)
        trnsf_float = self.model._load_transformation_global(t+1, self.model._origin)
        # trnsf_ref = self.model._transfs[t]
        # trnsf_float = self.model._transfs[t+1]
        # print(trnsf_ref)
        # print(trnsf_float)
        # print(trnsf_float)
        # trnsf = np.linalg.inv(trnsf)
        # print(trnsf)

        # Scaling matrices (fixing order of operations)
        img_ref = self.model.apply_trnsf(img_ref, trnsf_ref, self.model._scale)
        img_float = self.model.apply_trnsf(img_float, trnsf_float, self.model._scale)

        trnsf = self._registration_model.register(
            img_float, 
            img_ref, 
            self.model._scale, 
            verbose=True
        )

        # trnsf= np.linalg.inv(trnsf)

        # trnsf = vt.inv_trsf(trnsf).copy_to_array()
        # trnsf = trnsf.copy_to_array()

        # trnsf = scaling_inv @ trnsf @ scaling
        # trnsf = trnsf
        # trnsf[:3, :3] = trnsf[:3, :3][:, [2, 1, 0]]
        # trnsf[:3, :3] = trnsf[:3, :3][[2, 1, 0],:]
        # trnsf[:3, 3] = trnsf[:3, 3][[2, 1, 0]]

        # print(trnsf)
        trnsf = self.model.compose_trnsf([trnsf, trnsf_float])
        # print(trnsf)

        # t = self.viewer.dims.current_step[self.axis.index("T")]
        # layer.affine = np.linalg.inv(trnsf)
        self.model._save_transformation_global(trnsf, t+1, self.model._origin)
        self.update_images(None, refresh=True)
        # self.model._transfs[t+1] = trnsf

    def update_images(self, event, refresh=False):
        # Get current time step
        t = self.viewer.dims.current_step[self.axis.index("T")]

        # Check if t has actually changed
        if hasattr(self, "_last_t") and self._last_t == t and not refresh:
            return  # Skip update if t is the same as before

        # Store the new value of t
        self._last_t = t  

        # Now update the affine transformations only if necessary
        self.viewer.layers["Dataset (corrected)"].affine = np.linalg.inv(self.model._load_transformation_global(t, self.model._origin))
        self.viewer.layers["Dataset t+1"].affine = np.linalg.inv(self.model._load_transformation_global(min(t+1, self.model._t_max-1), self.model._origin))

        # self.viewer.layers["Dataset (corrected)"].affine = self.model._transfs[t]
        # self.viewer.layers["Dataset t+1"].affine = self.model._transfs[t + 1]

    def on_mouse_drag_major(self, viewer, event: Event):
        """Rotate dataset about its center around the camera view direction,
        accumulating the rotation incrementally.
        
        The rotation axis is given by self.viewer.camera.view_direction.
        """
        # Only operate if ALT is held (and CONTROL is not)
        if ALT not in event.modifiers or CONTROL in event.modifiers:
            return

        # Yield once to start listening for drag events
        yield

        # Initialize the starting mouse position and an accumulated transform.
        pos0 = None
        layer = self.viewer.layers["Dataset t+1"]

        while event.type == "mouse_move":
            if pos0 is None:
                pos0 = event.pos
            else:
                accumulated_affine = layer.affine.affine_matrix[2:,2:]

                # Compute the center of the dataset.
                # For a 3D image, we assume layer.data.shape is (z, y, x) and compute center in (x, y, z).
                center = np.array(layer.data.shape[-3:], dtype=float) / 2.0
                center = center*layer.scale[-3:]

                y = self.viewer.camera.up_direction
                y = y / np.linalg.norm(y)
                z= self.viewer.camera.view_direction
                z = z / np.linalg.norm(z)
                x = np.cross(y, z)
                x = x / np.linalg.norm(x)

                delta = (event.pos - pos0)/10

                t = self.viewer.dims.current_step[self.axis.index("T")]
                v1 = self.model._axis1[t, 1, :]
                v2 = v1 + x * delta[0] - y * delta[1]
                self.model._axis1[t, 1, :] = self.model._axis1[t, 1, :] / np.linalg.norm(self.model._axis1[t, 1, :]) * self.model._arrow_scale
                
                v1 /= np.linalg.norm(v1)
                v2 /= np.linalg.norm(v2)
                axis = np.cross(v1, v2)

                if np.linalg.norm(axis) < 1e-6:
                    # No rotation is possible if the vectors are parallel.
                    # Skip the rest of the loop and wait for the next event.
                    pos0 = event.pos
                else:
                    axis /= np.linalg.norm(axis)
                    angle_delta = np.arccos(np.dot(v1, v2)) * 0.01                

                    # For 3D rotation, build the 4×4 incremental transform.
                    # First, translate the center to the origin:
                    T_translate = np.eye(4)
                    T_translate[:3, 3] = -center
                    # Then, translate back:
                    T_back = np.eye(4)
                    T_back[:3, 3] = center

                    # Build the 3×3 rotation matrix using Rodrigues' formula.
                    I = np.eye(3)
                    K = np.array([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]
                    ])
                    R_3 = I * np.cos(angle_delta) + np.sin(angle_delta) * K + (1 - np.cos(angle_delta)) * np.outer(axis, axis)
                    # Embed the 3×3 rotation into a 4×4 homogeneous matrix.
                    R = np.eye(4)
                    R[:3, :3] = R_3

                    # Compute the incremental transformation: move to origin, rotate, and move back.
                    incremental_transform = T_back @ R @ T_translate

                    # Accumulate the transform (apply the new incremental transform on top of the previous one)
                    accumulated_affine = incremental_transform @ accumulated_affine

                    # Update the layer’s affine transform
                    layer.affine = accumulated_affine

                    # Update the starting position for the next delta computation.
                    pos0 = event.pos

            # Yield control back to napari so it can process other events.
            yield

    def propagate_forwards(self, accumulated_affine):
        t = self.viewer.dims.current_step[self.axis.index("T")]
        trnsf = self.model._load_transformation_global(t+1, self.model._origin)
        for t_ in range(t+2, self.model._t_max):
            self.model._save_transformation_global(trnsf, t_, self.model._origin)
            # self.model._transfs[t_] = trnsf
            # self.model._transfs[t_] = self.model._transfs[t+1]
        self.viewer.dims.set_current_step(self.axis.index("T"), min(self.model._t_max - 1, t + 1))
    
    def propagate_backwards(self, accumulated_affine):
        t = self.viewer.dims.current_step[self.axis.index("T")]
        trnsf = self.model._load_transformation_global(t, self.model._origin)
        for t_ in range(t-1, -1, -1):
            self.model._save_transformation_global(trnsf, t_, self.model._origin)
            # self.model._transfs[t_] = self.model._transfs[t-1]
        self.viewer.dims.set_current_step(self.axis.index("T"), max(0, t - 1))

    def rotate(self, layer, axis):
        angle = np.radians(self.rotation_slider.findChild(QSlider).value())
        accumulated_affine = layer.affine.affine_matrix[-self.model._n_spatial-1:,-self.model._n_spatial-1:]
        center = np.array(layer.data.shape[-3:], dtype=float) / 2.0
        center = center*layer.scale[-3:]
        # For 3D rotation, build the 4×4 incremental transform.
        # First, translate the center to the origin:
        T_translate = np.eye(4)
        T_translate[:3, 3] = -center
        # Then, translate back:
        T_back = np.eye(4)
        T_back[:3, 3] = center
        # Build the 3×3 rotation matrix using Rodrigues' formula.
        I = np.eye(3)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R_3 = I * np.cos(angle) + np.sin(angle) * K + (1 - np.cos(angle)) * np.outer(axis, axis)
        # Embed the 3×3 rotation into a 4×4 homogeneous matrix.
        R = np.eye(4)
        R[:3, :3] = R_3
        # Compute the incremental transformation: move to origin, rotate, and move back.
        incremental_transform = T_back @ R @ T_translate
        # Accumulate the transform (apply the new incremental transform on top of the previous one)
        accumulated_affine = incremental_transform @ accumulated_affine
        # Update the layer’s affine transform
        layer.affine = accumulated_affine
        t = self.viewer.dims.current_step[self.axis.index("T")]
        self.model._save_transformation_global(np.linalg.inv(accumulated_affine), t+1, self.model._origin)
        # self.model._transfs[t+1] = accumulated_affine
        # for t_ in range(t+1, self.model._t_max):
        #     self.model._transfs[t_] = accumulated_affine

    def translate(self, layer, axis):
        step = self.translation_slider.findChild(QSlider).value()
        
        # Get the current affine transformation (6×6 matrix)
        accumulated_affine = layer.affine.affine_matrix[-self.model._n_spatial-1:,-self.model._n_spatial-1:]

        # Build a 6×6 translation matrix
        T_translate = np.eye(4)  # Identity matrix
        T_translate[:3, -1] = np.array(axis) * step  # Apply translation correctly

        # Accumulate the transformation
        accumulated_affine = T_translate @ accumulated_affine  # Ensure correct order

        # Properly assign the new affine transformation
        layer.affine = accumulated_affine
        t = self.viewer.dims.current_step[self.axis.index("T")]
        self.model._save_transformation_global(np.linalg.inv(accumulated_affine), t+1, self.model._origin)
        # self.model._transfs[t+1] = accumulated_affine

    def on_keyboard_clockwise_rotate(self, viewer):
        """Rotate dataset about its center around the camera view direction,
        accumulating the rotation incrementally.
        
        The rotation axis is given by self.viewer.camera.view_direction.
        """
        layer = self.viewer.layers["Dataset t+1"]

        v = np.array(self.viewer.camera.view_direction)
        print("clockwise", v)

        self.rotate(layer, -v)

    def on_keyboard_counterclockwise_rotate(self, viewer):
        """Rotate dataset about its center around the camera view direction,
        accumulating the rotation incrementally.
        
        The rotation axis is given by self.viewer.camera.view_direction.
        """
        layer = self.viewer.layers["Dataset t+1"]

        v = np.array(self.viewer.camera.view_direction)
        print("counterclockwise", v)

        self.rotate(layer, v)

    def on_keyboard_up_translate(self, viewer):
        """Rotate dataset about its center around the camera view direction,
        accumulating the rotation incrementally.
        
        The rotation axis is given by self.viewer.camera.view_direction.
        """
        layer = self.viewer.layers["Dataset t+1"]

        v = self.up_direction()
        print("up", v)

        self.translate(layer, v)

    def on_keyboard_down_translate(self, viewer):
        """Rotate dataset about its center around the camera view direction,
        accumulating the rotation incrementally.
        
        The rotation axis is given by self.viewer.camera.view_direction.
        """
        layer = self.viewer.layers["Dataset t+1"]

        v = self.up_direction()
        print("down", v)

        self.translate(layer, -v)

    def on_keyboard_right_translate(self, viewer):
        """Rotate dataset about its center around the camera view direction,
        accumulating the rotation incrementally.
        
        The rotation axis is given by self.viewer.camera.view_direction.
        """
        layer = self.viewer.layers["Dataset t+1"]

        v = self.right_direction()
        print("right", v)

        self.translate(layer, v)

    def on_keyboard_left_translate(self, viewer):
        """Rotate dataset about its center around the camera view direction,
        accumulating the rotation incrementally.
        
        The rotation axis is given by self.viewer.camera.view_direction.
        """
        layer = self.viewer.layers["Dataset t+1"]

        v = self.right_direction()
        print("left", v)

        self.translate(layer, -v)

    def up_direction(self):
        x = self.viewer.camera.up_direction
        x = x / np.linalg.norm(x)
        return x

    def right_direction(self):
        y = self.viewer.camera.up_direction
        y = y / np.linalg.norm(y)
        z = self.viewer.camera.view_direction
        z = z / np.linalg.norm(z)
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)
        return x
