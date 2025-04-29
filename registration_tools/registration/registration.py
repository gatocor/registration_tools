import os
import re
import numpy as np
from skimage.io import imread, imsave
import skimage.measure as skm
import scipy.ndimage as ndi
import shutil
import json
from tqdm import tqdm
import time
import warnings  # Add this import
from copy import deepcopy
import zarr
import tempfile
import time
from copy import deepcopy
import threading
import queue
import dask.array as da
from pathlib import Path
# from dask.diagnostics import ProgressBar

try:
    import vt
except:
    None
import SimpleITK as sitk

from ..dataset.dataset import Dataset
from ..utils.auxiliar import _dict_axis_shape, _shape_padded, _get_axis_scale, _make_index, _image_padded, _suppress_stdout_stderr, _trnsf_padded, make_index
from ..utils.utils import apply_function
from ..constants import GPU_AVAILABLE, USE_GPU

REGISTRATION_TYPES_VT = ["translation", "translation2D", "translation3D", "rotation", "rotation2D", "rotation3D", "rigid", "rigid2D", "rigid3D", "affine", "affine2D", "affine3D", "vectorfield", "vectorfield2D", "vectorfield3D"]
SITK_TRANSFORMATIONS = ["TranslationTransform", "VersorTransform", "VersorRigid3DTransform", "Euler2DTransform", "Euler3DTransform", "Similarity2DTransform", "Similarity3DTransform", "ScaleTransform", "ScaleVersor3DTransform", "ScaleSkewVersor3DTransform", "ComposeScaleSkewVersor3DTransform", "AffineTransform", "BSplineTransform", "DisplacementFieldTransform"] 
SITK_METRICS = ["MeanSquares",  "Demons",  "Correlation",  "ANTSNeighborhoodCorrelation",  "JointHistogramMutualInformation",  "MattesMutualInformation"]        
SITK_OPTIMIZERS = ["Exhaustive", "Nelder-Mead", "Powell", "Evolutionary", "GradientDescent", "GradientDescentLineSearch", "RegularStepGradientDescent", "ConjugateGradientLineSearch", "L-BFGS-B"]       
SITK_SAMPLING = ["None", "Random", "Regular"]

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
        self._padding = None
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
                self.write_trnsf(trnsf, f"{self._out}/trnsf_global/{name}")
            elif "relative" in name:
                self.write_trnsf(trnsf, f"{self._out}/trnsf/{name}")

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
            self._transfs[f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}"] = trnsf
        else:
            self.write_trnsf(trnsf, f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}")

    def _save_transformation_global(self, trnsf, pos_float, pos_ref):
        if self._out is None:
            self._transfs[f"trnsf_global_{pos_float:04d}_{pos_ref:04d}"] = trnsf
        else:
            self.write_trnsf(trnsf, f"{self._out}/trnsf_global/trnsf_global_{pos_float:04d}_{pos_ref:04d}")

    def _load_transformation_relative(self, pos_float, pos_ref):
        if self._out is None:
            if f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}" in self._transfs:
                return self._transfs[f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}"]
            else:
                raise ValueError(f"The transformation between {pos_float:04d} and {pos_ref:04d} the specified positions does not exist.")
        else:
            if self._trnsf_exists_relative(pos_float, pos_ref):
                return self.read_trnsf(f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}")
            else:
                raise ValueError("The transformation between the specified positions does not exist.")

    def _load_transformation_global(self, pos_float, pos_ref):
        if self._out is None:
            if f"trnsf_global_{pos_float:04d}_{pos_ref:04d}" not in self._transfs:
                raise ValueError(f"The transformation between {pos_float:04d} and {pos_ref:04d} the specified positions does not exist.")
            else:
                return self._transfs[f"trnsf_global_{pos_float:04d}_{pos_ref:04d}"]
        else:
            if self._trnsf_exists_global(pos_float, pos_ref):
                return self.read_trnsf(f"{self._out}/trnsf_global/trnsf_global_{pos_float:04d}_{pos_ref:04d}")
            else:
                raise ValueError("The transformation between the specified positions does not exist.")

    def _trnsf_exists_relative(self, pos_float, pos_ref):
        if self._out is None:
            return f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}" in self._transfs.keys()   
        else:
            base_path = Path(f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}")
            return bool(list(base_path.parent.glob(base_path.name + ".*")))
    
    def _trnsf_exists_global(self, pos_float, pos_ref):
        if self._out is None:
            return f"trnsf_global_{pos_float:04d}_{pos_ref:04d}" in self._transfs.keys()
        else:
            base_path = Path(f"{self._out}/trnsf_global/trnsf_global_{pos_float:04d}_{pos_ref:04d}")
            return bool(list(base_path.parent.glob(base_path.name + ".*")))
        
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

        nx = cp if GPU_AVAILABLE and USE_GPU else np 
        ndix = cndi if GPU_AVAILABLE and USE_GPU else ndi

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

        return transformed_img.get() if GPU_AVAILABLE and USE_GPU else transformed_img

    def apply_trnsf_to_points(self, points, trnsf):

        points_ = trnsf @ np.concatenate((points, np.ones((points.shape[0], 1))), axis=1).T
        points_ = points_.T[:, :-1]

        return points_

    def write_trnsf(self, trnsf, out):
        np.savetxt(out+".trnsf", trnsf, fmt='%.6f')

    def read_trnsf(self, out):
        return np.loadtxt(out+".trnsf")

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

            pos_ref_current = pos_float
        
        self._fitted = True
        if self._out is not None:
            self.save()

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
        if padding is None:
            padding = self._padding
        new_shape = _shape_padded(dataset.shape, axis, padding)
        self._padding = padding

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
            desc=f"Applying registration to images",
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

        # New shape
        if padding is None:
            padding = self._padding
        new_shape = _shape_padded(dataset.shape, axis, padding)
        self._padding = padding

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
                    im_trnsf = self.apply_trnsf(img_, trnsf_img, new_scale, padding)
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

        self._fitted = True
        if self._out is not None:
            self.save()

        if padding:
            print("Padding data")
            return self.apply(data, axis=axis, scale=scale, downsample=downsample, save_behavior="Overwrite", perform_global_trnsf=perform_global_trnsf, padding=padding, verbose=verbose, **kwargs)
        else:
            return data

    def propagate(self):
        if self._registration_direction == "forward":
            self.propagate_backwards()
        elif self._registration_direction == "backward":
            self.propagate_forwards()

    def propagate_forwards(self):
        trnsf_global = np.eye(self._n_spatial+1)
        for t_ in range(0, self._t_max):
            trnsf = self._load_transformation_relative(t_, self._origin)
            trnsf_global = self.compose_trnsf([trnsf_global, trnsf])
            self._save_transformation_global(trnsf_global, t_, self._origin)
    
    def propagate_backwards(self):
        trnsf_global = np.eye(self._n_spatial+1)
        for t_ in range(self._t_max-1, -1, -1):
            trnsf = self._load_transformation_relative(t_, self._origin)
            trnsf_global = self.compose_trnsf([trnsf_global, trnsf])
            self._save_transformation_global(trnsf_global, t_, self._origin)

    def fit_manual(self, dataset, out=None, axis=None, scale=None, use_channel=None, stepping=None, direction="backward", verbose=False):

        try:
            import napari
            from ..widgets.widgets import AffineRegistrationWidget
        except:
            raise ImportError("Napari is not installed. Please install it to use this function.")

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

        if self._registration_direction == "backward":
            c = +1
        else:
            c = -1

        viewer = napari.Viewer()
        dataset_dask = da.from_array(dataset, chunks=dataset.shape)

        if self._registration_direction == "backward":
            c = +1
            # Align: Dataset[0:-1], Dataset_t+1[1:]
            ref_slice     = make_index(self._axis, T=slice(0, -1))  # i
            shifted_slice = make_index(self._axis, T=slice(1, None))  # i+1
        else:
            c = -1
            # Align: Dataset[1:], Dataset_t-1[0:-1]
            ref_slice     = make_index(self._axis, T=slice(1, None))  # i
            shifted_slice = make_index(self._axis, T=slice(0, -1))    # i-1

        # Add the sliced dataset (aligned in time)
        _layer_dataset = viewer.add_image(dataset_dask[ref_slice], scale=scale, name="Dataset", opacity=0.5)
        _layer_corrected = viewer.add_image(dataset_dask[ref_slice], scale=scale, name="Dataset (corrected)", opacity=0.5, colormap="green")
        _layer_next = viewer.add_image(dataset_dask[shifted_slice], scale=scale, name=f"Dataset t{c}", opacity=0.5, colormap="red")

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

        for t in range(self._t_max):
            if not self._trnsf_exists_relative(t, self._origin):
                self._save_transformation_relative(np.eye(self._n_spatial+1), t, self._origin)
        self.propagate()

        if self._n_spatial == 2:
            viewer.dims.ndisplay = 2
        else:
            viewer.dims.ndisplay = 3

        start_pos = [0] * len(self._axis)
        start_pos[self._axis.index("T")] = 0 if self._registration_direction == "backward" else self._t_max-2
        viewer.dims.current_step = start_pos
        widget = AffineRegistrationWidget(self, viewer, axis)
        widget._layer_dataset = _layer_dataset        
        widget._layer_corrected = _layer_corrected
        widget._layer_next = _layer_next        
        viewer.window.add_dock_widget(widget, area='right')
        napari.run()

        self._fitted = True
        if self._out is not None:
            self.save()

    def vectorfield(self, mask=None, out=None, axis=None, scale=None, n_points=20, transformation="relative", **kwargs):

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
                keep_points = points[mask[*mesh_mask].flatten()]
        else:
            keep_points = points

        if out is None:
            store = zarr.storage.MemoryStore()
            data = zarr.create_array(store=store, shape=(0,2,self._n_spatial+1), dtype=float, **kwargs)
        elif isinstance(out, str) and out.endswith(".zarr"):
            data = zarr.create_array(
                store=out,
                shape=(0,2,self._n_spatial+1),
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
                keep_points = points[mask[make_index(axis, T=t)][*mesh_mask].flatten()]
            else:
                keep_points = points

            if transformation == "global":
                trnsf = self._load_transformation_global(t, self._origin)
            elif transformation == "relative":
                trnsf = self._load_transformation_relative(t, t_next)
             
            points_out = self.apply_trnsf_to_points(keep_points, trnsf)
            vectorfield = points_out - keep_points
            # vectorfield = keep_points
            l = vectorfield.shape[0]
            data_add = np.zeros((l,2,self._n_spatial+1))
            data_add[:,0,0] = t
            data_add[:,1,0] = 0
            data_add[:,0,1:] = keep_points
            data_add[:,1,1:] = vectorfield

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
                keep_points = points[mask[*mesh_mask].flatten()]
            # else:
            #     raise ValueError("The mask cannot contain the time dimension 'T'.")
        else:
            keep_points = points

        if out is None:
            store = zarr.storage.MemoryStore()
            data = zarr.create_array(store=store, shape=(0,self._n_spatial+2), dtype=float, **kwargs)
        elif isinstance(out, str) and out.endswith(".zarr"):
            data = zarr.create_array(
                store=out,
                shape=(0,self._n_spatial+2),
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
        data_add = np.zeros((l,self._n_spatial+2))
        data_add[:,0] = range(len(keep_points))
        data_add[:,1] = t
        data_add[:,2:] = keep_points
        data.append(data_add, axis=0)

        for t, t_next in tqdm(zip(iterator, iterator_next), desc=f"Computing trajectories", unit="", total=self._t_max-1):

            if transformation == "global":
                trnsf = self._load_transformation_global(t, self._origin)
            elif transformation == "relative":
                trnsf = self._load_transformation_relative(t, t_next)
             
            points_out = self.apply_trnsf_to_points(keep_points, trnsf)
            keep_points = keep_points - (points_out-keep_points)
            l = keep_points.shape[0]
            data_add = np.zeros((l,self._n_spatial+2))
            data_add[:,0] = range(len(keep_points))
            data_add[:,1] = t_next
            data_add[:,2:] = keep_points

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

    def score_registration(self, img_float, img_ref, scale, 
            metric="MeanSquares", **kwargs):
            
        if metric not in SITK_METRICS:
            raise ValueError(f"Metric {metric} is not supported. Supported metrics are: {SITK_METRICS}")

        img_float_sitk = sitk.GetImageFromArray(img_float.astype(np.float32))
        img_float_sitk.SetSpacing(scale[::-1])
        img_ref_sitk = sitk.GetImageFromArray(img_ref.astype(np.float32))
        img_ref_sitk.SetSpacing(scale[::-1])

        evaluator = sitk.ImageRegistrationMethod()
        if metric == "MeanSquares":
            evaluator.SetMetricAsMeanSquares(**kwargs)  # or any other metric
        elif metric == "Correlation":
            evaluator.SetMetricAsCorrelation(**kwargs)
        elif metric == "JointHistogram":
            evaluator.SetMetricAsJointHistogram(**kwargs)
        # evaluator.SetMetricAsMeanSquares()  # or any other metric
        evaluator.SetOptimizerAsGradientDescent(0.1,100)
        evaluator.SetInitialTransform(sitk.TranslationTransform(self._n_spatial))
        evaluator.SetInterpolator(sitk.sitkLinear)
        score = evaluator.MetricEvaluate(img_ref_sitk, img_float_sitk)

        return score

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
    
    def apply_trnsf(self, image, trnsf, scale, padding):

        vtImage = self._array2image(image, scale)

        img = vt.apply_trsf(vtImage, trnsf).copy_to_array()

        return img

    def apply_trnsf_to_points(self, points, trnsf):
        """
        Apply a SimpleITK transform to a list/array of points.
        
        Parameters:
        - points: (N, D) numpy array of N D-dimensional points (D=2 or 3)
        - trnsf: SimpleITK.Transform (can be affine, displacement field, etc.)
        
        Returns:
        - Transformed points as numpy array of shape (N, D)
        """

        vtPoints = vt.vtPointList(points[:,::-1].tolist())
        vtPoints_ = vt.apply_trsf_to_points(vtPoints, vt.inv_trsf(trnsf))
        # print(vtPoints_.copy_to_array()[:,:self._n_spatial][:,::-1])

        return vtPoints_.copy_to_array()[:,:self._n_spatial][:,::-1]

    def write_trnsf(self, trnsf, out):
        vt.write_trnsf(trnsf, out+".trnsf")

    def read_trnsf(self, out):
        return vt.read_trnsf(out+".trnsf")

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

        #Invert axis order to make it consistent with input order
        if self._n_spatial == 2:
            v = [0,1,-1]
            v_ = [1,0,-1]
        else:
            v = [0,1,2,-1]
            v_ = [2,1,0,-1]
        # trnsf = trnsf.copy_to_array()
        # trnsf = trnsf[:,v_]
        # trnsf = trnsf[v_,:]

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

class RegistrationSITK(Registration):
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
            registration_type,
            metric="MeanSquares", 
            optimizer="RegularStepGradientDescent",
            sampling="None",
            optimizer_learningRate=1,
            optimizer_minStep=1e-5,
            optimizer_numberOfIterations=500,
            optimizer_gradientMagnitudeTolerance=1e-10,        
            optimizer_radius=1,
            pyramid_shrink_factor=(1,),
            pyramid_smoothing_sigmas=(0,),
            displacement_field_smoothing_sigmas=0.,
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

        if registration_type not in SITK_TRANSFORMATIONS:
            raise ValueError(f"Invalid registration type: {registration_type}. Supported types are: {SITK_TRANSFORMATIONS}")

        if metric not in SITK_METRICS:
            raise ValueError(f"Invalid metric: {metric}. Supported metrics are: {SITK_METRICS}")

        if optimizer not in SITK_OPTIMIZERS:
            raise ValueError(f"Invalid optimizer: {optimizer}. Supported optimizers are: {SITK_OPTIMIZERS}")
        
        if sampling not in SITK_SAMPLING:
            raise ValueError(f"Invalid sampling: {sampling}. Supported samplings are: {SITK_SAMPLING}")

        self._registration_type = registration_type
        self._metric = metric
        self._sampling = sampling
        self._optimizer = optimizer
        self._optimizer_learningRate = optimizer_learningRate
        self._optimizer_minStep = optimizer_minStep
        self._optimizer_numberOfIterations = optimizer_numberOfIterations
        self._optimizer_gradientMagnitudeTolerance = optimizer_gradientMagnitudeTolerance
        self._optimizer_radius = optimizer_radius
        self._pyramid_shrink_factor = pyramid_shrink_factor
        self._pyramid_smoothing_sigmas = pyramid_smoothing_sigmas
        self._displacement_field_smoothing_sigmas = displacement_field_smoothing_sigmas

    def setMetric(self, R):

        if self._metric == "MeanSquares":
            R.SetMetricAsMeanSquares()
        elif self._metric == "Demons":
            R.SetMetricAsDemons()
        elif self._metric == "Correlation":
            R.SetMetricAsCorrelation()
        elif self._metric == "ANTSNeighborhoodCorrelation":
            R.SetMetricAsANTSNeighborhoodCorrelation(
                self._optimizer_radius
            )
        elif self._metric == "JointHistogramMutualInformation":
            R.SetMetricAsJointHistogramMutualInformation()
        elif self._metric == "MattesMutualInformation":
            R.SetMetricAsMattesMutualInformation()
        else:
            raise ValueError(f"Invalid metric: {self._metric}. Supported metrics are: {SITK_METRICS}")
        
    def setOptimizer(self, R):

        if self._optimizer == "Exhaustive":
            R.SetOptimizerAsExhaustive(self._optimizer_iterations, 0.1)
        elif self._optimizer == "Nelder-Mead":
            R.SetOptimizerAsNelderMead()
        elif self._optimizer == "Powell":
            R.SetOptimizerAsPowell()
        elif self._optimizer == "Evolutionary":
            R.SetOptimizerAsEvolutionary()
        elif self._optimizer == "GradientDescent":
            # help(R.SetOptimizerAsGradientDescent)
            R.SetOptimizerAsGradientDescent(
                learningRate=self._optimizer_learningRate,
                numberOfIterations=self._optimizer_numberOfIterations,
            )
            R.SetOptimizerScalesFromIndexShift()
        elif self._optimizer == "GradientDescentLineSearch":
            # help(R.SetOptimizerAsGradientDescentLineSearch)
            R.SetOptimizerAsGradientDescentLineSearch(
                learningRate=self._optimizer_learningRate,
                numberOfIterations=self._optimizer_numberOfIterations,
            )
            R.SetOptimizerScalesFromIndexShift()
        elif self._optimizer == "RegularStepGradientDescent":
            # help(R.SetOptimizerAsRegularStepGradientDescent)
            R.SetOptimizerAsRegularStepGradientDescent(
                learningRate=self._optimizer_learningRate,
                numberOfIterations=self._optimizer_numberOfIterations,
                minStep=self._optimizer_minStep,
            )
            R.SetOptimizerScalesFromIndexShift()
        elif self._optimizer == "ConjugateGradientLineSearch":
            # help(R.SetOptimizerAsRegularStepGradientDescent)
            R.SetOptimizerAsConjugateGradientLineSearch(
                learningRate=self._optimizer_learningRate,
                numberOfIterations=self._optimizer_numberOfIterations,
            )
            R.SetOptimizerScalesFromIndexShift()
        elif self._optimizer == "L-BFGS-B":
            R.SetOptimizerAsLBFGSB()
        else:
            raise ValueError(f"Invalid optimizer: {self._optimizer}. Supported optimizers are: {SITK_OPTIMIZERS}")
        
    def setSampling(self, R):

        if self._sampling == "None":
            R.SetSamplingStrategy(sitk.sitkNone)
        elif self._sampling == "Random":
            R.SetSamplingStrategy(sitk.sitkRandom)
        elif self._sampling == "Regular":
            R.SetSamplingStrategy(sitk.sitkRegular)
        else:
            raise ValueError(f"Invalid sampling: {self._sampling}. Supported samplings are: {SITK_SAMPLING}")

    def setInitialTransform(self, R, dims, img_ref=None):

        displacementTx = None   
        if self._registration_type == "TranslationTransform":
            R.SetInitialTransform(sitk.TranslationTransform(dims))
        elif self._registration_type == "VersorTransform":
            R.SetInitialTransform(sitk.VersorTransform(dims))
        elif self._registration_type == "VersorRigid3DTransform":
            R.SetInitialTransform(sitk.VersorRigid3DTransform())
        elif self._registration_type == "Euler2DTransform":
            R.SetInitialTransform(sitk.Euler2DTransform())
        elif self._registration_type == "Euler3DTransform":
            R.SetInitialTransform(sitk.Euler3DTransform())
        elif self._registration_type == "Similarity2DTransform":
            R.SetInitialTransform(sitk.Similarity2DTransform())
        elif self._registration_type == "Similarity3DTransform":
            R.SetInitialTransform(sitk.Similarity3DTransform())
        elif self._registration_type == "ScaleTransform":
            R.SetInitialTransform(sitk.ScaleTransform(dims))
        elif self._registration_type == "ScaleVersor3DTransform":
            R.SetInitialTransform(sitk.ScaleVersor3DTransform())
        elif self._registration_type == "ScaleSkewVersor3DTransform":
            R.SetInitialTransform(sitk.ScaleSkewVersor3DTransform())
        elif self._registration_type == "ComposeScaleSkewVersor3DTransform":
            R.SetInitialTransform(sitk.ComposeScaleSkewVersor3DTransform())
        elif self._registration_type == "AffineTransform":
            R.SetInitialTransform(sitk.AffineTransform(dims))
        elif self._registration_type == "BSplineTransform":
            R.SetInitialTransform(sitk.BSplineTransform(dims))
        elif self._registration_type == "DisplacementFieldTransform":
            displacementField = sitk.Image(img_ref.GetSize(), sitk.sitkVectorFloat64)
            displacementTx = sitk.DisplacementFieldTransform(displacementField)
            displacementTx.SetSmoothingGaussianOnUpdate(
                varianceForUpdateField=0.0, varianceForTotalField=self._displacement_field_smoothing_sigmas
            )
            R.SetInitialTransform(displacementTx, inPlace=True)
        else:
            raise ValueError(f"Invalid registration type: {self._registration_type}. Supported types are: {SITK_TRANSFORMATIONS}")
        
        return displacementTx
    
    def setPyramid(self, R):
        R.SetShrinkFactorsPerLevel(self._pyramid_shrink_factor)
        R.SetSmoothingSigmasPerLevel(self._pyramid_smoothing_sigmas)

    def trnsf2array(self, R):

        if self._registration_type == "TranslationTransform":
            trnsf = np.eye(self._n_spatial+1)
            trnsf[:self._n_spatial,-1] = np.array(R.GetOffset())[::-1]
        elif self._registration_type == "VersorTransform":
            raise NotImplementedError("VersorTransform is not implemented yet.")
        elif self._registration_type == "VersorRigid3DTransform":
            raise NotImplementedError("VersorRigid3DTransform is not implemented yet.")
        elif self._registration_type == "Euler2DTransform" or self._registration_type == "Euler3DTransform":
            trnsf = np.eye(self._n_spatial+1)
            trnsf[:self._n_spatial,:self._n_spatial] = np.array(R.GetMatrix()).reshape((self._n_spatial, self._n_spatial))[:,::-1][::-1,:]
            trnsf[:self._n_spatial,-1] = np.array(R.GetTranslation())[::-1]
        elif self._registration_type == "Similarity2DTransform":
            raise NotImplementedError("Similarity2DTransform is not implemented yet.")
        elif self._registration_type == "Similarity3DTransform":
            raise NotImplementedError("Similarity3DTransform is not implemented yet.")
        elif self._registration_type == "ScaleTransform":
            raise NotImplementedError("ScaleTransform is not implemented yet.")
        elif self._registration_type == "ScaleVersor3DTransform":
            raise NotImplementedError("ScaleVersor3DTransform is not implemented yet.")
        elif self._registration_type == "ScaleSkewVersor3DTransform":
            raise NotImplementedError("ScaleSkewVersor3DTransform is not implemented yet.")
        elif self._registration_type == "ComposeScaleSkewVersor3DTransform":
            raise NotImplementedError("ComposeScaleSkewVersor3DTransform is not implemented yet.")
        elif self._registration_type == "AffineTransform":
            trnsf = np.eye(self._n_spatial+1)
            trnsf[:self._n_spatial,:self._n_spatial] = np.array(R.GetMatrix()).reshape((self._n_spatial, self._n_spatial))[:,::-1]
            trnsf[:self._n_spatial,-1] = np.array(trnsf.GetOffset())[::-1]
        elif self._registration_type == "BSplineTransform":
            raise NotImplementedError("BSplineTransform is not implemented yet.")
        elif self._registration_type == "DisplacementFieldTransform":
            # print(R.GetDisplacementField())
            df_image = R.GetDisplacementField()  # SimpleITK image

            # Get numpy array of displacements (shape: H x W x 2)
            displacement = sitk.GetArrayFromImage(df_image)  # shape: (rows, cols, 2)

            # Get physical coordinates of each pixel
            size = df_image.GetSize()  # (width, height)
            spacing = df_image.GetSpacing()
            origin = df_image.GetOrigin()
            direction = np.array(df_image.GetDirection()).reshape((2, 2))

            # Build grid of physical positions
            rows, cols = np.indices((size[1], size[0]))  # numpy uses (row, col)
            coords = np.stack([cols, rows], axis=-1)  # shape: (H, W, 2)

            # Convert to physical space
            coords = coords * spacing  # apply spacing
            coords = coords @ direction.T  # apply direction
            coords += origin  # apply origin

            # Flatten both coordinate and displacement fields
            start_points = coords.reshape(-1, 2)[:,::-1]
            vectors = -displacement.reshape(-1, 2)[:,::-1]

            # Build Napari vector array: (N, 2, 2)
            trnsf = np.stack([start_points, vectors], axis=1)
            # raise NotImplementedError("DisplacementFieldTransform is not implemented yet.")
        else:
            raise ValueError(f"Invalid registration type: {self._registration_type}. Supported types are: {SITK_TRANSFORMATIONS}")

        return trnsf
    
    def apply_trnsf(self, image, trnsf, scale, padding):

        img_float_sitk = sitk.GetImageFromArray(image.astype(np.float32))
        img_float_sitk.SetSpacing(scale[::-1])

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img_float_sitk)         # use ref image geometry
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(trnsf)                     # this maps img_float → img_ref
        out = resampler.Execute(img_float_sitk)           # apply T to float image

        img = sitk.GetArrayFromImage(out)

        return img

    def invert_trnsf(self, trnsf):
        if trnsf.GetName() == "DisplacementFieldTransform":
            inv_df = sitk.InvertDisplacementField(
                trnsf.GetDisplacementField(),
            )
            return sitk.DisplacementFieldTransform(inv_df)
        else:
            return trnsf.GetInverse()

    def apply_trnsf_to_points(self, points, trnsf):
        """
        Apply a SimpleITK transform to a list/array of points.
        
        Parameters:
        - points: (N, D) numpy array of N D-dimensional points (D=2 or 3)
        - trnsf: SimpleITK.Transform (can be affine, displacement field, etc.)
        
        Returns:
        - Transformed points as numpy array of shape (N, D)
        """
        trnsf_inv = self.invert_trnsf(trnsf)
        transformed_points = [trnsf_inv.TransformPoint(tuple(p[::-1])) for p in points]
        return np.array(transformed_points)[:,::-1]

    def write_trnsf(self, trnsf, out):
        if self._registration_type == "DisplacementFieldTransform":
            sitk.WriteTransform(trnsf, out+".hdf")
        else:
            sitk.WriteTransform(trnsf, out+".txt")

    def read_trnsf(self, out):
        if self._registration_type == "DisplacementFieldTransform":
            return sitk.ReadTransform(out+".hdf")
        else:
            return sitk.ReadTransform(out+".txt")

    def register(self, img_float, img_ref, scale, verbose=False):

        def command_iteration(method):
            """ Callback invoked when the optimization process is performing an iteration. """
            print(
                f"{method.GetOptimizerIteration():3} "
                + f"= {method.GetMetricValue():10.5f} "
                # + f": {method.GetOptimizerPosition()}"
            )

        img_float_sitk = sitk.GetImageFromArray(img_float.astype(np.float32))
        img_float_sitk.SetSpacing(scale[::-1])
        img_ref_sitk = sitk.GetImageFromArray(img_ref.astype(np.float32))
        img_ref_sitk.SetSpacing(scale[::-1])

        R = sitk.ImageRegistrationMethod()
        self.setMetric(R)
        self.setOptimizer(R)
        self.setPyramid(R)
        self.setInitialTransform(R, self._n_spatial, img_ref_sitk)
        R.SetInterpolator(sitk.sitkLinear)
        # displacementTx = self.setInitialTransform(R, self._n_spatial, img_ref_sitk)
        
        if verbose:
            R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

        trnsf = R.Execute(img_ref_sitk, img_float_sitk)

        # trnsf = self.trnsf2array(outTx)

        return trnsf

class RegistrationMoments(Registration):
    def __init__(self, n_axis, align_center=True, align_rotation=True):
        super().__init__()
        self.align_center = align_center
        self.align_rotation = align_rotation
        self._n_spatial = None
        self._n_axis = n_axis

    def register(self, img_float, img_ref, scale, verbose=False):
        """Compute the transformation (translation + rotation) to align img2 to img1, considering anisotropy."""

        nx = cp if GPU_AVAILABLE and USE_GPU else np 
        ndix = cndi if GPU_AVAILABLE and USE_GPU else ndi
        skmx = cskm if GPU_AVAILABLE and USE_GPU else skm
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

        return trnsf.get() if GPU_AVAILABLE and USE_GPU else trnsf
