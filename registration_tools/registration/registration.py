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
import atexit
from tqdm import tqdm
import time
import copy
import warnings  # Add this import
from copy import deepcopy
import zarr
from ..utils.auxiliar import _get_axis_scale, _make_index, _dict_axis_shape, _suppress_stdout_stderr, _shape_downsampled, _shape_padded
import tempfile
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cndi
    import cucim.skimage as cskm
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class Registration:
    """
    A class to perform image registration.

    Parameters:
        out (str, optional): Output path to save the registration results. Default is None.
        registration_type (str, optional): Type of registration to perform. Default is "rigid".
        perfom_global_trnsf (bool, optional): Whether to perform global transformation. Default is None.
        pyramid_lowest_level (int, optional): Lowest level of the pyramid for multi-resolution registration. Default is 0.
        pyramid_highest_level (int, optional): Highest level of the pyramid for multi-resolution registration. Default is 3.
        registration_direction (str, optional): Direction of registration, either "forward" or "backward". Default is "backward".
        args_registration (str, optional): Additional arguments for the registration process. Default is an empty string.

    Attributes:
        registration_type (str): The type of registration to perform.
        perfom_global_trnsf (bool): Whether to perform global transformation.
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

        self._perfom_global_trnsf = None
        self._registration_direction = None
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

        self._create_folder()

        metadata = deepcopy(self.__dict__)
        del metadata["_out"]
        del metadata["_transfs"]
        metadata_path = os.path.join(self._out, "parameters.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        for name, trnsf in self._transfs.items():
            if "global" in name:
                self.write_trnsf(f"{self._out}/trnsf_global/{name}.trnsf")
            elif "relative" in name:
                self.write_trnsf(f"{self._out}/trnsf/{name}.trnsf")

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

    def _create_folder(self):

        os.makedirs(f"{self._out}", exist_ok=True)
        os.makedirs(f"{self._out}/trnsf_relative", exist_ok=True)
        if self._perfom_global_trnsf:
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
            if f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf" not in self._transfs:
                raise ValueError(f"The transformation between {pos_float:04d} and {pos_ref:04d} the specified positions does not exist.")
            else:
                return self._transfs[f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf"]
        else:
            if not os.path.exists(f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf"):
                raise ValueError("The transformation between the specified positions does not exist.")
            else:
                return self.read_trnsf(f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf")

    def _load_transformation_global(self, pos_float, pos_ref):
        if self._out is None:
            if f"trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf" not in self._transfs:
                raise ValueError("The transformation between the specified positions does not exist.")
            else:
                return self._transfs[f"trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf"]
        else:
            if not os.path.exists(f"{self._out}/trnsf_global/trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf"):
                raise ValueError("The transformation between the specified positions does not exist.")
            else:
                return self.read_trnsf(f"{self._out}/trnsf_global/trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf")

    def _trnsf_exists_relative(self, pos_float, pos_ref):
        if self._out is None:
            return f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf" in self._transfs.keys()   
        else:
            return os.path.exists(f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf")
    
    def _trnsf_exists_global(self, pos_float, pos_ref):
        if self._out is None:
            return f"trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf" in self._transfs.keys()
        else:
            return os.path.exists(f"{self._out}/trnsf_global/trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf")
        
    def _padding_box_to_points(self, padding_reference, scale):
        return np.array(np.meshgrid(*[np.array(i) for i in padding_reference])).T.reshape(-1, self._n_spatial) * np.array(scale)

    def _points_to_padding_box(self, points, scale):
        points_scaled = (points / np.array(scale)).astype(int)
        box = []
        for i in range(points_scaled.shape[1]):
            box.append((points_scaled[:, i].min(), points_scaled[:, i].max()))
        return box
        
    def fit(self, dataset, out=None, direction="backward", perfom_global_trnsf=False, use_channel=None, axis=None, scale=None, downsample=None, stepping=1, save_behavior="Continue", verbose=False):
        """
        Registers a dataset and saves the results to the specified path.
        """

        self._out = out
        if self._out is not None:
            self._create_folder()
            self.save()
        if direction not in ["backward", "forward"]:
            raise ValueError("The direction must be either 'backward' or 'forward'.")
        self._registration_direction = direction
        self._perfom_global_trnsf = perfom_global_trnsf

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

        if self._out is not None:
            self.save()

        img_ref = None
        trnsf_global = None
        pos_ref_current = None
        for pos_ref, pos_float in tqdm(iterator, desc=f"Registering images using channel {use_channel}", unit="", total=self._t_max-1):

            if self._trnsf_exists_relative(pos_float, pos_ref):
                if save_behavior == "NotOverwrite":
                    raise ValueError("The relative transformation already exists. If you want to overwrite it, set save_behavior='Overwrite'.")
                elif save_behavior == "Continue":
                    None
            else:
                if img_ref is None or pos_ref_current != pos_ref:
                    img_ref = self.get_image(dataset, pos_ref, scale, axis, use_channel, downsample)
                else:
                    img_ref = img_float

                img_float = self.get_image(dataset, pos_float, scale, axis, use_channel, downsample)

                trnsf = self.register(img_float, img_ref, scale, verbose=verbose)
                # if failed:
                #     self._failed.append(f"{pos_float:04d}_{pos_ref:04d}")

                self._save_transformation_relative(trnsf, pos_float, pos_ref)

            if self._perfom_global_trnsf:

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

    def apply(self, dataset, out=None, axis=None, scale=None, downsample=None, save_behavior="Continue", transformation="global", padding=False, verbose=False, **kwargs):
        """
        Registers a dataset and saves the results to the specified path.
        """

        save_behaviors = ["NotOverwrite", "Overwrite", "Continue"]

        if not self._fitted:
            raise ValueError("The registration object has not been fitted. Please fit the registration object before applying it.")

        # Check inputs
        axis, scale = _get_axis_scale(dataset, axis, scale)
        
        if len(axis) != len(dataset.shape):
            raise ValueError("The axis must have the same length as the dataset shape.")
        
        if "T" not in axis:
            raise ValueError("The axis must contain the time dimension 'T'.")
                        
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

        # Shape
        if padding:
            padded_shape = _shape_padded(dataset.shape, axis, self._padding_box)
        else:
            padded_shape = dataset.shape
        new_shape = _shape_downsampled(padded_shape, axis, downsample)
        
        # Setup output
        if out is None:
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
        elif isinstance(out, str):# and out.endswith(".zarr"):
            if os.path.exists(out) and save_behavior == "NotOverwrite":
                raise ValueError("The output file already exists. If you want to overwrite it, set save_behavior='Overwrite'.")
            elif os.path.exists(out) and save_behavior == "Continue":
                out_path, out_file = os.path.split(out)
                data = zarr.open_array(out_file, path=out_path)
                if "computed" not in data.attrs:
                    data.attrs["computed"] = []
            elif os.path.exists(out) and save_behavior == "Overwrite":
                shutil.rmtree(out)
                data = zarr.create_array(
                    store=out,
                    shape=new_shape,
                    dtype=dataset.dtype,
                    **kwargs
                )
                data.attrs["axis"] = axis
                data.attrs["scale"] = new_scale
                data.attrs["computed"] = []
            else:
                data = zarr.create_array(
                    store=out,
                    shape=new_shape,
                    dtype=dataset.dtype,
                    **kwargs
                )
                data.attrs["axis"] = axis
                data.attrs["scale"] = new_scale
                data.attrs["computed"] = []
        else:
            raise ValueError("The output must be None or a the name to a zarr file.")

        mt = np.eye(4)
        # if padding:
        #     padding_drift = [i[0]*j for i,j in zip(self._padding_box, scale)][::-1]
        #     mt[:3,3] = np.array(padding_drift)
        #     padding_trnsf = vt.vtTransformation(mt)

        if self._registration_direction == "backward":
            origin = 0
            iterator = range(1, new_shape[np.where([i == "T" for i in axis])[0][0]])
        else:
            origin = new_shape[np.where([i == "T" for i in axis])[0][0]] - 1
            iterator = range(new_shape[np.where([i == "T" for i in axis])[0][0]] - 2, -1, -1)

        if "C" in axis:
            for ch in range(new_shape[np.where([i == "C" for i in axis])[0][0]]):
                img = self.get_image(dataset, origin, new_scale, axis, ch, downsample)
                if padding:
                    im = self.apply_trnsf(img, padding_trnsf)
                else:
                    im = img
                data[_make_index(origin, axis, ch)] = self.image2array(im)
        else:
            data[_make_index(origin, axis, None)] = dataset[_make_index(dataset.shape[np.where([i == "T" for i in axis])[0][0]] - 1, axis, None, downsample)]

        for t in tqdm(iterator, desc=f"Applying registration to images", unit="", total=self._t_max-1):
            #Skip is computed
            if t in data.attrs["computed"]:
                    continue

            if transformation == "global":
                if not self._trnsf_exists_global(t, origin):
                    print(t, origin)
                    raise ValueError("The global transformation does not exist.")
                trnsf = self._load_transformation_global(t, origin)
            else:
                if not self._trnsf_exists_relative(t, origin):
                    raise ValueError("The relative transformation does not exist.")
                trnsf = self._load_transformation_relative(t, origin)

            if "C" in axis:
                for ch in range(new_shape[np.where([i == "C" for i in axis])[0][0]]):
                    img = self.get_image(dataset, t, new_scale, axis, ch, downsample)
                    if padding:
                        joint_trnsf = self.compose_trsf([trnsf,padding_trnsf])
                    else:
                        joint_trnsf = trnsf
                    im_trnsf = self.apply_trnsf(img, joint_trnsf)
                    data[_make_index(t, axis, ch)] = self.image2array(im_trnsf)
            else:
                img = self.get_image(dataset, t, new_scale, axis, None, downsample)
                if padding:
                    joint_trnsf = self.compose_trsf([trnsf,padding_trnsf])
                else:
                    joint_trnsf = trnsf
                im = self.apply_trsf(img, joint_trnsf)
                data[_make_index(t, axis, None)] = self.image2array(im)

            data.attrs["computed"].append(t)

        return data

    def fit_apply(self, dataset, out_trnsf=None, out_dataset=None, direction="backward", use_channel=None, axis=None, scale=None, downsample=None, stepping=1, perfom_global_trnsf=False, save_behavior="Continue", verbose=False, padding=False, **kwargs):
        """
        Registers a dataset and saves the results to the specified path.
        """

        self._out = out_trnsf
        if self._out is not None:
            self._create_folder()
        if direction not in ["backward", "forward"]:
            raise ValueError("The direction must be either 'backward' or 'forward'.")
        self._registration_direction = direction
        self._perfom_global_trnsf = perfom_global_trnsf

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
        new_shape = _shape_downsampled(dataset.shape, axis, downsample)

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
                data[_make_index(self._origin, axis, ch)] = dataset[_make_index(self._origin, axis, ch, downsample=downsample)]
        else:
            data[_make_index(self._origin, axis, None)] = dataset[_make_index(self._origin, axis, None, downsample=downsample)]

        # Save parameters in the registration folder
        if self._out is not None:
            self.save()

        # Loop over the images for registration
        img_ref = None
        trnsf_global = None
        pos_ref_current = None
        for pos_ref, pos_float in tqdm(iterator, desc=f"Registering images using channel {use_channel}", unit="", total=self._t_max-1):

            if self._trnsf_exists_relative(pos_float, pos_ref) and save_behavior != "Overwrite":
                if save_behavior == "NotOverwrite":
                    raise ValueError("The relative transformation already exists. If you want to overwrite it, set save_behavior='Overwrite'.")
                elif save_behavior == "Continue":
                    None
            else:
                if img_ref is None or pos_ref_current != pos_ref:
                    img_ref = self.get_image(dataset, pos_ref, scale, axis, use_channel, downsample)
                else:
                    img_ref = img_float

                img_float = self.get_image(dataset, pos_float, scale, axis, use_channel, downsample)

                trnsf = self.register(img_float, img_ref, scale, verbose=verbose)

                self._save_transformation_relative(trnsf, pos_float, pos_ref)

            if self._perfom_global_trnsf:

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

            if perfom_global_trnsf:
                trnsf_img = trnsf_global
            else:
                trnsf_img = trnsf

            if "C" in axis:
                for ch in range(new_shape[np.where([i == "C" for i in axis])[0][0]]):
                    img_ = self.get_image(dataset, pos_float, new_scale, axis, ch, downsample)
                    im_trnsf = self.apply_trnsf(img_, trnsf_img)
                    data[_make_index(pos_float, axis, ch)] = self.image2array(im_trnsf)
            else:
                im = self.apply_trsf(img_float, trnsf_img)
                data[_make_index(pos_float, axis, None)] = self.image2array(im)

            pos_ref_current = pos_float
        
        self._padding_box = self._padding_box[::-1]
        self._fitted = True
        if self._out is not None:
            self.save()

        if padding:
            print("Padding data")
            return self.apply(data, axis=axis, scale=scale, downsample=downsample, save_behavior="Overwrite", perfom_global_trnsf=perfom_global_trnsf, padding=padding, verbose=verbose, **kwargs)
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
        perfom_global_trnsf (bool, optional): Whether to perform global transformation. Default is None.
        pyramid_lowest_level (int, optional): Lowest level of the pyramid for multi-resolution registration. Default is 0.
        pyramid_highest_level (int, optional): Highest level of the pyramid for multi-resolution registration. Default is 3.
        registration_direction (str, optional): Direction of registration, either "forward" or "backward". Default is "backward".
        args_registration (str, optional): Additional arguments for the registration process. Default is an empty string.

    Attributes:
        registration_type (str): The type of registration to perform.
        perfom_global_trnsf (bool): Whether to perform global transformation.
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

        rigid_transformations = ["translation2D", "translation3D", "translation", "rigid2D", "rigid3D", "rigid", "rotation2D", "rotation3D", "rotation"]

        if registration_type not in rigid_transformations:
            raise ValueError(f"Invalid registration type: {registration_type}. Supported types are: {rigid_transformations}")
        self._registration_type = registration_type
        self._pyramid_lowest_level = pyramid_lowest_level
        self._pyramid_highest_level = pyramid_highest_level
        self._args_registration = args_registration

        
    def get_image(self, dataset, t, scale, axis, use_channel, downsample, padding=None):
        img = dataset[_make_index(t, axis, use_channel, downsample)]
        # if padding is not None:
        #     padding_ = [(-p[0]//d, (p[1]-s)//d) for p,s,d in zip(padding, img.shape, downsample)]
        #     # padding_ = [(0, (p[1]-s)//d-p[0]//d) for p,s,d in zip(padding, img.shape, downsample)]
        #     img = np.pad(img, padding_, mode="constant")

        # Suppress skimage warnings for low contrast images
        warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image.*")

        try:
            with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as temp_file:
                imsave(temp_file.name, img)
                img = vt.vtImage(temp_file.name)
                # img = vt.vtImage(img.copy()) #Ideal but runs all the time in problems
                img.setSpacing(scale[::-1])
        except: #In case you do not have access permissions
            temp_file = f"temp_{np.random.randint(0,1E6)}.tiff"
            imsave(temp_file, img)
            img = vt.vtImage(temp_file)
            # img = vt.vtImage(img.copy()) #Ideal but runs all the time in problems
            img.setSpacing(scale[::-1])
            os.remove(temp_file)

        # Activate warnings again
        warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image.*")

        return img
    
    def write_trnsf(self, trnsf, out):
        trnsf.write(out)

    def read_trnsf(self, out):
        return vt.vtTransformation(out)
    
    def apply_trnsf(self, img, trnsf):
        return vt.apply_trsf(img, trnsf)
    
    def apply_trnsf_to_points(self, points, trnsf):
        points = vt.vtPointList(points)
        return vt.apply_trsf_to_points(points, trnsf).copy_to_array()

    def image2array(self, img):
        return img.copy_to_array()
    
    def compose_trnsf(self, trnsfs):
        return vt.compose_trsf(trnsfs)
    
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

        keep = True
        counter = 0
        registration_args = self._make_registration_args(verbose=verbose)
        while keep and counter < 10:
            # Usage
            if verbose:
                trnsf = vt.blockmatching(img_float, image_ref=img_ref, params=registration_args)
            else:
                with _suppress_stdout_stderr():
                    trnsf = vt.blockmatching(img_float, image_ref=img_ref, params=registration_args)
            if trnsf is None:
                trnsf = vt.vtTransformation(np.eye(self._n_spatial+1))
                failed = True
                counter += 1
                if self._pyramid_highest_level > 0:
                    if verbose:
                        print("Registration failed. Trying with a lower highest pyramid level.")
                    pyramid_highest_level = self._pyramid_highest_level - 1
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

    def get_image(self, dataset, t, scale, axis, use_channel, downsample, padding=None):
        return dataset[_make_index(t, axis, use_channel, downsample)]

    def write_trnsf(self, trnsf, out):
        np.savetxt(out, trnsf, fmt='%.6f')

    def read_trnsf(self, out):
        return np.loadtxt(out)

    def apply_trnsf(self, image, trnsf):

        nx = cp if GPU_AVAILABLE else np 
        ndix = cndi if GPU_AVAILABLE else ndi

        # Extract rotation and translation
        rotation_matrix = nx.array(trnsf[:3, :3])
        translation = nx.array(trnsf[:3, 3])

        # Apply the affine transformation to align img1 to img2
        transformed_img = ndix.affine_transform(image, rotation_matrix, offset=translation)

        if GPU_AVAILABLE:
            return transformed_img.get()
        else:
            return transformed_img

    def image2array(self, img):
        return img
    
    def compose_trnsf(self, trnsfs):
        return np.linalg.multi_dot(trnsfs)

    def register(self, img_float, img_ref, scale, verbose=False):
        """Compute the transformation (translation + rotation) to align img2 to img1, considering anisotropy."""

        nx = cp if GPU_AVAILABLE else np 
        ndix = cndi if GPU_AVAILABLE else ndi
        skmx = cskm if GPU_AVAILABLE else skm

        # Compute center of the images in real-world coordinates
        center_image = (nx.array(img_float.shape) - 1) / 2 * nx.array(scale)

        center1 = skmx.centroid(img_ref, spacing=scale)  # Center of img1 in real-world space

        center2 = skmx.centroid(img_float, spacing=scale)  # Center of img2 in real-world space

        # Compute translation to move img2 to img1
        translation = center2 - center1 if self.align_center else np.zeros(3)

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

                # Rodrigues' rotation formula
                K = np.array([
                    [0, -normal[2], normal[1]],
                    [normal[2], 0, -normal[0]],
                    [-normal[1], normal[0], 0]
                ])
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

        # Scaling matrices (fixing order of operations)
        scale_aff = nx.eye(4)
        scale_aff[:3,:3] = nx.diag(scale)
        
        scale_inv_aff = nx.eye(4)
        scale_inv_aff[:3,:3] = nx.diag(1/nx.array(scale))

        # Apply transformation in the correct order
        trnsf = scale_inv_aff @ translation_aff_center @ translation_inv_aff @ rotation_matrix_aff @ translation_aff @ scale_aff

        if GPU_AVAILABLE:
            return trnsf.get()
        else:
            return trnsf  # Return as a 4×4 affine transformation matrix
