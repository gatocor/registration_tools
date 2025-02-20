import os
import re
import numpy as np
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
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
from ..utils.auxiliar import _get_axis_scale, _make_index, _dict_axis_shape, _suppress_stdout_stderr, _shape_downsampled
import tempfile

def _get_vtImage(dataset, t, scale, axis, use_channel, downsample):
    img = dataset[_make_index(t, axis, use_channel, downsample)]
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
    return img

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

def load_registration(out):
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
    registration = Registration()
    for i,j in parameters.items():
        setattr(registration, i, j)
    registration._out = out
    return registration

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
        _physical_space (None): Placeholder for physical space.
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
            out=None, 
            registration_type="rigid", 
            perfom_global_trnsf=None, 
            pyramid_lowest_level=0, 
            pyramid_highest_level=3, 
            registration_direction="backward", 
            args_registration=""
        ):
        """
        Initialize the registration object with the specified parameters.

        Parameters:
        out (str, optional): Output path to save the registration results. Default is None.
        registration_type (str, optional): Type of registration to perform. Default is "rigid".
        perfom_global_trnsf (bool, optional): Whether to perform global transformation. Default is None.
        pyramid_lowest_level (int, optional): Lowest level of the pyramid for multi-resolution registration. Default is 0.
        pyramid_highest_level (int, optional): Highest level of the pyramid for multi-resolution registration. Default is 3.
        registration_direction (str, optional): Direction of registration, either "forward" or "backward". Default is "backward".
        args_registration (str, optional): Additional arguments for the registration process. Default is an empty string.
        """

        rigid_transformations = ["translation2D", "translation3D", "translation", "rigid2D", "rigid3D", "rigid"]
        registration_directions = ["forward", "backward"]

        self._registration_type = registration_type
        if self._registration_type in rigid_transformations and perfom_global_trnsf is None:
            self._perfom_global_trnsf = True
        elif perfom_global_trnsf is None:
            self._perfom_global_trnsf = False
        else:
            self._perfom_global_trnsf = perfom_global_trnsf
        self._pyramid_lowest_level = pyramid_lowest_level
        self._pyramid_highest_level = pyramid_highest_level
        if registration_direction not in registration_directions:
            raise ValueError(f"registration_direction must be either of the following: {registration_directions}")
        self._registration_direction = registration_direction
        self._args_registration = args_registration

        self._out = None
        self._physical_space = None
        self._n_spatial = None
        self._fitted = False
        self._box = None
        self._t_max = None
        self._origin = None
        self._transfs = {}
        self._failed = {}

        if out is not None:
            self.save(out=out)
        
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
                trnsf.write(f"{self._out}/trnsf_global/{name}.trnsf")
            elif "relative" in name:
                trnsf.write(f"{self._out}/trnsf/{name}.trnsf")

    def load(self, out):
        """
        Loads a registration object from the specified path.

        Args:
            out (str): The path to the registration object.
        """
        if not os.path.exists(f"{out}/parameters.json"):
            raise ValueError("The registration object does not exist.")

        with open(f"{out}/parameters.json", "r") as f:
            parameters = json.load(f)
        self.__dict__.update(parameters)
        self._out = out

    def _create_folder(self):

        os.makedirs(f"{self._out}", exist_ok=True)
        os.makedirs(f"{self._out}/trnsf_relative", exist_ok=True)
        if self._perfom_global_trnsf:
            os.makedirs(f"{self._out}/trnsf_global", exist_ok=True)

    def _save_transformation_relative(self, trnsf, pos_float, pos_ref):
        if self._out is None:
            self._transfs[f"trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf"] = trnsf
        else:
            trnsf.write(f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf")

    def _save_transformation_global(self, trnsf, pos_float, pos_ref):
        if self._out is None:
            self._transfs[f"trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf"] = trnsf
        else:
            trnsf.write(f"{self._out}/trnsf_global/trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf")

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
                return vt.vtTransformation(f"{self._out}/trnsf_relative/trnsf_relative_{pos_float:04d}_{pos_ref:04d}.trnsf")

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
                return vt.vtTransformation(f"{self._out}/trnsf_global/trnsf_global_{pos_float:04d}_{pos_ref:04d}.trnsf")

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

        return registration_args

    def fit(self, dataset, use_channel=None, axis=None, scale=None, downsample=None, save_behavior="Continue", verbose=False):
        """
        Registers a dataset and saves the results to the specified path.
        """

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
        self._box = tuple([int(i) for i in np.array(dataset.shape)[[axis.index(ax) for ax in axis if ax in "XYZ"]] * np.array(new_scale)])

        # Setup arguments
        registration_args = self._make_registration_args(verbose=verbose)

        # Suppress skimage warnings for low contrast images
        warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image.*")

        # Set registration direction
        self._t_max = _dict_axis_shape(axis, dataset.shape)["T"]
        if self._registration_direction == "backward":
            origin = 0
            iterator = zip(
                range(self._t_max-1),
                range(1,self._t_max)
            )
        else:
            origin = self._t_max - 1
            iterator = zip(
                range(self._t_max-1, 0, -1),
                range(self._t_max-2, -1, -1)
            )
        self._origin = origin

        img_ref = None
        trnsf_global = None
        for pos_ref, pos_float in tqdm(iterator, desc=f"Registering images using channel {use_channel}", unit="", total=self._t_max-1):

            if self._trnsf_exists_relative(pos_float, pos_ref):
                if save_behavior == "NotOverwrite":
                    raise ValueError("The relative transformation already exists. If you want to overwrite it, set save_behavior='Overwrite'.")
                elif save_behavior == "Continue":
                    None
            else:
                if img_ref is None:
                    img_ref = _get_vtImage(dataset, pos_ref, scale, axis, use_channel, downsample)
                else:
                    img_ref = img_float

                img_float = _get_vtImage(dataset, pos_float, scale, axis, use_channel, downsample)

                keep = True
                counter = 0
                failed = False
                while keep and counter < 10:
                    # Usage
                    if verbose:
                        trnsf = vt.blockmatching(img_float, image_ref=img_ref, params=registration_args)
                    else:
                        with _suppress_stdout_stderr():
                            trnsf = vt.blockmatching(img_float, image_ref=img_ref, params=registration_args)

                    if trnsf is None:
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
                                if verbose:
                                    print(f"Registration failed between positions {pos_float} and {pos_ref}. Setting it to identity and continuing.")
                                trnsf = vt.vtTransformation(np.eye(self._n_spatial+1))
                                keep = False
                    else:
                        if failed:
                            self._failed[f"{pos_float:04d}_{pos_ref:04d}"] = registration_args
                        registration_args = self._make_registration_args(verbose=verbose)
                        keep = False

                self._save_transformation_relative(trnsf, pos_float, pos_ref)

            if self._perfom_global_trnsf:

                if self._trnsf_exists_global(pos_float, origin):
                    if save_behavior == "NotOverwrite":
                        raise ValueError("The global transformation already exists. If you want to overwrite it, set save_behavior='Overwrite'.")
                    elif save_behavior == "Continue":
                        None
                else:
                    if trnsf_global is None and pos_ref != origin:
                        trnsf_global = self._load_transformation_global(pos_ref, origin)
                    elif trnsf_global is None:
                        trnsf_global = trnsf
                        self._save_transformation_global(trnsf_global, pos_float, origin)
                    else:
                        trnsf_global = vt.compose_trsf([
                            trnsf_global,
                            trnsf
                        ])
                        self._save_transformation_global(trnsf_global, pos_float, origin)

        self._fitted = True
        if self._out is not None:
            self.save()

    def apply(self, dataset, out=None, axis=None, scale=None, downsample=None, save_behavior="Continue", transformation="global", verbose=False, **kwargs):
        """
        Registers a dataset and saves the results to the specified path.
        """

        save_behaviors = ["NotOverwrite", "Continue"]

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
        new_shape = _shape_downsampled(dataset.shape,axis,downsample)

        # Suppress skimage warnings for low contrast images
        warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image.*")
        
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
        elif isinstance(out, str) and out.endswith(".zarr"):
            data = zarr.create_array(
                store=out,
                shape=new_shape,
                dtype=dataset.dtype,
                **kwargs
            )
            data.attrs["axis"] = axis
            data.attrs["scale"] = new_scale
        else:
            raise ValueError("The output must be None or a the name to a zarr file.")

        if self._registration_direction == "backward":
            origin = 0
            iterator = range(1, new_shape[np.where([i == "T" for i in axis])[0][0]])
            if "C" in axis:
                for ch in range(new_shape[np.where([i == "C" for i in axis])[0][0]]):
                    img = _get_vtImage(dataset, 0, new_scale, axis, ch, downsample)
                    data[_make_index(0, axis, ch)] = img.copy_to_array()
            else:
                data[_make_index(0, axis, None)] = dataset[_make_index(0, axis, None, downsample)]
        else:
            origin = new_shape[np.where([i == "T" for i in axis])[0][0]] - 1
            iterator = range(new_shape[np.where([i == "T" for i in axis])[0][0]] - 1, 0, -1)
            if "C" in axis:
                for ch in range(new_shape[np.where([i == "C" for i in axis])[0][0]]):
                    img = _get_vtImage(dataset, origin, new_scale, axis, ch, downsample)
                    data[_make_index(new_shape[np.where([i == "T" for i in axis])[0][0]] - 1, axis, ch)] = img.copy_to_array()
            else:
                data[_make_index(new_shape[np.where([i == "T" for i in axis])[0][0]] - 1, axis, None)] = dataset[_make_index(dataset.shape[np.where([i == "T" for i in axis])[0][0]] - 1, axis, None, downsample)]

        for t in tqdm(iterator, desc=f"Applying registration to images", unit="", total=self._t_max-1):
            if transformation == "global":
                if not self._trnsf_exists_global(t, origin):
                    raise ValueError("The global transformation does not exist.")
                trnsf = self._load_transformation_global(t, origin)
            else:
                if not self._trnsf_exists_relative(t, origin):
                    raise ValueError("The relative transformation does not exist.")
                trnsf = self._load_transformation_relative(t, origin)

            if "C" in axis:
                for ch in range(new_shape[np.where([i == "C" for i in axis])[0][0]]):
                    img = _get_vtImage(dataset, t, new_scale, axis, ch, downsample)
                    data[_make_index(t, axis, ch)] = vt.apply_trsf(img, trnsf).copy_to_array()
            else:
                img = _get_vtImage(dataset, t, new_scale, axis, None, downsample)
                data[_make_index(t, axis, None)] = vt.apply_trsf(img, trnsf).copy_to_array()

        return data

    def fit_apply(self, dataset, out=None, use_channel=None, axis=None, scale=None, downsample=None, save_behavior="Continue", transformation="global", verbose=False):

        self.fit(dataset, use_channel=use_channel, axis=axis, scale=scale, downsample=downsample, save_behavior=save_behavior, verbose=verbose)
        return self.apply(dataset, out=out, axis=axis, scale=scale, downsample=downsample, save_behavior=save_behavior, transformation=transformation, verbose=verbose)

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