import os
import re
import numpy as np
from skimage.io import imread, imsave
import shutil
import vt
from ..dataset import Dataset, load_dataset
import json
import atexit
from tqdm import tqdm
import warnings  # Add this import

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

def register(
    dataset,
    save_path,
    use_channel = 0,
    numbers = None,
    save_behavior = "Continue",
    perfom_global_trnsf = None,
    apply_registration = True,
    registration_type = "rigid",
    pyramid_lowest_level = 0,
    pyramid_highest_level = 3,
    registration_direction = None,
    padding = True,
    downsample = None,
    verbose = False,
    debug = 0,
    args_registration = "",
    plot_old_projections = True,
    plot_projections = True,
    make_vectorfield = None,
    vectorfield_threshold = 1.,
    vectorfield_spacing = 1,
):
    """
    Registers a dataset and saves the results to the specified path.
    
    Parameters:
        dataset (Dataset): The dataset to be registered. Must be an instance of the Dataset class.
        save_path (str): The directory where the registration results will be saved.
        use_channel (int, optional): The channel to use for registration. Default is 0.
        numbers (list, optional): List of numbers to register. If None, all numbers in the dataset will be used. Default is None.
        save_behavior (str, optional): Behavior for saving files. Options are "NotOverwrite", "Continue", "Overwrite". Default is "Continue".
        perfom_global_trnsf (bool, optional): Whether to perform global transformation. Default is None.
        apply_registration (bool, optional): Whether to apply the registration. Default is True.
        registration_type (str, optional): Type of registration to perform. Default is "rigid". Select among:
            translation2D, translation3D, translation-scaling2D, translation-scaling3D,
            rigid2D, rigid3D, rigid, similitude2D, similitude3D, similitude,
            affine2D, affine3D, affine, vectorfield2D, vectorfield3D, vectorfield, vector

        pyramid_lowest_level (int, optional): pyramid lowest level. Default is 0.
        pyramid_highest_level: pyramid highest level. Default is 3: it corresponds to 32x32x32 for an original 256x256x256 image.
        registration_direction (str, optional): Direction of registration. Options are "forward", "backward". Default is None.
        padding (bool, optional): Whether to apply padding. Default is True.
        downsample (tuple, optional): Downsample factor for the images. Default is None.
        debug (int, optional): Debug level. Default is 1.
        args_registration (str, optional) : Additional arguments for the blockmatching algorithm. These can be (from vt-python documentation):
    
    Parameters for the blockmatching algorithm:
        ### image geometry ### 
        [-reference-voxel %lf %lf [%lf]]:
        changes/sets the voxel sizes of the reference image
        [-floating-voxel %lf %lf [%lf]]:
        changes/sets the voxel sizes of the floating image
        ### pre-processing ###
        [-normalisation|-norma|-rescale] # input images are normalized on one byte
        before matching (this may be the default behavior)
        [-no-normalisation|-no-norma|-no-rescale] # input images are not normalized on
        one byte before matching
        ### post-processing ###
        [-no-composition-with-left] # the written result transformation is only the
        computed one, ie it is not composed with the left/initial one (thus does not allow
        to resample the floating image if an left/initial transformation is given) [default]
        [-composition-with-left] # the written result transformation is the
        computed one composed with the left/initial one (thus allows to resample the
        floating image if an left/initial transformation is given) 
        ### pyramid building ###
        [-pyramid-gaussian-filtering | -py-gf] # before subsampling, the images 
        are filtered (ie smoothed) by a gaussian kernel.
        ### block geometry (floating image) ###
        -block-size|-bl-size %d %d %d       # size of the block along X, Y, Z
        -block-spacing|-bl-space %d %d %d   # block spacing in the floating image
        -block-border|-bl-border %d %d %d   # block borders: to be added twice at
        each dimension for statistics computation
        ### block selection ###
        [-floating-low-threshold | -flo-lt %d]     # values <= low threshold are not
        considered
        [-floating-high-threshold | -flo-ht %d]    # values >= high threshold are not
        considered
        [-floating-removed-fraction | -flo-rf %f]  # maximal fraction of removed points
        because of the threshold. If too many points are removed, the block is
        discarded
        [-reference-low-threshold | -ref-lt %d]    # values <= low threshold are not
        considered
        [-reference-high-threshold | -ref-ht %d]   # values >= high threshold are not
        considered
        [-reference-removed-fraction | -ref-rf %f] # maximal fraction of removed points
        because of the threshold. If too many points are removed, the block is
        discarded
        [-floating-selection-fraction[-ll|-hl] | -flo-frac[-ll|-hl] %lf] # fraction of
        blocks from the floating image kept at a pyramid level, the blocks being
        sorted w.r.t their variance (see note (1) for [-ll|-hl])
        ### pairing ###
        [-search-neighborhood-half-size | -se-hsize %d %d %d] # half size of the search
        neighborhood in the reference when looking for similar blocks
        [-search-neighborhood-step | -se-step %d %d %d] # step between blocks to be
        tested in the search neighborhood
        [-similarity-measure | -similarity | -si [cc|ecc|ssd|sad]]  # similarity measure
        cc: correlation coefficient
        ecc: extended correlation coefficient
        [-similarity-measure-threshold | -si-th %lf]    # threshold on the similarity
        measure: pairings below that threshold are discarded
        ### transformation regularization ###
        [-elastic-regularization-sigma[-ll|-hl] | -elastic-sigma[-ll|-hl]  %lf %lf %lf]
        # sigma for elastic regularization (only for vector field) (see note (1) for
        [-ll|-hl])
        ### transformation estimation ###
        [-estimator-type|-estimator|-es-type %s] # transformation estimator
        wlts: weighted least trimmed squares
        lts: least trimmed squares
        wls: weighted least squares
        ls: least squares
        [-lts-cut|-lts-fraction %lf] # for trimmed estimations, fraction of pairs that are kept
        [-lts-deviation %lf] # for trimmed estimations, defines the threshold to discard
        pairings, ie 'average + this_value * standard_deviation'
        [-lts-iterations %d] # for trimmed estimations, the maximal number of iterations
        [-fluid-sigma|-lts-sigma[-ll|-hl] %lf %lf %lf] # sigma for fluid regularization,
        ie field interpolation and regularization for pairings (only for vector field)
        (see note (1) for [-ll|-hl])
        [-vector-propagation-distance|-propagation-distance|-pdistance %f] # 
        distance propagation of initial pairings (ie displacements)
        this implies the same displacement for the spanned sphere
        (only for vectorfield)
        [-vector-fading-distance|-fading-distance|-fdistance %f] # 
        area of fading for initial pairings (ie displacements)
        this allows progressive transition towards null displacements
        and thus avoid discontinuites
        ### end conditions for matching loop ###
        [-max-iteration[-ll|-hl]|-max-iterations[-ll|-hl]|-max-iter[-ll|-hl] %d]|...
        ...|-iterations[-ll|-hl] %d]   # maximal number of iteration
        (see note (1) for [-ll|-hl])
        [-corner-ending-condition|-rms] # evolution of image corners
        ### filter type ###
        [-gaussian-filter-type|-filter-type deriche|fidrich|young-1995|young-2002|...
        ...|gabor-young-2002|convolution] # type of filter for image/vector field
        smoothing
        ### misc writing stuff ###
        [-default-filenames|-df]     # use default filename names
        [-no-default-filenames|-ndf] # do not use default filename names
        [-command-line %s]           # write the command line
        [-logfile %s]                # write some output in this logfile
        [-vischeck]  # write an image with 'active' blocks
        [-write_def] # id. 
        ### parallelism ###
        [-parallel|-no-parallel] # use parallelism (or not)
        [-parallelism-type|-parallel-type default|none|openmp|omp|pthread|thread]
        [-max-chunks %d] # maximal number of chunks
        [-parallel-scheduling|-ps default|static|dynamic-one|dynamic|guided] # type
        of scheduling for open mp
        ### general parameters ###
        -verbose|-v: increase verboseness parameters being read several time, use '-nv -v -v ...' to set the verboseness level
        -debug|-D: increase debug level
        -no-debug|-nodebug: no debug indication
        -trace:
        -no-trace:
        -print-parameters|-param:
        -print-time|-time:
        -no-time|-notime:
        -trace-memory|-memory: keep trace of allocated pointers (in instrumented procedures)
        display some information about memory consumption
        Attention: it disables the parallel mode, because of concurrent access to memory parallel mode may be restored by specifying '-parallel' after '-memory' but unexpected crashes may be experienced
        -no-memory|-nomemory:
        -h: print option list
        -help: print option list + details
        
        Notes
        (1) If -ll or -hl are respectively added to the option, this specifies only the
        value for respectively the lowest or the highest level of the pyramid (recall
        that the most lowest level, ie #0, refers to the original image). For
        intermediary levels, values are linearly interpolated.
     
    Returns:
    None
    """

    save_behaviors = [
        "NotOverwrite",
        "Continue",
        "Overwrite"
    ]

    # Check dataset is an instance of Dataset
    if not isinstance(dataset, Dataset):
        raise ValueError("dataset must be an instance of the Dataset class.")

    # Check save_path is a directory and create if it does not exist
    if isinstance(save_path, str):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        elif not os.path.isdir(save_path):
            raise ValueError("save_path must be a directory.")
        elif len(os.listdir(save_path)) > 0:
            raise ValueError("save_path must be an empty directory.")
    else:
        raise ValueError("save_path must be a string representing a directory path.")

    # Check use_channel is a number between 0 and the number of channels in the dataset
    if not isinstance(use_channel, int) or use_channel < 0 or use_channel > dataset._nchannels-1:
        raise ValueError(f"use_channel must be an integer between 0 and {dataset._nchannels - 1}.")
        
    # Check numbers
    if numbers is None:
        numbers = dataset._numbers
    else:
        if not all(num in dataset._numbers for num in numbers):
            raise ValueError("All elements in numbers must be present in dataset._numbers.")

    # Check save behavior
    if save_behavior not in save_behaviors:
        raise ValueError(f"save_behavior should be one of the following: {save_behaviors}.")

    # Perform global transformation
    if perfom_global_trnsf == None and registration_type in ["translation2D", "translation3D", "translation", "rigid2D", "rigid3D", "rigid"]:
        perfom_global_trnsf = True
    
    #Define registration_direction
    if registration_direction == None:
        if registration_type == "rigid":
            registration_direction = "backward"
        else:
            registration_direction = "forward"
    elif registration_direction not in ["forward", "backward"]:
        raise ValueError("registration_direction must be either None or 'forward' or 'backward'.")
    
    # Check downsample
    if downsample is None:
        downsample = (1,) * dataset._ndim_spatial

    # Check if dataset spatial dimensions are not 3, set plot projections to False
    if dataset._ndim_spatial != 3:
        plot_old_projections = False
        plot_projections = False
        print("Warning: Dataset spatial dimensions are not 3. Plot projections have been disabled.")

    # Check make_vectorfield
    if make_vectorfield is None and registration_type not in ["translation2D", "translation3D", "translation", "rigid2D", "rigid3D", "rigid"]:
        make_vectorfield = True

    #Create directories
    os.makedirs(save_path, exist_ok=True)
    directory_trnsf_relative = f"{save_path}/trnsf_relative"
    os.makedirs(directory_trnsf_relative, exist_ok=True)
    if perfom_global_trnsf:
        directory_trnsf_global = f"{save_path}/trnsf_global"
        os.makedirs(directory_trnsf_global, exist_ok=True)
    if apply_registration:
        for ch in range(dataset._nchannels):
            directory_register_files = f"{save_path}/files_ch{ch}"
            os.makedirs(directory_register_files, exist_ok=True)
    if plot_old_projections or plot_projections:
        directory_projections = f"{save_path}/projections"
        os.makedirs(directory_projections, exist_ok=True)
        if plot_old_projections:    
            for ch in range(dataset._nchannels):
                directory_old_projections = f"{directory_projections}/old_projections_ch{ch}"
                os.makedirs(directory_old_projections, exist_ok=True)
        if plot_projections:
            for ch in range(dataset._nchannels):
                directory_new_projections = f"{directory_projections}/projections_ch{ch}"
                os.makedirs(directory_new_projections, exist_ok=True)
    if make_vectorfield:
        directory_vectorfield = f"{save_path}/vectorfield"
        os.makedirs(directory_vectorfield, exist_ok=True)

    directory_tmp = f"{save_path}/.tmp_files"
    os.makedirs(directory_tmp, exist_ok=True)

    # Register cleanup function to remove the temporary directory
    def cleanup():
        if debug == 0 and os.path.exists(directory_tmp):
            shutil.rmtree(directory_tmp)
        warnings.filterwarnings("default", category=UserWarning, message=".*low contrast image.*")
    atexit.register(cleanup)

    # Suppress skimage warnings for low contrast images
    warnings.filterwarnings("ignore", category=UserWarning, message=".*low contrast image.*")

    #Check file is not too heavy
    tmp_file = dataset.get_time_file(os.path.join(directory_tmp, "ref"), 0, 0, downsample)
    try:
        l = np.prod(imread(tmp_file).shape)
        l2 = np.prod(vt.vtImage(tmp_file).shape())
    except:
        raise ValueError("The images are too big to be handled by the registration package (vt). The only way around so far is to define a downsample factor for the images.")

    if l != l2:
        raise ValueError("The images are too big to be handled by the registration package (vt). The only way around so far is to define a downsample factor for the images.")

    if registration_direction == "backward":
        origin = numbers[0]
        iterator = zip(
            range(len(numbers)-1),
            range(1,len(numbers)),
            numbers[:-1],
            numbers[1:]
        )
    else:
        origin = numbers[-1]
        iterator = zip(
            range(len(numbers), 1, -1),
            range(len(numbers)-1, 0, -1),
            numbers[1:][::-1],
            numbers[:-1][::-1]
        )

    for pos_ref, pos_float, file_ref, file_float in tqdm(iterator, desc=f"Registering images using channel {use_channel}", unit=""):
            
        if pos_ref in [0, len(numbers)-1]:
            tmp_file_ref = dataset.get_time_file(os.path.join(directory_tmp, "ref"), pos_ref, use_channel, downsample)
            img_ref = vt.vtImage(tmp_file_ref)
            img_ref.setSpacing(dataset._scale[::-1])
        else:
            img_ref = img_float

        tmp_file_float = dataset.get_time_file(os.path.join(directory_tmp, "float"), pos_float, use_channel, downsample)
        img_float = vt.vtImage(tmp_file_float)
        img_float.setSpacing(dataset._scale[::-1])

        add = ""
        if not verbose:
            add = " -no-verbose"
        add += f" -transformation-type {registration_type} -pyramid-lowest-level {pyramid_lowest_level} -pyramid-highest-level {pyramid_highest_level}"

        trnsf = vt.blockmatching(img_float, image_ref = img_ref, params=args_registration+add)
        trnsf.write(f"{save_path}/trnsf_relative/trnsf_relative_{file_float:04d}_{file_ref:04d}")

        if make_vectorfield:
            vectors = transformation_to_vectorfield(
                img_ref.to_array() > vectorfield_threshold,
                trnsf,
                dataset._scale,
                vectorfield_spacing,
                pos_float
            )
            np.save(f"{directory_vectorfield}/vectorfield_{file_float:04d}_{file_ref:04d}.npy", vectors)

        if perfom_global_trnsf:
            if file_ref != origin:
                trnsf = vt.compose_trsf([
                    vt.vtTransformation(f"{save_path}/trnsf_global/trnsf_global_{file_ref:04d}_{origin:04d}"),
                    trnsf
                ])
                trnsf.write(f"{save_path}/trnsf_global/trnsf_global_{file_float:04d}_{origin:04d}")
            else:
                trnsf.write(f"{save_path}/trnsf_global/trnsf_global_{file_float:04d}_{origin:04d}")
        
        if apply_registration:
            if file_ref == origin:
                img_ref_ = img_ref.to_array()
                imsave(f"{save_path}/files_ch{use_channel}/registered_files_{file_ref:04d}.tiff", img_ref_)            
                if plot_old_projections:
                    for dim in range(dataset._ndim_spatial):
                        imsave(f"{directory_projections}/old_projections_ch{use_channel}/old_projections_{dim}_{file_ref:04d}.tiff", img_ref_.max(axis=dim))
                if plot_projections:
                    for dim in range(dataset._ndim_spatial):
                        imsave(f"{directory_projections}/projections_ch{use_channel}/projections_{dim}_{file_ref:04d}.tiff", img_ref_.max(axis=dim))

            img_corr = vt.apply_trsf(img_float, trnsf).to_array()
            imsave(f"{save_path}/files_ch{use_channel}/registered_files_{file_float:04d}.tiff", img_corr)
            if plot_old_projections:
                for dim in range(dataset._ndim_spatial):
                    imsave(f"{directory_projections}/old_projections_ch{use_channel}/old_projections_{dim}_{file_float:04d}.tiff", imread(tmp_file_float).max(axis=dim))
            if plot_projections:
                for dim in range(dataset._ndim_spatial):
                    imsave(f"{directory_projections}/projections_ch{use_channel}/projections_{dim}_{file_float:04d}.tiff", img_corr.max(axis=dim))

    # Apply registration to all other channels if apply_registration is True
    if apply_registration:
        for ch in range(dataset._nchannels):
            if ch != use_channel:

                iterator = zip(
                    range(len(numbers)-1),
                    range(1,len(numbers)),
                    numbers[:-1],
                    numbers[1:]
                ) if registration_direction == "backward" else zip(
                    range(len(numbers), 1, -1),
                    range(len(numbers)-1, 0, -1),
                    numbers[1:][::-1],
                    numbers[:-1][::-1]
                )

                for pos_ref, pos_float, file_ref, file_float in tqdm(iterator, desc=f"Applying registration to channel {ch}", unit=""):
                    if file_ref == origin:
                        img_corr_ch = dataset.get_time_data(pos_ref, ch, downsample)
                        imsave(f"{save_path}/files_ch{ch}/registered_files_{file_ref:04d}.tiff", img_corr_ch)

                        if plot_old_projections:
                            for dim in range(dataset._ndim_spatial):
                                imsave(f"{directory_projections}/old_projections_ch{ch}/old_projections_{dim}_{file_ref:04d}.tiff", img_corr_ch.max(axis=dim))

                        if plot_projections:
                            for dim in range(dataset._ndim_spatial):
                                imsave(f"{directory_projections}/projections_ch{ch}/projections_{dim}_{file_ref:04d}.tiff", img_corr_ch.max(axis=dim))

                    tmp_file_float_ch = dataset.get_time_file(os.path.join(directory_tmp, f"float_ch{ch}"), pos_float, ch, downsample)
                    img_float_ch = vt.vtImage(tmp_file_float_ch)
                    img_float_ch.setSpacing(dataset._scale[::-1])
                        
                    if perfom_global_trnsf:
                        trnsf_global = vt.vtTransformation(f"{save_path}/trnsf_global/trnsf_global_{file_float:04d}_{origin:04d}")
                        img_corr_ch = vt.apply_trsf(img_float_ch, trnsf_global).to_array()
                    else:
                        trnsf_relative = vt.vtTransformation(f"{save_path}/trnsf_relative/trnsf_relative_{file_float:04d}_{file_ref:04d}")
                        img_corr_ch = vt.apply_trsf(img_float_ch, trnsf_relative).to_array()

                    imsave(f"{save_path}/files_ch{ch}/registered_files_{file_float:04d}.tiff", img_corr_ch)

                    if plot_old_projections:
                        for dim in range(dataset._ndim_spatial):
                            imsave(f"{directory_projections}/old_projections_ch{ch}/old_projections_{dim}_{file_float:04d}.tiff", img_corr_ch.max(axis=dim))

                    if plot_projections:
                        for dim in range(dataset._ndim_spatial):
                            imsave(f"{directory_projections}/projections_ch{ch}/projections_{dim}_{file_float:04d}.tiff", img_corr_ch.max(axis=dim))

        # Joint all vectorfields into a single file
        if make_vectorfield:
            vectorfield_files = [f"{directory_vectorfield}/vectorfield_{num:04d}_{num-1:04d}.npy" for num in numbers if num != origin]
            vectorfields = [np.load(file) for file in vectorfield_files]
            joint_vectorfield = np.concatenate(vectorfields, axis=0)
            np.save(f"{directory_vectorfield}/joint_vectorfield.npy", joint_vectorfield)

            # Remove original vectorfield files
            for file in vectorfield_files:
                os.remove(file)

        # Joint all projections from one axis into a single file
        if plot_projections:
            for ch in range(dataset._nchannels):
                for dim in range(dataset._ndim_spatial):
                    projection_files = [f"{directory_projections}/projections_ch{ch}/projections_{dim}_{num:04d}.tiff" for num in numbers]
                    projections = [imread(file)[np.newaxis,:,:] for file in projection_files]
                    joint_projection = np.concatenate(projections, axis=0)
                    imsave(f"{directory_projections}/projections_ch{ch}/joint_projections_{dim}.tiff", joint_projection)
                    
                    # Remove original projection files
                    for file in projection_files:
                        os.remove(file)

        # Joint all old projections from one axis into a single file
        if plot_old_projections:
            for ch in range(dataset._nchannels):
                for dim in range(dataset._ndim_spatial):
                    old_projection_files = [f"{directory_projections}/old_projections_ch{ch}/old_projections_{dim}_{num:04d}.tiff" for num in numbers]
                    old_projections = [imread(file)[np.newaxis,:,:] for file in old_projection_files]
                    joint_old_projection = np.concatenate(old_projections, axis=0)
                    imsave(f"{directory_projections}/old_projections_ch{ch}/joint_old_projections_{dim}.tiff", joint_old_projection)

                    # Remove original old projection files
                    for file in old_projection_files:
                        os.remove(file)

    # Remove temporary directory if it exists
    if os.path.exists(directory_tmp):
        shutil.rmtree(directory_tmp)

    # Save parameters to JSON file
    transformation_metadata = {
        "use_channel": use_channel,
        "numbers": numbers,
        "registration_type": registration_type,
        "registration_direction": registration_direction,
        "args_registration": args_registration,
        "padding": padding,
        "downsample": downsample,
        "pyramid_lowest_level": pyramid_lowest_level,
        "pyramid_highest_level": pyramid_highest_level,
        "perfom_global_trnsf": perfom_global_trnsf,
        "apply_registration": apply_registration,
        "save_behavior": save_behavior,
        "plot_old_projections": plot_old_projections,
        "plot_projections": plot_projections,
        "make_vectorfield": make_vectorfield,
        "vectorfield_threshold": vectorfield_threshold,
        "vectorfield_spacing": vectorfield_spacing,
    }

    metadata = dataset.get_metadata()
    metadata["data"] = [os.path.join(save_path,f"files_ch{ch}","registered_files_{:04d}.tiff") for ch in range(dataset._nchannels)]
    metadata["dtype"] = "regex"
    metadata["transformations"] = transformation_metadata

    with open(f"{save_path}/dataset.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return load_dataset(save_path)

def transformation_to_vectorfield(mask, transformation, spacing, separation, time):
    """
    Create points from a mask, apply a transformation, and return the transformed points.

    Parameters:
    mask (np.ndarray): Binary mask where points will be created.
    transformation (vt.vtTransformation): Transformation to apply to the points.
    spacing (tuple): Spacing of the mask in each dimension.
    separation (float): Minimum separation between points.
    Returns:
    np.ndarray: Transformed points.
    """

    # Create points from the mask
    slices = tuple(slice(None, None, separation) for i in range(mask.ndim))
    points = np.argwhere(mask[slices])*separation
    points = points[:, ::-1].tolist()
    points_vt = vt.vtPointList(points)
    points_vt.setSpacing(spacing[::-1])

    # Apply transformation
    transformed_points = vt.apply_trsf_to_points(points_vt, transformation).copy_to_array()

    # Convert points and transformed points to napari vector format
    vectors = np.zeros((len(points), 2, mask.ndim + 1))
    vectors[:, 0, :-1] = points
    vectors[:, 1, :-1] = transformed_points
    vectors[:, :, -1] = time

    return vectors