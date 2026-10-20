"""
Function-based registration.

Stateless wrappers around the registration classes. They take numpy arrays and
return a *registration*: a plain dict that says, for every pixel of a reference
grid, where to sample the moving image.

    {
        "kind": "affine" or "field",
        "matrix": (D+1, D+1) or (N, D+1, D+1) array,   # kind "affine"
        "field": (*grid, D) or (N, *grid, D) array,     # kind "field"
        "shape": reference (output) grid shape,
        "moving_shape": shape of the images it is meant for,
        "scale": voxel size used during the registration,
        "frames": N for a stack, None for a pair,
        "transformation", "backend", "mode", "reference_index", "accumulate",
    }

Translation, rigid and affine registrations are stored as one homogeneous
matrix per frame; only "vectorfield" stores displacements, optionally on a
coarser grid (`grid_spacing`) that spans the reference from its first to its
last pixel.

Both are in pixels and numpy axis order, and point from the reference grid into
the moving image, as in `scipy.ndimage.affine_transform`: the reference pixel
`x` is found at `matrix @ [x, 1]` or `x + field(x)` in the moving image. The
displacement is therefore how far the content at `x` moved from the reference
to the moving image.
"""

import itertools
import json
import os
import uuid
import warnings
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

TRANSFORMATIONS = ("translation", "rigid", "affine", "vectorfield")
BACKENDS = ("vt", "sitk")
STACK_MODES = ("reference", "consecutive")
BOUNDARY_MODES = ("constant", "nearest", "reflect", "mirror", "wrap")

FORMAT_TAG = "registration_tools.registration"
FORMAT_VERSION = 1

# Points mapped or pixels warped per batch, so dense steps on large 3D frames
# keep a bounded amount of coordinates in memory (scipy works in float64).
_BATCH = 2**21

_META_KEYS = ("kind", "shape", "moving_shape", "scale", "frames",
              "transformation", "backend", "mode", "reference_index", "accumulate")

_SITK_TYPES = {
    "translation": "TranslationTransform",
    "rigid": "Euler{}DTransform",
    "affine": "AffineTransform",
    "vectorfield": "DisplacementFieldTransform",
}
# SimpleITK settings that replace RegistrationSITK's defaults. With mean squares, empty
# borders or a change of brightness between the images stop the optimizer far from the
# answer (a 6 degree rotation came back as 0.5); correlation does not have that problem.
# The field settings follow tests/test_vectorfield_sitk.py.
_SITK_LINEAR_DEFAULTS = {"metric": "Correlation"}
_SITK_FIELD_DEFAULTS = {
    "metric": "ANTSNeighborhoodCorrelation",
    "optimizer": "GradientDescent",
    "optimizer_radius": 2,
    "displacement_field_smoothing_sigmas": 1.5,
}


def register_pair(reference, moving, *, transformation="rigid", backend="vt", scale=None,
                  pyramid_levels=3, grid_spacing=1, verbose=False, **backend_kwargs):
    """
    Registers one image onto another.

    Args:
        reference (array): 2D or 3D image that stays fixed. Its grid is the output grid.
        moving (array): Image to align onto `reference`, with the same number of dimensions.
            Its shape may differ.
        transformation (str): "translation", "rigid", "affine" or "vectorfield".
        backend (str): "vt" (blockmatching) or "sitk" (SimpleITK).
        scale (tuple, optional): Voxel size per axis in numpy order, e.g. (2, 0.5, 0.5).
            Default is 1 along every axis.
        pyramid_levels (int): Coarser levels used before the full resolution.
        grid_spacing (int): Pixels between the stored vectors of a "vectorfield" registration.
            1 stores one vector per pixel. Ignored by the linear transformations.
        verbose (bool): Print the backend's messages.
        **backend_kwargs: Passed to RegistrationVT or RegistrationSITK, e.g.
            args_registration="-elastic-sigma 2 2 2" or metric="Correlation".

    Returns:
        dict: The registration of `moving` onto `reference`.
    """
    reference = _load(reference)
    moving = _load(moving)
    if reference.ndim not in (2, 3) or moving.ndim != reference.ndim:
        raise ValueError(
            f"register_pair needs two 2D or two 3D images; got shapes {reference.shape} and {moving.shape}."
        )
    estimate = _Estimator(transformation, backend, reference.ndim, scale,
                          pyramid_levels, grid_spacing, verbose, backend_kwargs)
    return _new_registration(
        estimate.kind, estimate(reference, moving), reference.shape, moving.shape,
        estimate.scale, estimate.transformation, estimate.backend,
    )


def register_stack(stack, reference=None, *, mode="reference", reference_index=0, accumulate=True,
                   transformation="rigid", backend="vt", scale=None, pyramid_levels=3,
                   grid_spacing=1, verbose=False, **backend_kwargs):
    """
    Registers every frame of a time-lapse or a z-stack.

    Args:
        stack (array): (N, Y, X) or (N, Z, Y, X). The first axis is the one registered along
            (time, or z to align the slices of a z-stack). Frames are read one at a time,
            so zarr and dask arrays are not loaded whole.
        reference (array, optional): External image to register every frame onto. Only
            with mode="reference"; by default the frame `reference_index` is used.
        mode (str): "reference" registers every frame onto the reference frame (or image).
            "consecutive" registers every frame onto its neighbour towards `reference_index`.
        reference_index (int): Frame that stays fixed. Negative values count from the end.
        accumulate (bool): With mode="consecutive", chain the neighbour registrations so that
            every frame maps onto the reference frame (drift correction). False keeps the
            frame-to-frame registrations.
        transformation, backend, scale, pyramid_levels, grid_spacing, verbose, **backend_kwargs:
            As in `register_pair`, applied to every frame.

    Returns:
        dict: A registration with one transformation per frame. The reference frame gets the
            identity.
    """
    mode = _check_choice("mode", mode, STACK_MODES)
    stack = _load(stack, lazy=True)
    if stack.ndim not in (3, 4):
        raise ValueError(f"register_stack needs a (N, Y, X) or (N, Z, Y, X) stack; got shape {stack.shape}.")
    n_frames = stack.shape[0]
    if not -n_frames <= reference_index < n_frames:
        raise IndexError(f"reference_index {reference_index} is out of range for {n_frames} frames.")
    reference_index %= n_frames
    if reference is not None and mode != "reference":
        raise ValueError("An external reference image only works with mode='reference'.")

    ndim = stack.ndim - 1
    estimate = _Estimator(transformation, backend, ndim, scale,
                          pyramid_levels, grid_spacing, verbose, backend_kwargs)

    def frame(t):
        return np.asarray(stack[t])

    if reference is not None:
        reference = _load(reference)
        if reference.ndim != ndim:
            raise ValueError(f"The reference image must be {ndim}D like the frames; got shape {reference.shape}.")
        shape = reference.shape
    else:
        shape = tuple(stack.shape[1:])
    identity = _identity(estimate.kind, shape, estimate.grid_spacing)

    from tqdm import tqdm

    progress = tqdm(total=n_frames, desc="Registering frames", unit="frame", disable=not verbose)
    maps = [None] * n_frames
    if mode == "reference":
        target = reference if reference is not None else frame(reference_index)
        for t in range(n_frames):
            if reference is None and t == reference_index:
                maps[t] = identity
            else:
                maps[t] = estimate(target, frame(t))
            progress.update()
    else:
        maps[reference_index] = identity
        progress.update()
        for direction in (range(reference_index + 1, n_frames), range(reference_index - 1, -1, -1)):
            previous, previous_image = reference_index, frame(reference_index)
            for t in direction:
                image = frame(t)
                relative = estimate(previous_image, image)
                if accumulate:
                    # Reference -> previous frame, then previous frame -> this frame.
                    _, maps[t] = _compose_frames(estimate.kind, relative, shape,
                                                 estimate.kind, maps[previous], shape)
                else:
                    maps[t] = relative
                previous, previous_image = t, image
                progress.update()
    progress.close()

    return _new_registration(
        estimate.kind, np.stack(maps), shape, stack.shape[1:], estimate.scale,
        estimate.transformation, estimate.backend, frames=n_frames, mode=mode,
        reference_index=reference_index,
        accumulate=bool(accumulate) if mode == "consecutive" else None,
    )


def apply_registration(image, registration, *, order=1, boundary="constant", fill_value=0.0):
    """
    Warps an image or a stack with a registration.

    Args:
        image (array): (*spatial) or (N, ..., *spatial). A stack registration needs the first
            axis to have one entry per registered frame; any other leading axes (e.g. channels)
            get the same transformation. A pair registration warps every leading index alike.
        registration (dict): From register_pair, register_stack, compose_registrations or
            load_registration.
        order (int): Interpolation order: 0 nearest (labels), 1 linear, 3 cubic.
        boundary (str): Values sampled outside the moving image: "constant", "nearest",
            "reflect", "mirror" or "wrap".
        fill_value (float): Value used with boundary="constant".

    Returns:
        array: The image on the reference grid, with the input's dtype. A dask input gives a
            lazy dask array computed frame by frame.
    """
    registration = _check_registration(registration)
    boundary = _check_choice("boundary", boundary, BOUNDARY_MODES)
    order = int(order)
    if not 0 <= order <= 5:
        raise ValueError(f"order must be between 0 and 5; got {order}.")

    image = image[0] if isinstance(image, list) else image
    shape = registration["shape"]
    ndim = len(shape)
    frames = registration["frames"]
    if image.ndim < ndim:
        raise ValueError(f"The registration is {ndim}D but the image has shape {image.shape}.")
    leading = tuple(image.shape[:-ndim])
    if frames is not None and (not leading or leading[0] != frames):
        raise ValueError(
            f"The registration has {frames} frames, so the image's first axis must have {frames} "
            f"entries; got shape {image.shape}."
        )
    if tuple(image.shape[-ndim:]) != registration["moving_shape"]:
        warnings.warn(
            f"The image's spatial shape {tuple(image.shape[-ndim:])} differs from the shape the "
            f"registration was computed on ({registration['moving_shape']})."
        )

    data = registration["matrix"] if registration["kind"] == "affine" else registration["field"]

    def warp(frame, index):
        frame_data = data[index[0]] if frames is not None else data
        return _warp(frame, registration["kind"], frame_data, shape, order, boundary, fill_value)

    if _is_dask(image):
        import dask.array as da

        blocks = image.rechunk((1,) * len(leading) + (-1,) * ndim)

        def warp_block(block, block_info=None):
            index = block_info[0]["chunk-location"]
            warped = warp(block.reshape(block.shape[-ndim:]), index)
            return warped.reshape((1,) * len(leading) + shape)

        out_chunks = tuple((1,) * n for n in leading) + tuple((n,) for n in shape)
        return da.map_blocks(warp_block, blocks, dtype=image.dtype, chunks=out_chunks,
                             meta=np.empty((0,) * image.ndim, dtype=image.dtype))

    out = np.empty(leading + shape, dtype=image.dtype)
    for index in np.ndindex(*leading):
        out[index] = warp(np.asarray(image[index]), index)
    return out


def registration_to_vectors(registration, spacing=16):
    """
    Samples a registration as napari Vectors (arrows), in pixels of the reference grid.

    Args:
        registration (dict): The registration to draw.
        spacing (int): Pixels between arrows.

    Returns:
        array: (M, 2, D) [start, displacement] for a pair, or (M, 2, 1+D) for a stack, where the
            first coordinate is the frame (as `Registration.vectorfield` returns). Add it to napari
            with `scale=registration["scale"]` (prefixed with 1 for a stack) to place it in
            physical units, e.g. `visualization.add_vectors(viewer, vectors, scale=...)`.
    """
    registration = _check_registration(registration)
    shape = registration["shape"]
    spacing = max(1, int(spacing))
    axes = [np.arange(spacing // 2, n, spacing, dtype=float) if spacing // 2 < n else np.array([n // 2], float)
            for n in shape]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(shape))

    arrows = []
    for t, (kind, data) in enumerate(_frames(registration)):
        displacement = _map_points(kind, data, shape, points) - points
        if registration["frames"] is None:
            start, direction = points, displacement
        else:
            start = np.c_[np.full(len(points), float(t)), points]
            direction = np.c_[np.zeros(len(points)), displacement]
        arrows.append(np.stack([start, direction], axis=1))
    return np.concatenate(arrows).astype(np.float32)


def registration_to_field(registration, grid_spacing=1):
    """
    Expands a registration into displacements, for analysis.

    Args:
        registration (dict): The registration to expand.
        grid_spacing (int): Pixels between samples; 1 gives one displacement per pixel.

    Returns:
        array: (*grid, D) for a pair or (N, *grid, D) for a stack, in pixels, components in
            numpy axis order. The grid spans the reference from its first to its last pixel.
    """
    registration = _check_registration(registration)
    axes = _grid_axes(registration["shape"], grid_spacing)
    fields = [_sample_displacement(kind, data, registration["shape"], axes)
              for kind, data in _frames(registration)]
    return np.stack(fields) if registration["frames"] is not None else fields[0]


def compose_registrations(first, second):
    """
    Combines two registrations into one that does both.

    `first` registers the moving image onto an intermediate grid and `second` registers that
    result onto the final reference, e.g. a rigid pre-alignment followed by a vectorfield
    refinement of the rigidly registered image.

    Args:
        first (dict): Registration applied first to the moving image.
        second (dict): Registration applied to the output of `first`.

    Returns:
        dict: The combined registration. Two affine registrations stay affine; anything with a
            field becomes a field on `second`'s grid (or `first`'s spacing). Stacks are combined
            frame by frame and a pair is used for every frame of a stack.
    """
    first = _check_registration(first)
    second = _check_registration(second)
    if len(first["shape"]) != len(second["shape"]):
        raise ValueError("Both registrations must have the same number of spatial dimensions.")
    if first["frames"] is not None and second["frames"] is not None and first["frames"] != second["frames"]:
        raise ValueError(f"Cannot combine stacks of {first['frames']} and {second['frames']} frames.")
    if second["moving_shape"] != first["shape"]:
        warnings.warn(
            f"`second` was computed on images of shape {second['moving_shape']}, but `first` "
            f"outputs shape {first['shape']}."
        )

    frames = first["frames"] if first["frames"] is not None else second["frames"]
    firsts = list(_frames(first))
    seconds = list(_frames(second))
    composed = [
        _compose_frames(*firsts[t if first["frames"] is not None else 0], first["shape"],
                        *seconds[t if second["frames"] is not None else 0], second["shape"])
        for t in range(frames if frames is not None else 1)
    ]
    kind = composed[0][0]
    data = [frame_data for _, frame_data in composed]

    stacked = second if second["frames"] is not None else first

    def combined(key):
        values = [first[key], second[key]]
        return values[0] if values[0] == values[1] else " + ".join(str(v) for v in values)

    return _new_registration(
        kind, np.stack(data) if frames is not None else data[0], second["shape"],
        first["moving_shape"], second["scale"], combined("transformation"), combined("backend"),
        frames=frames, mode=stacked["mode"], reference_index=stacked["reference_index"],
        accumulate=stacked["accumulate"],
    )


def save_registration(registration, path, *, overwrite=True):
    """
    Saves a registration to a .npz file so it can be reused.

    Args:
        registration (dict): The registration to save.
        path (str or Path): Destination; ".npz" is added when missing. Missing folders are created.
        overwrite (bool): Replace an existing file. False raises instead.

    Returns:
        Path: The file written.
    """
    registration = _check_registration(registration)
    path = _npz_path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {key: registration[key] for key in _META_KEYS}
    meta.update(format=FORMAT_TAG, version=FORMAT_VERSION)
    key = "matrix" if registration["kind"] == "affine" else "field"

    # Written beside the target and renamed, so an interrupted save never leaves a broken file.
    # open() rather than tempfile, whose files are private to the owner whatever the umask.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "xb") as file:
            np.savez(file, meta=np.array(json.dumps(meta, default=_json_value)), **{key: registration[key]})
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return path


def load_registration(path):
    """
    Loads a registration saved with `save_registration`.

    Args:
        path (str or Path): The .npz file; ".npz" may be omitted.

    Returns:
        dict: The registration.
    """
    path = Path(path).expanduser()
    if not path.exists() and path.suffix != ".npz" and _npz_path(path).exists():
        path = _npz_path(path)
    if not path.exists():
        raise FileNotFoundError(f"No registration file at {path}.")

    with np.load(path, allow_pickle=False) as content:
        if "meta" not in content.files:
            raise ValueError(f"{path} is not a registration file (no 'meta' entry).")
        meta = json.loads(str(content["meta"]))
        if meta.get("format") != FORMAT_TAG:
            raise ValueError(f"{path} is not a registration file (format {meta.get('format')!r}).")
        if meta.get("version", 0) > FORMAT_VERSION:
            raise ValueError(
                f"{path} was written by a newer registration_tools (format version {meta['version']})."
            )
        key = "matrix" if meta["kind"] == "affine" else "field"
        registration = {name: meta.get(name) for name in _META_KEYS}
        registration[key] = content[key]
    return _check_registration(registration)


class _Estimator:
    """Runs the backend on one pair of images and converts its transform to a frame of a registration."""

    def __init__(self, transformation, backend, ndim, scale, pyramid_levels, grid_spacing,
                 verbose, backend_kwargs):
        self.transformation = _check_choice("transformation", transformation, TRANSFORMATIONS)
        self.backend = _check_choice("backend", backend, BACKENDS)
        self.kind = "field" if self.transformation == "vectorfield" else "affine"
        self.ndim = ndim
        self.scale = _scale(scale, ndim)
        self.pyramid_levels = int(pyramid_levels)
        if self.pyramid_levels < 0:
            raise ValueError(f"pyramid_levels must be 0 or more; got {pyramid_levels}.")
        self.grid_spacing = int(grid_spacing)
        if self.grid_spacing < 1:
            raise ValueError(f"grid_spacing must be 1 or more; got {grid_spacing}.")
        self.verbose = verbose
        self.backend_kwargs = dict(backend_kwargs)

        if self.backend == "vt":
            try:
                import vt  # noqa: F401
            except ImportError as error:
                raise ImportError(
                    "backend='vt' needs vt-python (conda install vt-python -c morpheme); "
                    "backend='sitk' works with SimpleITK instead."
                ) from error

    def __call__(self, reference, moving):
        model = self._model(reference.shape)
        transform = model.register(moving, reference, tuple(float(s) for s in self.scale), verbose=self.verbose)
        if self.kind == "affine":
            corners = np.array(list(itertools.product(*[(0.0, float(max(n - 1, 1))) for n in reference.shape])))
            return _fit_affine(corners, self._map(transform, corners))
        return self._field(transform, reference.shape)

    def _model(self, shape):
        # A fresh model per pair: register() only reads the constructor's settings and _n_spatial.
        if self.backend == "vt":
            from .registration import RegistrationVT

            name = self.transformation
            if name != "vectorfield":
                name = f"{name}{self.ndim}D"
            kwargs = {"pyramid_highest_level": self.pyramid_levels, **self.backend_kwargs}
            model = RegistrationVT(registration_type=name, **kwargs)
        else:
            from .registration import RegistrationSITK

            # Keep the coarsest level at least 8 pixels wide.
            levels = min(self.pyramid_levels, max(0, int(np.log2(max(min(shape), 1) / 8))))
            kwargs = {
                "pyramid_shrink_factor": tuple(2**k for k in range(levels, -1, -1)),
                "pyramid_smoothing_sigmas": tuple(float(k) for k in range(levels, -1, -1)),
            }
            kwargs.update(_SITK_FIELD_DEFAULTS if self.kind == "field" else _SITK_LINEAR_DEFAULTS)
            kwargs.update(self.backend_kwargs)
            model = RegistrationSITK(_SITK_TYPES[self.transformation].format(self.ndim), **kwargs)
        model._n_spatial = self.ndim
        return model

    def _map(self, transform, points):
        """Positions in the moving image (pixels) of reference pixels `points` (P, D)."""
        physical = points * self.scale
        if self.backend == "vt":
            import vt

            mapped = np.empty_like(physical)
            for start in range(0, len(physical), _BATCH):
                chunk = np.ascontiguousarray(physical[start:start + _BATCH, ::-1])
                result = vt.apply_trsf_to_points(vt.vtPointList(chunk), transform).copy_to_array()
                mapped[start:start + len(chunk)] = result[:, :self.ndim][:, ::-1]
        else:
            mapped = np.array([transform.TransformPoint(tuple(float(v) for v in p[::-1])) for p in physical])
            mapped = mapped[:, ::-1]
        return mapped / self.scale

    def _field(self, transform, shape):
        axes = _grid_axes(shape, self.grid_spacing)
        grid = tuple(len(a) for a in axes)
        if self.backend == "sitk":
            import SimpleITK as sitk

            steps = [a[1] - a[0] if len(a) > 1 else 1.0 for a in axes]
            field = sitk.TransformToDisplacementField(
                transform, sitk.sitkVectorFloat64, list(grid[::-1]), [0.0] * self.ndim,
                [float(step * s) for step, s in zip(steps, self.scale)][::-1],
                np.eye(self.ndim).ravel().tolist(),
            )
            return (sitk.GetArrayFromImage(field)[..., ::-1] / self.scale).astype(np.float32)

        total = int(np.prod(grid))
        displacement = np.empty((total, self.ndim), np.float32)
        for start in range(0, total, _BATCH):
            index = np.unravel_index(np.arange(start, min(start + _BATCH, total)), grid)
            points = np.stack([a[i] for a, i in zip(axes, index)], axis=1)
            displacement[start:start + len(points)] = self._map(transform, points) - points
        return displacement.reshape(grid + (self.ndim,))


def _new_registration(kind, data, shape, moving_shape, scale, transformation, backend, *,
                      frames=None, mode=None, reference_index=None, accumulate=None):
    registration = {
        "kind": kind,
        "shape": tuple(int(n) for n in shape),
        "moving_shape": tuple(int(n) for n in moving_shape),
        "scale": tuple(float(s) for s in scale),
        "frames": None if frames is None else int(frames),
        "transformation": transformation,
        "backend": backend,
        "mode": mode,
        "reference_index": None if reference_index is None else int(reference_index),
        "accumulate": accumulate,
    }
    registration["matrix" if kind == "affine" else "field"] = (
        np.asarray(data, dtype=np.float64) if kind == "affine" else np.asarray(data, dtype=np.float32)
    )
    return registration


def _check_registration(registration):
    """Validates a registration and returns a normalised shallow copy."""
    kind = registration.get("kind") if isinstance(registration, dict) else None
    key = {"affine": "matrix", "field": "field"}.get(kind)
    if key is None or key not in registration:
        raise TypeError(
            "Expected a registration (the dict returned by register_pair, register_stack or "
            f"load_registration); got {_describe(registration)}."
        )
    try:
        shape = tuple(int(n) for n in registration["shape"])
        frames = registration.get("frames")
        frames = None if frames is None else int(frames)
        out = dict(registration)
        out.update(
            shape=shape,
            moving_shape=tuple(int(n) for n in registration.get("moving_shape") or shape),
            scale=tuple(float(s) for s in registration.get("scale") or (1.0,) * len(shape)),
            frames=frames,
        )
        for name in ("transformation", "backend", "mode", "reference_index", "accumulate"):
            out.setdefault(name, None)
    except (KeyError, TypeError, ValueError) as error:
        raise TypeError(f"Malformed registration: {error}") from error

    data = np.asarray(registration[key])
    ndim = len(shape)
    leading = () if frames is None else (frames,)
    if kind == "affine":
        expected = leading + (ndim + 1, ndim + 1)
        if data.shape != expected:
            raise TypeError(f"Malformed registration: matrix of shape {data.shape}, expected {expected}.")
    elif data.shape[:len(leading)] != leading or data.ndim != len(leading) + ndim + 1 or data.shape[-1] != ndim:
        raise TypeError(
            f"Malformed registration: field of shape {data.shape} for a {ndim}D grid"
            + (f" with {frames} frames." if frames is not None else ".")
        )
    out[key] = data
    return out


def _frames(registration):
    """(kind, data) for every frame of a registration; a pair has one."""
    kind = registration["kind"]
    data = registration["matrix"] if kind == "affine" else registration["field"]
    if registration["frames"] is None:
        yield kind, data
    else:
        for frame in data:
            yield kind, frame


def _identity(kind, shape, grid_spacing):
    if kind == "affine":
        return np.eye(len(shape) + 1)
    grid = tuple(len(a) for a in _grid_axes(shape, grid_spacing))
    return np.zeros(grid + (len(shape),), np.float32)


def _grid_axes(shape, spacing):
    """Sample positions per axis, from the first to the last pixel, at most `spacing` apart."""
    if np.isscalar(spacing):
        spacing = (spacing,) * len(shape)
    axes = []
    for n, step in zip(shape, spacing):
        if step <= 0:
            raise ValueError(f"grid spacing must be positive; got {step}.")
        count = int(np.ceil((n - 1) / step - 1e-9)) + 1 if n > 1 else 1
        axes.append(np.linspace(0.0, n - 1.0, count) if n > 1 else np.zeros(1))
    return axes


def _grid_coords(points, shape, grid):
    """Index coordinates (D, P) on a field grid spanning `shape`, of reference pixels `points` (P, D)."""
    factors = np.array([(m - 1) / (n - 1) if n > 1 else 0.0 for n, m in zip(shape, grid)])
    return (points * factors).T


def _map_points(kind, data, shape, points):
    """Positions in the moving image of reference pixels `points` (P, D), for one frame."""
    if kind == "affine":
        return points @ data[:-1, :-1].T + data[:-1, -1]
    coords = _grid_coords(points, shape, data.shape[:-1])
    shift = np.stack([ndi.map_coordinates(data[..., i], coords, order=1, mode="nearest")
                      for i in range(data.shape[-1])], axis=1)
    return points + shift


def _sample_displacement(kind, data, shape, axes):
    """Displacement of one frame on the grid `axes`, as (*grid, D) float32."""
    grid = tuple(len(a) for a in axes)
    total = int(np.prod(grid))
    displacement = np.empty((total, len(shape)), np.float32)
    for start in range(0, total, _BATCH):
        index = np.unravel_index(np.arange(start, min(start + _BATCH, total)), grid)
        points = np.stack([a[i] for a, i in zip(axes, index)], axis=1)
        displacement[start:start + len(points)] = _map_points(kind, data, shape, points) - points
    return displacement.reshape(grid + (len(shape),))


def _compose_frames(first_kind, first, first_shape, second_kind, second, second_shape):
    """One frame of `first` followed by `second`: x -> first(second(x)). Returns (kind, data)."""
    if first_kind == "affine" and second_kind == "affine":
        return "affine", first @ second
    # On second's field grid, or on second's shape at the spacing of first's field.
    field, field_shape = (second, second_shape) if second_kind == "field" else (first, first_shape)
    steps = [(n - 1) / (m - 1) if m > 1 else 1.0 for n, m in zip(field_shape, field.shape[:-1])]
    axes = _grid_axes(second_shape, steps)
    grid = tuple(len(a) for a in axes)
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(second_shape))
    mapped = _map_points(first_kind, first, first_shape, _map_points(second_kind, second, second_shape, points))
    return "field", (mapped - points).reshape(grid + (len(second_shape),)).astype(np.float32)


def _warp(frame, kind, data, shape, order, boundary, fill_value):
    """One frame of `apply_registration`."""
    dtype = frame.dtype
    if dtype == bool:
        frame = frame.view(np.uint8)
    elif dtype.kind == "f" and dtype.itemsize < 4:
        frame = frame.astype(np.float32)
    if order == 0:
        work = frame.dtype
    else:
        work = frame.dtype if frame.dtype.kind == "f" else np.dtype(np.float32)

    if kind == "affine":
        out = ndi.affine_transform(frame, data, output_shape=shape, output=work, order=order,
                                   mode=boundary, cval=fill_value)
        return _cast(out, dtype)

    out = np.empty(shape, dtype=work)
    # Splines (order > 1) prefilter the whole frame on every call, so they get a single slab.
    rows = shape[0] if order > 1 else max(1, _BATCH // max(1, int(np.prod(shape[1:]))))
    for start in range(0, shape[0], rows):
        stop = min(start + rows, shape[0])
        coords = _slab_coordinates(data, shape, start, stop)
        out[start:stop] = ndi.map_coordinates(frame, coords, output=work, order=order,
                                              mode=boundary, cval=fill_value)
    return _cast(out, dtype)


def _slab_coordinates(field, shape, start, stop):
    """Moving-image coordinates (D, rows, ...) of reference rows `start:stop`."""
    axes = [np.arange(start, stop, dtype=float)] + [np.arange(n, dtype=float) for n in shape[1:]]
    position = np.stack(np.meshgrid(*axes, indexing="ij"))
    if field.shape[:-1] == tuple(shape):
        return position + np.moveaxis(field[start:stop], -1, 0)
    grid = field.shape[:-1]
    scaled = [a * ((m - 1) / (n - 1) if n > 1 else 0.0) for a, n, m in zip(axes, shape, grid)]
    coords = np.stack(np.meshgrid(*scaled, indexing="ij"))
    shift = np.stack([ndi.map_coordinates(field[..., i], coords, order=1, mode="nearest")
                      for i in range(len(shape))])
    return position + shift


def _cast(out, dtype):
    if out.dtype == dtype:
        return out
    if dtype == bool:
        return out > 0.5
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(dtype)
    return out.astype(dtype)


def _fit_affine(points, mapped):
    """Homogeneous matrix A with mapped = A @ [points, 1], from point pairs (P, D)."""
    ndim = points.shape[1]
    solution, *_ = np.linalg.lstsq(np.c_[points, np.ones(len(points))], mapped, rcond=None)
    matrix = np.eye(ndim + 1)
    matrix[:ndim] = solution.T
    return matrix


def _scale(scale, ndim):
    if scale is None:
        return np.ones(ndim)
    scale = np.asarray(scale, dtype=float).ravel()
    if len(scale) != ndim or np.any(scale <= 0):
        raise ValueError(f"scale needs {ndim} positive values (numpy axis order); got {tuple(scale)}.")
    return scale


def _check_choice(name, value, options):
    normalised = str(value).strip().lower()
    if normalised not in options:
        raise ValueError(f"{name} must be one of {', '.join(repr(o) for o in options)}; got {value!r}.")
    return normalised


def _load(image, lazy=False):
    """An image as a numpy array; lists are multiscale pyramids and give their first level."""
    if isinstance(image, list):
        image = image[0]
    if lazy and hasattr(image, "shape") and hasattr(image, "__getitem__"):
        return image
    return np.asarray(image.compute() if hasattr(image, "compute") else image)


def _is_dask(array):
    return type(array).__module__.split(".")[0] == "dask"


def _npz_path(path):
    path = Path(path).expanduser()
    return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")


def _describe(value):
    if hasattr(value, "shape"):
        return f"{type(value).__name__} of shape {tuple(value.shape)}"
    return type(value).__name__


def _json_value(value):
    """numpy scalars in a registration's metadata, for json.dumps."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} cannot be saved in a registration's metadata.")
