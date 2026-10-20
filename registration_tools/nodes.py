"""
Napari_Flow nodes for the function-based registration API.

Napari_Flow (napari_flow_editor) turns the functions in NODES into nodes through the
napari function hook `napari_experimental_provide_function` of this package. After
installing registration_tools, run `nfe plugins --refresh` and reopen the editor: the
nodes appear under Plugins > registration-tools.

That route builds a node from the signature, so these wrappers follow its rules:

- napari layer annotations (strings, so napari is not imported) are input sockets,
  float/int/bool/str arguments are parameters and the return annotation is the only
  output;
- there are no dropdowns, so choices are checked by the functions, whose errors list
  the valid values;
- a registration travels as (dict, metadata). The editor does not draw dicts, and its
  per-node environments can carry them;
- the image socket of the apply nodes comes last: Napari_Flow merges the inputs'
  metadata in socket order, so the image's name, scale and contrast win over the
  registration's.

Nothing heavy is imported here: the editor imports this module even for nodes that it
runs in another environment (e.g. one with vt-python).
"""

import os
import string
import sys
import warnings

from . import functional


def register_pair(
    reference: "napari.types.ImageData",
    moving: "napari.types.ImageData",
    transformation: str = "rigid",
    backend: str = "vt",
    pyramid_levels: int = 3,
    grid_spacing: int = 1,
    use_layer_scale: bool = True,
    extra_args: str = "",
) -> "napari.types.VectorsData":
    """
    Register the moving image onto the reference image (2D or 3D).

    The output is a registration for Apply Registration, Save Registration or
    Registration To Vectors; it is not drawn.

    Transformation: translation, rigid, affine or vectorfield.
    Backend: vt (blockmatching, needs vt-python) or sitk (SimpleITK).
    Pyramid_levels: coarser levels used before the full resolution.
    Grid_spacing: pixels between stored vectors (vectorfield only); larger is lighter.
    Use_layer_scale: register in physical space using the layers' voxel size.
    Extra_args: vt: blockmatching flags, e.g. "-elastic-sigma 2 2 2"; sitk: key=value
    settings, e.g. "metric=Correlation optimizer_numberOfIterations=200".
    """
    reference = _image(reference)
    registration = functional.register_pair(
        reference, _image(moving), transformation=transformation, backend=backend,
        scale=_layer_scale(reference.ndim) if use_layer_scale else None,
        pyramid_levels=pyramid_levels, grid_spacing=grid_spacing,
        **_backend_kwargs(backend, extra_args),
    )
    return registration, {}


def register_stack(
    stack: "napari.types.ImageData",
    reference: "napari.types.ImageData" = None,
    mode: str = "reference",
    reference_index: int = 0,
    accumulate: bool = True,
    transformation: str = "rigid",
    backend: str = "vt",
    pyramid_levels: int = 3,
    grid_spacing: int = 1,
    use_layer_scale: bool = True,
    extra_args: str = "",
) -> "napari.types.VectorsData":
    """
    Register every frame of a time-lapse (T, [Z], Y, X), or every slice of a z-stack
    (Z, Y, X), along the first axis.

    The output is one registration per frame, for Apply Registration, Save Registration
    or Registration To Vectors; it is not drawn.

    Mode: reference registers every frame onto frame Reference_index, or onto the
    reference image when that input is connected. consecutive registers every frame onto
    its neighbour towards Reference_index.
    Reference_index: frame that stays fixed; negative values count from the end.
    Accumulate: with consecutive, chain the neighbour registrations so every frame maps
    onto the reference frame (drift correction); off keeps frame-to-frame motion.
    Transformation, Backend, Pyramid_levels, Grid_spacing, Use_layer_scale, Extra_args:
    as in Register Pair.
    """
    stack = _image(stack)
    registration = functional.register_stack(
        stack, None if reference is None else _image(reference), mode=mode,
        reference_index=reference_index, accumulate=accumulate,
        transformation=transformation, backend=backend,
        scale=_layer_scale(stack.ndim - 1) if use_layer_scale else None,
        pyramid_levels=pyramid_levels, grid_spacing=grid_spacing,
        **_backend_kwargs(backend, extra_args),
    )
    return registration, {}


def apply_registration(
    registration: "napari.types.VectorsData",
    image: "napari.types.ImageData",
    order: int = 1,
    boundary: str = "constant",
    fill_value: float = 0.0,
) -> "napari.types.ImageData":
    """
    Warp an image or a stack with a registration, onto the reference grid.

    A stack registration needs one frame per registered frame along the first axis;
    other leading axes (e.g. channels) get the same transformation. A pair registration
    warps every frame of a stack alike.

    Order: interpolation order, 0 nearest, 1 linear, 3 cubic.
    Boundary: values from outside the image: constant, nearest, reflect, mirror or wrap.
    Fill_value: value used with constant.
    """
    return functional.apply_registration(_image(image), registration, order=order,
                                         boundary=boundary, fill_value=fill_value)


def apply_registration_labels(
    registration: "napari.types.VectorsData",
    labels: "napari.types.LabelsData",
) -> "napari.types.LabelsData":
    """
    Warp a labels layer with a registration, keeping its label values (nearest neighbour).
    """
    return functional.apply_registration(_image(labels), registration, order=0)


def registration_to_vectors(
    registration: "napari.types.VectorsData",
    spacing: int = 16,
) -> "napari.types.VectorsData":
    """
    Draw a registration as arrows: where the content of the reference moved to in the
    moving image, one arrow every Spacing pixels (and per frame for a stack).
    """
    registration = functional._check_registration(registration)
    arrows = functional.registration_to_vectors(registration, spacing=spacing)
    scale = registration["scale"]
    if registration["frames"] is not None:
        scale = (1.0,) + scale
    return arrows, {"layer_type": "vectors", "scale": scale, "edge_width": 1.0, "edge_color": "#FFD166"}


def save_registration(
    registration: "napari.types.VectorsData",
    path: str = "registration.npz",
    overwrite: bool = True,
) -> "napari.types.VectorsData":
    """
    Save a registration to a .npz file and pass it on unchanged.

    Path: a relative path is saved next to the pipeline file. In a batch, {row}, {node}
    and the batch CSV's {<column>} are filled in, e.g. "registrations/{embryo}.npz".
    Overwrite: replace an existing file; off raises instead.
    """
    written = functional.save_registration(registration, _output_path(path), overwrite=overwrite)
    print(f"Registration saved to: {written}")
    return registration, {}


def load_registration(
    path: str = "registration.npz",
    image: "napari.types.ImageData" = None,
) -> "napari.types.VectorsData":
    """
    Load a registration saved with Save Registration.

    Path: as in Save Registration; a relative path not found from the current folder
    is looked up next to the pipeline file.
    image (optional): the image the registration will be applied to, only to check that
    it fits (dimensions and number of frames).
    """
    registration = functional.load_registration(_input_path(path))
    if image is not None:
        _check_fits(registration, _image(image))
    return registration, {}


NODES = (
    register_pair,
    register_stack,
    apply_registration,
    apply_registration_labels,
    registration_to_vectors,
    save_registration,
    load_registration,
)


def _context():
    """Napari_Flow's dispatch context, read without importing it (that import has side effects)."""
    decorator = sys.modules.get("napari_flow_editor.flow_nodes.decorator")
    if decorator is None:
        return {}
    try:
        return decorator.get_dispatch_context() or {}
    except Exception:
        return {}


def _layer_scale(ndim):
    """Voxel size of the last `ndim` axes of the input layers, when Napari_Flow knows it."""
    scale = (_context().get("metadata") or {}).get("scale")
    if scale is None or len(scale) < ndim:
        return None
    return tuple(float(s) for s in scale)[-ndim:]


def _image(image):
    """Napari_Flow passes multiscale layers as a list of levels; use the full resolution."""
    return image[0] if isinstance(image, list) else image


def _backend_kwargs(backend, extra_args):
    text = str(extra_args or "").strip()
    if not text:
        return {}
    if str(backend).strip().lower() == "vt":
        return {"args_registration": text}
    kwargs = {}
    for token in text.split():
        key, separator, value = token.partition("=")
        if not separator or not key:
            raise ValueError(f"Extra_args for sitk are key=value settings; got {token!r}.")
        kwargs[key] = _parse_value(value)
    return kwargs


def _parse_value(text):
    if "," in text:
        return tuple(_parse_value(part) for part in text.split(",") if part)
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _fill(template):
    """Fills {row}, {node} and the batch CSV's columns, as Napari_Flow's Save Plot does."""
    context = _context()
    values = {str(key): value for key, value in (context.get("batch_row") or {}).items()}
    index = context.get("batch_index")
    values["row"] = int(index) if index is not None else 1
    values["node"] = context.get("node_title") or "registration"

    fields = [name for _, name, _, _ in string.Formatter().parse(template) if name]
    missing = [name for name in fields if name not in values]
    if missing:
        available = ", ".join(sorted(values))
        raise ValueError(f"Path uses {{{missing[0]}}}, which is not available. Available here: {available}.")
    # Values become part of a file name; the template's own folders are kept.
    safe = {key: _safe(value) if isinstance(value, str) else value for key, value in values.items()}
    return template.format(**safe)


def _safe(text):
    for character in ("/", "\\", ":"):
        text = text.replace(character, "_")
    return text


def _output_path(template):
    path = os.path.expanduser(_fill(str(template).strip()))
    folder = _context().get("pipeline_dir")
    if folder and not os.path.isabs(path):
        path = os.path.join(folder, path)
    return path


def _input_path(template):
    path = os.path.expanduser(_fill(str(template).strip()))
    folder = _context().get("pipeline_dir")
    if folder and not os.path.isabs(path) and not _registration_exists(path):
        candidate = os.path.join(folder, path)
        if _registration_exists(candidate):
            return candidate
    return path


def _registration_exists(path):
    return os.path.exists(path) or os.path.exists(path + ".npz")


def _check_fits(registration, image):
    shape = registration["shape"]
    ndim = len(shape)
    frames = registration["frames"]
    if image.ndim < ndim:
        raise ValueError(f"The registration is {ndim}D but the image has shape {tuple(image.shape)}.")
    if frames is not None and (image.ndim == ndim or image.shape[0] != frames):
        raise ValueError(
            f"The registration has {frames} frames but the image's first axis has "
            f"{image.shape[0] if image.ndim > ndim else 'none'} (shape {tuple(image.shape)})."
        )
    if tuple(image.shape[-ndim:]) != registration["moving_shape"]:
        warnings.warn(
            f"The image's spatial shape {tuple(image.shape[-ndim:])} differs from the shape the "
            f"registration was computed on ({registration['moving_shape']})."
        )
