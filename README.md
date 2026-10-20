# Registration Tools

[![Documentation Status](https://img.shields.io/badge/docs-passing-brightgreen)](https://gatocor.github.io/registration_tools/)

## Overview

This project provides tools for working with batches of images. The main functionalities include:

- Dataset creation and manipulation
- Image registration with [vt-python](https://gitlab.inria.fr/morpheme/vt-python). 
- Visualization tools with [Napari](https://napari.org).

## Installation

### Installation from yml file
A `environment.yml` is provided. This will install all the dependencies, even the optative ones. If you want to select just some of them, check maual installation.

```bash
conda create -n registration_tools -f environment.yml
```

### Manual installation

Requirements:

Python version:
  - python>=3.11.11

```bash
conda create -n registration_tools python=3.11
conda activate registration_tools #To activate environment
```

Compulsory packages:
  - h5py
  - zarr>=3.0.3
  - scikit-image

```bash
#Assuming you are inside the environment (registration_tools)
conda install h5py zarr>=3.0.3 scikit-image
```

Additional packages
  - vt-python: In case yuo want to register files. Installation docs (here)[https://gitlab.inria.fr/morpheme/vt-python].

```bash
#Assuming you are inside the environment (registration_tools)
conda install vt-python -c morpheme
```

  - napari: If you want to visualize the data. Installation docs (here)[https://napari.org/dev/tutorials/fundamentals/installation.html].
  - pyqt: Required by napari. Other alternatives in napari docs. 

```bash
#Assuming you are inside the environment (registration_tools)
conda install napari pyqt -c conda-forge
conda update napari #From documentation of napari, sometimes help is after installation napari has not find pyqt correctly
```

## Functional API

Besides the registration classes, `registration_tools` has stateless functions that take numpy (or dask/zarr) arrays:

```python
import registration_tools as rt

reg = rt.register_stack(movie_ch1, mode="consecutive", transformation="rigid", scale=(2, 0.5, 0.5))
aligned_ch2 = rt.apply_registration(movie_ch2, reg)          # same transformation, other channel
rt.save_registration(reg, "embryo1_registration.npz")         # reuse later with rt.load_registration
```

- `register_pair(reference, moving, ...)` registers one 2D/3D image onto another.
- `register_stack(stack, reference=None, mode="reference" | "consecutive", reference_index=0, accumulate=True, ...)` registers every frame of a time-lapse (or every slice of a z-stack) along the first axis.
- `apply_registration(image, registration, order=1, ...)` warps an image or stack onto the reference grid (`order=0` for labels).
- `registration_to_vectors`, `registration_to_field`, `compose_registrations`, `save_registration` and `load_registration` draw, expand, chain and store registrations.

Both backends are available through `backend="vt"` or `backend="sitk"`, with `transformation` one of `"translation"`, `"rigid"`, `"affine"` or `"vectorfield"`.

A registration is a plain dict. Translation, rigid and affine registrations store one homogeneous matrix per frame; only `"vectorfield"` stores displacements, and `grid_spacing` keeps them on a coarser grid. Both are in pixels and numpy axis order, and map every reference pixel `x` to the position `matrix @ [x, 1]` (or `x + field(x)`) of the moving image, as `scipy.ndimage.affine_transform` does.

## Napari_Flow nodes

`registration_tools` provides nodes for [Napari_Flow](https://git.embl.org/grp-torres-sanchez/Napari_Flow): Register Pair, Register Stack, Apply Registration, Apply Registration Labels, Registration To Vectors, Save Registration and Load Registration (`registration_tools/nodes.py`).

```bash
# In the environment that runs Napari_Flow
pip install -e /path/to/registration_tools
nfe plugins --refresh        # then reopen the editor: Plugins > registration-tools
```

vt-python is only available from conda. When the Napari_Flow environment does not have it, either use `backend="sitk"` or run the register nodes in an environment that has vt (set the node's environment, or an `[envs.<name>]` block in `~/.napari_flow/profiles.toml`); that environment needs `napari_flow_editor` installed with `pip install -e /path/to/Napari_Flow --no-deps`.

Example pipelines for every node, with synthetic data whose true motion is known, are in [`examples/napari_flow`](examples/napari_flow/README.md).

## Documentation

For detailed documentation, visit [here](https://gatocor.github.io/registration_tools/).

## Running Tests

To run the tests, use:

```bash
python -m unittest discover -s tests
```

## License

This project is licensed under the MIT License.
