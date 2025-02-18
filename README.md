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

## Documentation

For detailed documentation, visit [here](https://gatocor.github.io/registration_tools/).

## Running Tests

To run the tests, use:

```bash
python -m unittest discover -s tests
```

## License

This project is licensed under the MIT License.
