# Registration Tools

[![Documentation Status](https://img.shields.io/badge/docs-passing-brightgreen)](https://gatocor.github.io/registration_tools/)

## Overview

This project provides tools for working with batches of images. The main functionalities include:

- Dataset creation and manipulation
- Image registration with [vt-python](https://gitlab.inria.fr/morpheme/vt-python). 
- Visualization tools with [Napari](https://napari.org).

## Installation

It is encouraged to make a conda environment for the package.

1. Create the environment using the yml file provided in the repository:

    ```bash
    # Create the environment
    conda create --name registration_tools -f environment.yml
    ```

    Alternatively, you can explicitly install the environment dependencies:

    ```bash
    # Create the environment
    conda create --name registration_tools python=3.8 napari=0.5.5 pyqt=5.15.9 scikit-image=0.25.0 vt-python=1.3.1 -c morpheme -c conda-forge
    ```

2. Activate the conda environment:

    ```bash
    # Activate the environment
    conda activate registration_tools
    ```

3. Finally install this package:

    ```bash
    pip install git+https://github.com/gatocor/registration_tools
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
