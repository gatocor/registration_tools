# Registration Tools Documentation

## Overview

This project provides tools for registering datasets of images. The main functionalities include:

- Dataset creation and manipulation
- Image registration
- Transformation application

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Creating a Dataset

To create a dataset of spherical images, use the `dataset_sphere` function:

```python
from registration_tools.data import dataset_sphere

dataset_sphere(
    folder='path/to/save/folder',
    num_images=10,
    image_size=100,
    num_channels=3,
    min_radius=5,
    max_radius=20,
    jump=2,
    stride=(1, 2, 3)
)
```

### Registering Images

To register images in a dataset, use the `register` function:

```python
from registration_tools import Dataset
from registration_tools.registration import register

dataset = Dataset('path/to/dataset', format='XYZ', numbers=[1, 2, 3, 4, 5])
register(
    dataset=dataset,
    save_path='path/to/save/folder',
    use_channel=0
)
```

## Running Tests

To run the tests, use:

```bash
python -m unittest discover -s src/tests
```

## License

This project is licensed under the MIT License.
