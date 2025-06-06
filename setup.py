from setuptools import setup, find_packages

setup(
    name='registration_tools',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-image',
        'zarr>=3.0.3',
        'h5py>=3',
        'dask[distributed]>=2025',
        'simpleitk>=2',
        'pynvml>=12'
    ],
    extras_require={
        'vt-python': ['vt-python'],
        'napari': ['napari'],
        'pyqt': ['pyqt5'],
    },
    test_suite='tests',
    entry_points={
        'console_scripts': [
            # Add command line scripts here
            # e.g., 'my-tool=my_package.module:main_function'
        ],
    },
    author='Gabriel Torregrosa Cortés',
    author_email='g.torregrosa@example.com',
    description='A description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/gatocor/registration_tools',  # Update with your project's URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)