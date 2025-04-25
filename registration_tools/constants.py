try:
    import cupy as cp
    import cupyx.scipy.ndimage as cndi
    import cucim.skimage.measure as cskm
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

USE_GPU = True

def get_gpu_available():
    """
    Check if a GPU is available.
    """
    return GPU_AVAILABLE

def set_use_gpu(use_gpu):
    """
    Set the GPU usage.
    """
    global USE_GPU
    USE_GPU = use_gpu
