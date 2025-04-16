import pynvml
import threading
import time
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cndi
    import cucim.skimage.measure as cskm
    GPU_AVAILABLE = True
    print("GPU_AVAILABLE!, Steps that can be accelerated with CUDA will be passed to the GPU.")
except ImportError:
    GPU_AVAILABLE = False

class GpuProfiler:
    def __init__(self, dt=0.01, device_index=0):
        self.dt = dt
        self.device_index = device_index
        self._stop_event = threading.Event()
        self._thread = None
        self.peak_memory_mb = 0
        self.total_memory_mb = 1
        if GPU_AVAILABLE:
            self.has_gpu = True
        else:
            self.has_gpu = False

    def _monitor(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.total_memory_mb = meminfo.total / 1024**2  # Total once at start

        while not self._stop_event.is_set():
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = meminfo.used / 1024**2
            self.peak_memory_mb = max(self.peak_memory_mb, used_mb)
            time.sleep(self.dt)

        pynvml.nvmlShutdown()

    @property
    def percent_used(self):
        if self.total_memory_mb == 0:
            return 0
        return (self.peak_memory_mb / self.total_memory_mb) * 100

    def __enter__(self):
        if self.has_gpu:
            self._thread = threading.Thread(target=self._monitor)
            self._thread.start()            
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.has_gpu:
            self._stop_event.set()
            self._thread.join()
