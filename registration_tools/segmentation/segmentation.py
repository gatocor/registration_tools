import napari
import numpy as np
import zarr

from ..dataset.dataset import Dataset
from ..utils.auxiliar import _get_img_prop, make_index
from ..utils.utils import apply_function
from ..constants import GPU_AVAILABLE, USE_GPU

from ..widgets.widgets_segmentation import BlobSegmentationWidget

class Segmentation():

    def __init__(self):
        self._labels = None
        self._out = None

    def save(self, out=None):
        if out is None and self._out is None:
            raise ValueError("Output path must be specified.")
        elif out is not None:
            self._out = out

        zarr.save(self._out, self._labels)

class SegmentationManualBlob(Segmentation):

    def __init__(self):
        super().__init__()

    def fit_manual(self, image, labels=None, axis=None, scale=None):

        self._img_prop = _get_img_prop(image, axis, scale, requires="")
        scale = self._img_prop.scale
        axis = self._img_prop.axis

        if labels is not None:
            self._labels = labels
        else:
            self._labels = np.zeros(self._img_prop.shape, dtype=np.uint16)
        
        viewer = napari.Viewer()
        viewer.add_image(image, scale=scale, name="Image")
        self._layer_labels = viewer.add_labels(self._labels, scale=scale, name="Labels")
        BlobSegmentationWidget(self, viewer, axis)
        napari.run()

        return self._labels

        