import napari
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QVBoxLayout, QWidget, QSlider, QLabel, QPushButton, QLineEdit, QHBoxLayout, QGridLayout
from qtpy.QtCore import Qt, QTimer
from napari.utils.events import Event
from vispy.util.keys import ALT, CONTROL
from skimage.segmentation import flood_fill

from ..utils.auxiliar import *
from .common import *

class BlobSegmentationWidget(QWidget):
    
    def __init__(self, model, viewer, axis):
        super().__init__()

        self.viewer = viewer
        self.model = model
        self.axis = axis
        self._n_points = self.model._layer_labels.data.max()

        self._registration_model = model
        self._last_t = viewer.dims.current_step[axis.index("T")]
        self.setWindowTitle("Blob Segmentation Widget")
        self.layout = QVBoxLayout()
        self.explanation = self.create_explanation()
        self.save = create_button(self, "Save", self.save)
        self.close = create_button(self, "Close", self.close)
        self.model._layer_labels.events.paint.connect(self.add_cell)

        # self.viewer.mouse_drag_callbacks.append(self.add_cell)
        self.setLayout(self.layout)
        viewer.window.add_dock_widget(self, area='right')

    def create_explanation(self):
        widget = QWidget()
        layout = QVBoxLayout()
        label = QLabel(
            """
            """
        )
        layout.addWidget(label)
        widget.setLayout(layout)
        self.layout.addWidget(widget)
        return widget

    def save(self):
        if self.model._out is None:
            show_save_popup(self.model, self.viewer)

        if self.model._out is not None:
            self.model.save()

    def close(self):

        show_close_popup(self.model, self.viewer)

        QTimer.singleShot(0, self.parent().close)
        self.viewer.close()
    
    def add_cell(self, data):

        self.model._layer_labels.selected_label += 1
        self._n_points += 1

        return