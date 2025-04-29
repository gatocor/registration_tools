import napari
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QVBoxLayout, QWidget, QSlider, QLabel, QPushButton, QLineEdit, QHBoxLayout, QGridLayout, QApplication
from qtpy.QtCore import Qt, QTimer, QEvent
from qtpy.QtGui import QKeyEvent
from napari.utils.events import Event
from vispy.util.keys import ALT, CONTROL
from skimage.segmentation import flood_fill

import numpy as np
from scipy.ndimage import gaussian_filter


from ..utils.auxiliar import *
from .common import *

class BlobSegmentationWidget(QWidget):
    
    def __init__(self, model, viewer, axis):
        super().__init__()

        self.viewer = viewer
        self.model = model
        self.axis = axis
        self._n_points = np.max(self.model._layer_labels.data)
        self._n_points_original = self._n_points
        self._camera = (0,0,0)
        self.model._layer_labels.selected_label = self._n_points + 1
        self.model._layer_labels.n_edit_dimensions = self.model._img_prop.n_spatial

        self._registration_model = model
        self.setWindowTitle("Blob Segmentation Widget")
        self.layout = QVBoxLayout()
        self.explanation = self.create_explanation()
        self.save = create_button(self, "Save", self.save)
        self.close = create_button(self, "Close", self.close)
        self.
        self.model._layer_labels.events.paint.connect(self.add_cell)

        self.viewer.bind_key('Shift+A', self.move_left)
        self.viewer.bind_key('Shift+D', self.move_right)
        self.viewer.bind_key('Shift+S', self.reduce_brush_size)
        self.viewer.bind_key('Shift+W', self.increase_brush_size)
        self.viewer.bind_key('Shift+Q', self._toggle_labels_paint)
        self.viewer.bind_key('Shift+E', self._toggle_2D3D)
        # self.viewer.bind_key('Ctrl-Z', self.undo)

        self._original_paint = self.model._layer_labels.paint
        self.model._layer_labels.paint = self.custom_paint#.__get__(self.model._layer_labels)

        # self.viewer.mouse_drag_callbacks.append(self.add_cell)
        self.setLayout(self.layout)
        viewer.window.add_dock_widget(self, area='right')

    def custom_paint(self, *args, **kwargs):
        print("Monkey-patched paint!")
        
        pos = args[0]  # clicked position (in data coordinates, not world)
        pos = np.round(pos).astype(int)  # Make sure it's integer indices
        image = self.model._layer_image.data  # Your associated image layer
        size = self.model._layer_labels.brush_size  # Brush radius (your ball size)

        # Define patch boundaries (make sure you don't go out of bounds)
        zmin, zmax = max(pos[0] - size, 0), min(pos[0] + size + 1, image.shape[0])
        ymin, ymax = max(pos[1] - size, 0), min(pos[1] + size + 1, image.shape[1])
        xmin, xmax = max(pos[2] - size, 0), min(pos[2] + size + 1, image.shape[2])
        patch = image[zmin:zmax, ymin:ymax, xmin:xmax].copy()

        # Create ball mask
        zz, yy, xx = np.meshgrid(
            np.arange(zmin, zmax) - pos[0],
            np.arange(ymin, ymax) - pos[1],
            np.arange(xmin, xmax) - pos[2],
            indexing='ij'
        )
        dist = np.sqrt(zz**2 + yy**2 + xx**2)
        ball_mask = dist <= size

        # Apply mask: set outside ball to 0
        patch[~ball_mask] = 0

        # Smooth the patch (optional: use same radius as sigma)
        patch_smoothed = gaussian_filter(patch, sigma=size/2)

        # Find maximum position inside the patch
        local_max_idx = np.unravel_index(np.argmax(patch_smoothed), patch.shape)

        # Map local patch max back to global coordinates
        global_max_pos = (zmin + local_max_idx[0], ymin + local_max_idx[1], xmin + local_max_idx[2])

        args = (global_max_pos,*args[1:])  # Update the position to the global max position
        self._original_paint(*args, **kwargs)

    def create_explanation(self):
        widget = QWidget()
        layout = QVBoxLayout()
        label = QLabel(
            """
            <h2>Blob Segmentation</h2>
            <p>Use the following keys to control the segmentation:</p>
            <ul>
                <li><strong>Shift+A</strong>: Move left</li>
                <li><strong>Shift+D</strong>: Move right</li>
                <li><strong>Shift+W</strong>: Decrease brush size</li>
                <li><strong>Shift+S</strong>: Increase brush size</li>
                <li><strong>Shift+Q</strong>: Toggle paint mode for labels layer</li>
                <li><strong>Shift+E</strong>: Toggle between 2D and 3D view</li>
            </ul>
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
    
    def move_left(self, event):
        axis = 0
        current = self.viewer.dims.current_step[axis]
        self.viewer.dims.set_current_step(axis, current - 1)
        print("Moving left")

    def move_right(self, event):
        axis = 0
        current = self.viewer.dims.current_step[axis]
        self.viewer.dims.set_current_step(axis, current + 1)
        print("Moving right")

    def reduce_brush_size(self, event):
        self._adjust_brush_size(-1)

    def increase_brush_size(self, event):
        self._adjust_brush_size(1)

    def _adjust_brush_size(self, delta):
        layer = self.model._layer_labels
        if hasattr(layer, 'brush_size'):
            new_size = max(1, layer.brush_size + delta)  # Ensure brush size doesn't go below 1
            layer.brush_size = new_size
            print(f"Brush size changed to: {new_size}")

    def _toggle_labels_paint(self, event):
        
        self.viewer.dims.ndisplay = 2
        labels_layer = self.model._layer_labels
        is_active = self.viewer.layers.selection.active == labels_layer
        is_paint_mode = labels_layer.mode == 'paint'

        if is_active and is_paint_mode:
            labels_layer.mode = 'pan_zoom'  # Deactivate paint tool
            print("Deactivated paint brush.")
        else:
            self.viewer.layers.selection.active = labels_layer
            labels_layer.mode = 'paint'  # Activate paint tool
            print("Activated paint brush.")

    def _toggle_2D3D(self, event):

        labels_layer = self.model._layer_labels
        is_active = self.viewer.layers.selection.active == labels_layer
        is_paint_mode = labels_layer.mode == 'paint'

        n_display = self.viewer.dims.ndisplay

        if is_active and is_paint_mode:
            labels_layer.mode = 'pan_zoom'
        
        if n_display == 3:
            self._camera = self.viewer.camera.angles
            self.viewer.dims.ndisplay = 2
            self._toggle_labels_paint(None)
        else:
            self.viewer.dims.ndisplay = 3
            self.viewer.camera.angles = self._camera

    def undo(self, event):

        # def simulate_ctrl_z(self):
        #     widget = self.viewer.window._qt_viewer
        #     event = QKeyEvent(QEvent.KeyPress, Qt.Key_Z, Qt.ControlModifier)
        #     QApplication.sendEvent(widget, event)

        # # Simulate Ctrl+Z (Undo)
        # simulate_ctrl_z()  # Internal napari Qt method (not public API, but works)
        
        # Do your custom action
        self.model._layer_labels.selected_label = max(1, self._n_points_original, self.model._layer_labels.selected_label - 1)
        self._n_points = max(0, self._n_points_original, self._n_points - 1)
        event.handled = False
        print("Undo")

