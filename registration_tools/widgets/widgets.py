import napari
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QVBoxLayout, QWidget, QSlider, QLabel, QPushButton, QLineEdit, QHBoxLayout, QGridLayout
from qtpy.QtCore import Qt, QTimer
from napari.utils.events import Event
from vispy.util.keys import ALT, CONTROL

from ..utils.auxiliar import *

REGISTRATION_TYPES = [
    "translation",
    "translation2D",
    "translation3D",
    "rotation",
    "rotation2D",
    "rotation3D",
    "rigid",
    "rigid2D",
    "rigid3D",
]

REGISTRATION_TYPES_TODO = [
    "affine",
    "affine2D",
    "affine3D",
]

class AffineRegistrationWidget(QWidget):
    
    def __init__(self, model, viewer, axis):
        super().__init__()

        if model._registration_type in REGISTRATION_TYPES_TODO:
            raise TypeError(f"{model._registration_type} registration type is not yet supported for manual but will be added at some point.")
        elif model._registration_type not in REGISTRATION_TYPES:
            raise TypeError(f"{model._registration_type} registration type is not supported for manual registration.")

        self.viewer = viewer
        self.model = model
        self.axis = axis

        self._registration_model = model
        self._registration_model._n_spatial = model._n_spatial
        self._registration_model._spatial_shape = model._spatial_shape
        self._last_t = viewer.dims.current_step[axis.index("T")]
        self.setWindowTitle("Rotation Widget")
        self.layout = QVBoxLayout()
        self.explanation = self.create_explanation()
        self.register_button = self.create_button("Register", self.register)
        self.rotation_slider = self.create_slider("Rotation step", 1, 360, 1)
        self.translation_slider = self.create_slider("Translation Step", 1, np.min(model._spatial_shape), min(1,np.min(model._spatial_shape)//100))
        self.propagate_button = self.create_button("Propagate", self.propagate)
        self.make_limits()
        self.setLayout(self.layout)
        self.save = self.create_button("Save", self.save)
        self.close = self.create_button("Close", self.close)

        if self.model._n_spatial == 2:
            faces = self.bounding_box(0, self.model._spatial_shape[0], 0, self.model._spatial_shape[1])
        else:
            faces = self.bounding_box(0, self.model._spatial_shape[0], 0, self.model._spatial_shape[1], 0, self.model._spatial_shape[2])
        self.viewer.layers["Original Bounding Box"].data = faces

        # viewer.mouse_drag_callbacks.append(self.on_mouse_drag)
        self.viewer.bind_key("Shift+W",self.on_keyboard_up_translate)
        self.viewer.bind_key("Shift+S",self.on_keyboard_down_translate)
        self.viewer.bind_key("Shift+A",self.on_keyboard_left_translate)
        self.viewer.bind_key("Shift+D",self.on_keyboard_right_translate)
        self.viewer.bind_key("Shift+Q",self.on_keyboard_counterclockwise_rotate)
        self.viewer.bind_key("Shift+E",self.on_keyboard_clockwise_rotate)
        self.viewer.bind_key("Shift+R",self.register)
        self.viewer.dims.events.current_step.connect(self.update_images)
        # self.viewer.window._qt_viewer.canvas.setFocus()

    def create_explanation(self):
        widget = QWidget()
        layout = QVBoxLayout()
        label = QLabel(
            """
            Shift + W: Translate up.
            Shift + S: Translate down.
            Shift + A: Translate left.
            Shift + D: Translate right.
            Shift + Q: Rotate counterclockwise.
            Shift + E: Rotate clockwise.

            Shift + R: Register the transformation.
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

        self.viewer.bind_key("Shift+W", None)
        self.viewer.bind_key("Shift+S", None)
        self.viewer.bind_key("Shift+A", None)
        self.viewer.bind_key("Shift+D", None)
        self.viewer.bind_key("Shift+Q", None)
        self.viewer.bind_key("Shift+E", None)
        self.viewer.bind_key("Shift+R", None)
        self.viewer.dims.events.current_step.disconnect(self.update_images)
        QTimer.singleShot(0, self.parent().close)
        self.viewer.close()

    def create_button(self, label_text, callback):
        button = QPushButton(label_text)
        button.clicked.connect(callback)
        self.layout.addWidget(button)
        return button

    def create_slider(self, label_text, min_value, max_value, set_value):
        widget = QWidget()
        layout = QVBoxLayout()
        label = QLabel(f"{label_text}: {set_value}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(set_value)
        slider.valueChanged.connect(lambda value, l=label: l.setText(f"{label_text}: {value}"))
        layout.addWidget(label)
        layout.addWidget(slider)
        widget.setLayout(layout)
        self.layout.addWidget(widget)
        return widget

    def make_limits(self):
        coord_widget = QWidget()
        coord_layout = QGridLayout()
        coord_layout.setContentsMargins(0, 0, 0, 0)
        coord_layout.setSpacing(5)

        # Row 0: Min values
        self.minX_input, self.minX_field = self.create_labeled_input("Min X", 0)
        self.minY_input, self.minY_field = self.create_labeled_input("Min Y", 0)
        if self._registration_model._n_spatial == 3:
            self.minZ_input, self.minZ_field = self.create_labeled_input("Min Z", 0)

        coord_layout.addWidget(self.minX_input, 0, 0)
        coord_layout.addWidget(self.minY_input, 0, 1)
        if self._registration_model._n_spatial == 3:
            coord_layout.addWidget(self.minZ_input, 0, 2)

        # Row 1: Max values
        self.maxX_input, self.maxX_field = self.create_labeled_input("Max X", self._registration_model._spatial_shape[0])
        self.maxY_input, self.maxY_field = self.create_labeled_input("Max Y", self._registration_model._spatial_shape[1])
        if self._registration_model._n_spatial == 3:
            self.maxZ_input, self.maxZ_field = self.create_labeled_input("Max Z", self._registration_model._spatial_shape[2])

        coord_layout.addWidget(self.maxX_input, 1, 0)
        coord_layout.addWidget(self.maxY_input, 1, 1)
        if self._registration_model._n_spatial == 3:
            coord_layout.addWidget(self.maxZ_input, 1, 2)

        coord_widget.setLayout(coord_layout)
        self.layout.addWidget(coord_widget)

        # Connect input changes to bounding_box
        self.minX_field.textChanged.connect(self.update_bounding_box)
        self.minY_field.textChanged.connect(self.update_bounding_box)
        self.maxX_field.textChanged.connect(self.update_bounding_box)
        self.maxY_field.textChanged.connect(self.update_bounding_box)
        if self._registration_model._n_spatial == 3:
            self.minZ_field.textChanged.connect(self.update_bounding_box)
            self.maxZ_field.textChanged.connect(self.update_bounding_box)

        # Initial box
        self.update_bounding_box()

    def create_labeled_input(self, label_text, default_value):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        label = QLabel(label_text)
        label.setFixedWidth(45)
        input_field = QLineEdit()
        input_field.setText(str(default_value))
        input_field.setFixedWidth(60)

        layout.addWidget(label)
        layout.addWidget(input_field)
        widget.setLayout(layout)

        return widget, input_field  # Return both

    def bounding_box(self, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None):

        if self.model._n_spatial == 3:
            # Define the 8 corners of the box
            corners = np.array([
                [z_min, y_min, x_min],
                [z_max, y_min, x_min],
                [z_max, y_max, x_min],
                [z_min, y_max, x_min],
                [z_min, y_min, x_max],
                [z_max, y_min, x_max],
                [z_max, y_max, x_max],
                [z_min, y_max, x_max],
            ])

            # Define the 6 faces using 4 vertices each (polygons)
            faces = [
                [corners[0], corners[1], corners[2], corners[3]],  # Bottom
                [corners[4], corners[5], corners[6], corners[7]],  # Top
                [corners[0], corners[1], corners[5], corners[4]],  # Front
                [corners[2], corners[3], corners[7], corners[6]],  # Back
                [corners[1], corners[2], corners[6], corners[5]],  # Right
                [corners[3], corners[0], corners[4], corners[7]],  # Left
            ]
        else:
            # Define the 8 corners of the box
            corners = np.array([
                [y_min, x_min],
                [y_max, x_min],
                [y_max, x_max],
                [y_min, x_max],
            ])

            # Define the 6 faces using 4 vertices each (polygons)
            faces = [
                [corners[0], corners[3]],  # Bottom
                [corners[1], corners[2]],  # Top
                [corners[2], corners[3]],  # Right
                [corners[0], corners[1]],  # Left
            ]


        return np.array(faces)
    
    def update_bounding_box(self):

        # Get coordinates from UI
        x_min = float(self.minX_input.findChild(QLineEdit).text())
        x_max = float(self.maxX_input.findChild(QLineEdit).text())
        y_min = float(self.minY_input.findChild(QLineEdit).text())
        y_max = float(self.maxY_input.findChild(QLineEdit).text())
        z_min = float(self.minZ_input.findChild(QLineEdit).text()) if self._registration_model._n_spatial == 3 else 0
        z_max = float(self.maxZ_input.findChild(QLineEdit).text()) if self._registration_model._n_spatial == 3 else 0

        faces = self.bounding_box(x_min, x_max, y_min, y_max, z_min, z_max)

        self.viewer.layers["Bounding Box"].data = faces

    def register(self,*args):
        t = self.viewer.dims.current_step[self.axis.index("T")]

        if self.model._registration_direction == "forward":
            t += 1
            t_next = t-1
            c = -1
        else:
            t_next = t+1
            c = 0
        
        if t_next < 0 or t_next >= self.model._t_max:
            return

        d_ref = {"T":t+c}
        d_float = {"T":t+c}
        if "C" in self.axis:
            d_ref["C"] = self.viewer.dims.current_step[self.axis.index("C")]
            d_float["C"] = self.viewer.dims.current_step[self.axis.index("C")]

        slicing_ref = make_index(self.axis,**d_ref)
        slicing_float = make_index(self.axis,**d_float)

        img_ref = self._layer_dataset.data[slicing_ref]
        img_float = self._layer_next.data[slicing_float]
        trnsf_ref = self.model._load_transformation_global(t, self.model._origin)
        trnsf_float = self.model._load_transformation_global(t_next, self.model._origin)

        # Scaling matrices (fixing order of operations)
        img_ref = self.model.apply_trnsf(img_ref, trnsf_ref, self.model._scale, self.model._padding)
        img_float = self.model.apply_trnsf(img_float, trnsf_float, self.model._scale, self.model._padding)

        trnsf = self._registration_model.register(
            img_float, 
            img_ref, 
            self.model._scale, 
            verbose=True
        )

        self.model._save_transformation_relative(trnsf, t_next, self.model._origin)
        self.update_images(None, refresh=True)

    def update_images(self, event, refresh=False):
        # Get current time step
        t = self.viewer.dims.current_step[self.axis.index("T")]
        if self.model._registration_direction == "forward":
            t += 1
            t_next = t-1
        else:
            t_next = t+1

        # Check if t has actually changed
        if hasattr(self, "_last_t") and self._last_t == t and not refresh:
            return  # Skip update if t is the same as before

        self.model.propagate()

        # Store the new value of t
        self._last_t = t

        # Now update the affine transformations only if necessary
        self._layer_corrected.affine = np.linalg.inv(self.model._load_transformation_global(t, self.model._origin))
        self._layer_next.affine = np.linalg.inv(self.model._load_transformation_global(t_next, self.model._origin))

    def on_mouse_drag_major(self, viewer, event: Event):
        """Rotate dataset about its center around the camera view direction,
        accumulating the rotation incrementally.
        
        The rotation axis is given by self.viewer.camera.view_direction.
        """
        # Only operate if ALT is held (and CONTROL is not)
        if ALT not in event.modifiers or CONTROL in event.modifiers:
            return

        # Yield once to start listening for drag events
        yield

        # Initialize the starting mouse position and an accumulated transform.
        pos0 = None
        layer = self._layer_next

        while event.type == "mouse_move":
            if pos0 is None:
                pos0 = event.pos
            else:
                accumulated_affine = layer.affine.affine_matrix[2:,2:]

                # Compute the center of the dataset.
                # For a 3D image, we assume layer.data.shape is (z, y, x) and compute center in (x, y, z).
                center = np.array(layer.data.shape[-3:], dtype=float) / 2.0
                center = center*layer.scale[-3:]

                y = self.viewer.camera.up_direction
                y = y / np.linalg.norm(y)
                z= self.viewer.camera.view_direction
                z = z / np.linalg.norm(z)
                x = np.cross(y, z)
                x = x / np.linalg.norm(x)

                delta = (event.pos - pos0)/10

                t = self.viewer.dims.current_step[self.axis.index("T")]
                v1 = self.model._axis1[t, 1, :]
                v2 = v1 + x * delta[0] - y * delta[1]
                self.model._axis1[t, 1, :] = self.model._axis1[t, 1, :] / np.linalg.norm(self.model._axis1[t, 1, :]) * self.model._arrow_scale
                
                v1 /= np.linalg.norm(v1)
                v2 /= np.linalg.norm(v2)
                axis = np.cross(v1, v2)

                if np.linalg.norm(axis) < 1e-6:
                    # No rotation is possible if the vectors are parallel.
                    # Skip the rest of the loop and wait for the next event.
                    pos0 = event.pos
                else:
                    axis /= np.linalg.norm(axis)
                    angle_delta = np.arccos(np.dot(v1, v2)) * 0.01                

                    # For 3D rotation, build the 4×4 incremental transform.
                    # First, translate the center to the origin:
                    T_translate = np.eye(4)
                    T_translate[:3, 3] = -center
                    # Then, translate back:
                    T_back = np.eye(4)
                    T_back[:3, 3] = center

                    # Build the 3×3 rotation matrix using Rodrigues' formula.
                    I = np.eye(3)
                    K = np.array([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]
                    ])
                    R_3 = I * np.cos(angle_delta) + np.sin(angle_delta) * K + (1 - np.cos(angle_delta)) * np.outer(axis, axis)
                    # Embed the 3×3 rotation into a 4×4 homogeneous matrix.
                    R = np.eye(4)
                    R[:3, :3] = R_3

                    # Compute the incremental transformation: move to origin, rotate, and move back.
                    incremental_transform = T_back @ R @ T_translate

                    # Accumulate the transform (apply the new incremental transform on top of the previous one)
                    accumulated_affine = incremental_transform @ accumulated_affine

                    # Update the layer’s affine transform
                    layer.affine = accumulated_affine

                    # Update the starting position for the next delta computation.
                    pos0 = event.pos

            # Yield control back to napari so it can process other events.
            yield

    def propagate(self):
        if self.model._registration_direction == "forward":
            self.propagate_backwards()
        elif self.model._registration_direction == "backward":
            self.propagate_forwards()

    def propagate_forwards(self):
        t = self.viewer.dims.current_step[self.axis.index("T")]
        self.model.propagate()
        self.viewer.dims.set_current_step(self.axis.index("T"), min(self.model._t_max - 1, t + 1))
    
    def propagate_backwards(self):
        t = self.viewer.dims.current_step[self.axis.index("T")]
        self.model.propagate()
        self.viewer.dims.set_current_step(self.axis.index("T"), max(0, t - 1))

    def rotate(self, layer, axis):
        t = self.viewer.dims.current_step[self.axis.index("T")]   

        if self.model._registration_direction == "forward":
            t_next = t-1
        else:
            t_next = t+1

        if t_next < 0 or t_next >= self.model._t_max:
            return

        global_affine = self.model._load_transformation_global(t, self.model._origin)
        local_affine = self.model._load_transformation_relative(t_next, self.model._origin)

        if self.model._n_spatial == 2:
            axis = (0,0,np.sum(axis))
        angle = np.radians(self.rotation_slider.findChild(QSlider).value())
        center = np.array(layer.data.shape[-self.model._n_spatial:], dtype=float) / 2.0
        center = center*layer.scale[-self.model._n_spatial:]
        # For 3D rotation, build the 4×4 incremental transform.
        # First, translate the center to the origin:
        T_translate = np.eye(self.model._n_spatial+1)
        T_translate[:self.model._n_spatial, self.model._n_spatial] = -center
        # Then, translate back:
        T_back = np.eye(self.model._n_spatial+1)
        T_back[:self.model._n_spatial, self.model._n_spatial] = center
        # Build the 3×3 rotation matrix using Rodrigues' formula.
        I = np.eye(self.model._n_spatial)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])[:self.model._n_spatial,:self.model._n_spatial]
        R_3 = I * np.cos(angle) + np.sin(angle) * K + (1 - np.cos(angle)) * np.outer(axis, axis)[:self.model._n_spatial,:self.model._n_spatial]
        # Embed the 3×3 rotation into a 4×4 homogeneous matrix.
        R = np.eye(self.model._n_spatial+1)
        R[:self.model._n_spatial, :self.model._n_spatial] = R_3
        # Compute the incremental transformation: move to origin, rotate, and move back.
        incremental_transform = T_back @ R @ T_translate
        # Accumulate the transform (apply the new incremental transform on top of the previous one)
        local_affine = incremental_transform @ local_affine
        self.model._save_transformation_relative(local_affine, t_next, self.model._origin)

        # Properly assign the new affine transformation
        accumulated_affine = self.model.compose_trnsf([global_affine, local_affine])
        layer.affine = np.linalg.inv(accumulated_affine)

    def translate(self, layer, axis):
        t = self.viewer.dims.current_step[self.axis.index("T")]   

        if self.model._registration_direction == "forward":
            t_next = t-1
        else:
            t_next = t+1

        if t_next < 0 or t_next >= self.model._t_max:
            return

        if self.model._n_spatial == 2:
            axis = axis[1:]
        
        global_affine = self.model._load_transformation_global(t, self.model._origin)
        local_affine = self.model._load_transformation_relative(t_next, self.model._origin)

        # Build a 6×6 translation matrix
        step = self.translation_slider.findChild(QSlider).value()
        T_translate = np.eye(self.model._n_spatial+1)  # Identity matrix
        T_translate[:self.model._n_spatial, -1] = np.array(axis) * step  # Apply translation correctly

        # Accumulate the transformation
        local_affine = T_translate @ local_affine  # Ensure correct order
        self.model._save_transformation_relative(local_affine, t_next, self.model._origin)

        # Properly assign the new affine transformation
        accumulated_affine = self.model.compose_trnsf([global_affine, local_affine])
        layer.affine = np.linalg.inv(accumulated_affine)

    def on_keyboard_clockwise_rotate(self, viewer):
        """Rotate dataset about its center around the camera view direction,
        accumulating the rotation incrementally.
        
        The rotation axis is given by self.viewer.camera.view_direction.
        """
        layer = self._layer_next

        v = np.array(self.viewer.camera.view_direction)
        # print("clockwise", v)

        self.rotate(layer, v)

    def on_keyboard_counterclockwise_rotate(self, viewer):
        """Rotate dataset about its center around the camera view direction,
        accumulating the rotation incrementally.
        
        The rotation axis is given by self.viewer.camera.view_direction.
        """
        layer = self._layer_next

        v = np.array(self.viewer.camera.view_direction)
        # print("counterclockwise", v)

        self.rotate(layer, -v)

    def on_keyboard_up_translate(self, viewer):
        """Rotate dataset about its center around the camera view direction,
        accumulating the rotation incrementally.
        
        The rotation axis is given by self.viewer.camera.view_direction.
        """
        layer = self._layer_next

        v = self.up_direction()
        # print("up", v)

        self.translate(layer, -v)

    def on_keyboard_down_translate(self, viewer):
        """Rotate dataset about its center around the camera view direction,
        accumulating the rotation incrementally.
        
        The rotation axis is given by self.viewer.camera.view_direction.
        """
        layer = self._layer_next

        v = self.up_direction()
        # print("down", v)

        self.translate(layer, v)

    def on_keyboard_right_translate(self, viewer):
        """Rotate dataset about its center around the camera view direction,
        accumulating the rotation incrementally.
        
        The rotation axis is given by self.viewer.camera.view_direction.
        """
        layer = self._layer_next

        v = self.right_direction()
        # print("right", v)

        self.translate(layer, -v)

    def on_keyboard_left_translate(self, viewer):
        """Rotate dataset about its center around the camera view direction,
        accumulating the rotation incrementally.
        
        The rotation axis is given by self.viewer.camera.view_direction.
        """
        layer = self._layer_next

        v = self.right_direction()
        # print("left", v)

        self.translate(layer, v)

    def up_direction(self):
        x = self.viewer.camera.up_direction
        x = x / np.linalg.norm(x)
        return x

    def right_direction(self):
        y = self.viewer.camera.up_direction
        y = y / np.linalg.norm(y)
        z = self.viewer.camera.view_direction
        z = z / np.linalg.norm(z)
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)
        return x

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QLineEdit, QPushButton, QFileDialog, QMessageBox, QHBoxLayout
)
import os

class SaveDialog(QDialog):
    def __init__(self, model, viewer):
        super().__init__()
        self.model = model
        self.viewer = viewer
        self.setWindowTitle("Save to Directory")
        self.setMinimumWidth(400)
        self.init_ui()
        self.setModal(True)  # Block interaction with the main window

    def init_ui(self):
        layout = QVBoxLayout()

        # Directory input + browse
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Select a directory...")

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_directory)

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(browse_button)
        layout.addLayout(dir_layout)

        # Action buttons
        save_button = QPushButton("Create Directory and Save")
        close_button = QPushButton("Close Without Saving")

        save_button.clicked.connect(self.save_and_close)
        close_button.clicked.connect(self.reject)

        layout.addWidget(save_button)
        layout.addWidget(close_button)

        self.setLayout(layout)

    def browse_directory(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            self.dir_input.setText(path)

    def save_and_close(self):
        path = self.dir_input.text()
        if not path:
            QMessageBox.warning(self, "No Directory", "Please select a directory.")
            return

        if not os.path.exists(path):
            self.model._out = path
        elif not os.listdir(path):
            self.model._out = path
        else:
            QMessageBox.warning(self, "Directory not empty", "Please select an empty directory.")

        # Optional: save logic here
        self.accept()

# Function to show the dialog from Napari
def show_save_popup(model, viewer):
    dialog = SaveDialog(model, viewer)
    dialog.exec_()

class CloseDialog(QDialog):
    def __init__(self, model, viewer):
        super().__init__()
        self.model = model
        self.viewer = viewer
        self.setWindowTitle("Close: Files not saved")
        self.setMinimumWidth(400)
        self.init_ui()
        self.setModal(True)  # Block interaction with the main window

    def init_ui(self):
        layout = QVBoxLayout()

        # Action buttons
        save_button = QPushButton("Save")
        close_button = QPushButton("Close Without Saving")

        save_button.clicked.connect(self.save_and_close)
        close_button.clicked.connect(self.reject)

        layout.addWidget(save_button)
        layout.addWidget(close_button)

        self.setLayout(layout)

    def save_and_close(self):

        show_save_popup(self.model, self.viewer)

        # Optional: save logic here
        self.accept()

# Function to show the dialog from Napari
def show_close_popup(model, viewer):
    dialog = CloseDialog(model, viewer)
    dialog.exec_()
