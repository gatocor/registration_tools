import inspect
from skimage import feature
import numpy as np
import webbrowser
import contextlib
import io

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel,
    QFormLayout, QDoubleSpinBox, QCheckBox, QPushButton, QSpinBox, QGroupBox, QLineEdit
)
import napari

default_map = {
    ("h_maxima", "h"): 0,
    ("h_minima", "h"): 0,
    ("isotropic_closing", "h"): 1.0,
    ("isotropic_dilation", "h"): 1.0,
    ("isotropic_erosion", "h"): 1.0,
    ("isotropic_opening", "h"): 1.0,
}

def get_help_text(obj):
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        help(obj)
        return buf.getvalue()

def get_skimage_feature_functions():
    funcs = {}
    for name, func in inspect.getmembers(feature, inspect.isfunction):
        sig = inspect.signature(func)
        params = sig.parameters

        # Must have "image" parameter
        if "image" not in params:
            continue

        # Count how many required parameters there are (no default)
        required_params = [
            p for p in params.values()
            if p.default is inspect.Parameter.empty
        ]

        if len(required_params) == 0 and [i for i in params.values()][0].name == "image":
            funcs[name] = func
        elif len(required_params) == 1 and required_params[0].name == "image":
            funcs[name] = func
        elif name in [i[0] for i in default_map.keys()]:
            funcs[name] = func
        else:
            print(f"Skipping {name}: requires multiple parameters or has non-image required parameters ({required_params}). To be implemented.")

    return funcs

filter_func = get_skimage_feature_functions()

class FeatureWidget(QWidget):
    def __init__(self, viewer: napari.Viewer, filter_name: str):
        super().__init__()
        self.viewer = viewer
        self.filter_func = filter_func[filter_name]
        self.setWindowTitle(f"Feature: {filter_name}")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.help_text = get_help_text(self.filter_func)

        # --- Name and help ---
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel(filter_name))
        self.help_button = QPushButton("?")
        self.help_button.setMinimumHeight(10)
        self.help_button.setToolTip(self.help_text)
        self.help_button.clicked.connect(lambda: webbrowser.open(f"https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.{filter_name}"))
        name_row.addWidget(self.help_button)
        self.layout.addLayout(name_row)

        # --- Image selector ---
        image_row = QHBoxLayout()
        image_row.addWidget(QLabel("Image:"))
        self.image_selector = QComboBox()
        self.update_image_layers()
        image_row.addWidget(self.image_selector)
        self.layout.addLayout(image_row)

        # --- Output name input ---
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output name:"))
        self.output_name_edit = QLineEdit(f"{self.image_selector.currentText()}_{filter_name}")
        output_row.addWidget(self.output_name_edit)
        output_row.addWidget(QLabel("Override:"))
        self.output_override = QCheckBox()
        self.output_override.setChecked(True)
        output_row.addWidget(self.output_override)
        self.layout.addLayout(output_row)

        # --- Feature parameters ---
        self.form = QFormLayout()
        self.param_widgets = {}

        sig = inspect.signature(self.filter_func)
        for name, param in sig.parameters.items():
            if name == "image":
                continue

            widget = None

            annotation = param.annotation

            if param.default is not inspect.Parameter.empty:
                default = param.default    
            elif (filter_name, param.name) in default_map.keys():
                default = default_map[(filter_name, name)]    
                annotation = type(default)
            else:
                default = None

            if param.annotation == inspect._empty:
                annotation = type(default)

            # print(param.name, param.annotation, default)
            if annotation == float:
                widget = QDoubleSpinBox()
                widget.setValue(float(default))
                widget.setSingleStep(0.1)
                widget.setRange(-1e6, 1e6)
                self.form.addRow(name, widget)
            elif annotation == int:
                widget = QSpinBox()
                widget.setValue(int(default))
                widget.setRange(-10000, 10000)
                self.form.addRow(name, widget)
            elif annotation == bool:
                widget = QCheckBox()
                widget.setChecked(bool(default))
                self.form.addRow(name, widget)
            elif annotation == range and isinstance(default, range):
                widget_start = QSpinBox()
                widget_start.setRange(-10000, 10000)
                widget_start.setValue(default.start)

                widget_stop = QSpinBox()
                widget_stop.setRange(-10000, 10000)
                widget_stop.setValue(default.stop)

                widget_step = QSpinBox()
                widget_step.setRange(1, 10000)  # step must be positive
                widget_step.setValue(default.step)

                # Store the three widgets
                self.param_widgets[name] = (widget_start, widget_stop, widget_step)

                # Create a container group box
                widget = QGroupBox(name)
                sub_layout = QFormLayout()
                sub_layout.addRow("Start", widget_start)
                sub_layout.addRow("Stop", widget_stop)
                sub_layout.addRow("Step", widget_step)
                widget.setLayout(sub_layout)
                self.form.addRow(widget)
            else:
                print(f"Unsupported parameter type for {name}: {annotation} {default}")
                continue  # skip unsupported types

            self.param_widgets[name] = widget

        self.layout.addLayout(self.form)

        # --- Apply button ---
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_filter)
        self.layout.addWidget(self.apply_btn)

    def update_image_layers(self):
        self.image_selector.clear()
        image_layers = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
        self.image_selector.addItems(image_layers)

    def apply_filter(self):
        layer_name = self.image_selector.currentText()
        layer = self.viewer.layers[layer_name] if layer_name in self.viewer.layers else None
        if layer is None:
            return

        image = layer.data
        kwargs = {}
        for name, widget in self.param_widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                kwargs[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                kwargs[name] = widget.isChecked()

        result = self.filter_func(image, **kwargs)
        output_name = self.output_name_edit.text().strip() or f"{self.windowTitle()} result"
        if self.output_override.isChecked():
            if output_name in self.viewer.layers:
                self.viewer.layers.remove(output_name)
        self.viewer.add_image(result, name=output_name, colormap="gray")

class FeatureSelector(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.setWindowTitle("Skimage Feature Selector")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        row = QHBoxLayout()
        row.addWidget(QLabel("Choose a filter:"))
        self.dropdown = QComboBox()
        self.feature = get_skimage_feature_functions()
        self.dropdown.addItems(sorted(self.feature.keys()))
        row.addWidget(self.dropdown)
        self.layout.addLayout(row)

        self.load_btn = QPushButton("Load Feature UI")
        self.load_btn.clicked.connect(self.load_filter_ui)
        self.layout.addWidget(self.load_btn)

    def load_filter_ui(self):
        name = self.dropdown.currentText()
        func = self.feature[name]
        widget = FeatureWidget(self.viewer, name, func)
        self.viewer.window.add_dock_widget(widget, name=f"Feature: {name}", area="right")
