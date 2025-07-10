from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLineEdit, QHBoxLayout,
    QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsLineItem, QMenu, QLabel, QComboBox,
    QFormLayout, QDoubleSpinBox, QCheckBox, QSpinBox, QGroupBox, QFileDialog, QMessageBox
)
from qtpy.QtGui import QBrush, QPen, QColor
from qtpy.QtCore import Qt, QPoint
import inspect
import webbrowser
import contextlib
import io
import numpy as np
from numpy import uint8, uint32, int8, int16, int32, float16, float32, float64
import json
import pandas as pd
import os
from skimage.io import imread
import napari

def register_nodes(d, d_partial):
    for names in d_partial.keys():
        if names not in d:
            d[names] = {}
        for subnames, item in d_partial[names].items():
            if subnames not in d[names]:
                d[names][subnames] = item

from .nodes_skimage import NODES_SKIMAGE
from .nodes_registration_tools import NODES_REGISTRATION_TOOLS
NODE_TYPES = {} 
register_nodes(NODE_TYPES, NODES_REGISTRATION_TOOLS)
register_nodes(NODE_TYPES, NODES_SKIMAGE)

class Node(QGraphicsRectItem):  # Still extends QGraphicsRectItem for now
    def __init__(self, meta, main, color, shape, pos=(0,0)):
        self.main = main
        self.edges = []
        self.meta = meta
        self.widget = None
        self.results = {}

        super().__init__(*shape)  # circular shape
        self.setBrush(QBrush(QColor(color)))

        self.setPen(QPen(Qt.black))
        self.setPos(*pos)
        self.setFlag(self.ItemIsMovable)
        self.setFlag(self.ItemSendsGeometryChanges)
        self.setToolTip(self.meta["name"])

        self.text = QGraphicsTextItem(self.meta["name"], self)
        self.text.setDefaultTextColor(Qt.black)
        self.text.setPos(shape[0],shape[1])

    def add_edge(self, edge):
        self.edges.append(edge)

    def itemChange(self, change, value):
        if change == self.ItemPositionChange:
            for edge in self.edges:
                edge.update_position()
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

        # Reset border color for previously selected node
        if self.main.selected_node:
            self.main.selected_node.setPen(QPen(Qt.black))

        # Set this node as selected and highlight its border in red
        self.main.selected_node = self
        self.setPen(QPen(QColor("red"), 2))  # red border, thicker line

        self.main.load(self.meta)

    def mark_dirty(self):
        self.setBrush(QBrush(QColor("yellow")))

    def mark_clean(self):
        if self.meta["type"] == "image":
            self.setBrush(QBrush(QColor("lightblue")))
        else:
            self.setBrush(QBrush(QColor("lightgreen")))

class NodeData(Node):
    def __init__(self, name, image, flow_widget, pos=(0, 0)):
        try:
            shape = image.shape
            dtype = image.dtype
            default_axes = "YXZCT"[:len(shape)]
            default_scale = "("+("1," * len(shape))+")"
        except Exception:
            shape = None
            dtype = None
            default_axes = ""
            default_scale = ""

        meta = {   
            "name": name,
            "type": "image",
            "dtype": dtype,
            "shape": shape,
            "axes": default_axes,
            "scale": default_scale
        }

        super().__init__(meta, flow_widget, "lightblue", (-100, -100, 100, 100), pos=pos)
        self.axes_input = None
        self.scale_inputs = []

    def display_widget(self):
        """
        Returns a widget to display the image data.
        """
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        title = QLabel(f"Dataset: {self.meta['name']}")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)

        layout.addWidget(QLabel(f"Shape: {self.meta['shape']}"))
        layout.addWidget(QLabel(f"Dtype: {self.meta['dtype']}"))

        # --- Axes ---
        axes_row = QHBoxLayout()
        axes_row.addWidget(QLabel("Axes:"))
        self.axes_input = QLineEdit(self.meta.get("axes", ""))
        self.axes_input.setMaxLength(5)
        self.axes_input.setToolTip("Specify axis order using only letters from XYZCT (e.g. 'YXC', 'XYT')")
        self.axes_input.editingFinished.connect(self.update_axes)
        axes_row.addWidget(self.axes_input)
        layout.addLayout(axes_row)

        # --- Scale ---
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale:"))
        self.scale_input = QLineEdit(self.meta.get("scale", ""))
        self.scale_input.setMaxLength(500)
        self.scale_input.setToolTip("Specify scale of each axis. It must be the same size as the number of axes. (e.g. if aixis 'XYCT' then scale must be '(0.1,0.1,1,1)')")
        self.scale_input.editingFinished.connect(self.update_scale)
        scale_row.addWidget(self.scale_input)
        layout.addLayout(scale_row)

        self.widget = widget
        return widget

    def update_axes(self):
        new_axes = self.axes_input.text().upper()
        valid_letters = set("XYZCT")
        shape_len = len(self.meta.get("shape", []))

        if len(new_axes) != shape_len:
            QMessageBox.warning(None, "Invalid Axes",
                f"Length of axes must match number of dimensions: {shape_len}")
            self.axes_input.setText(self.meta["axes"])
            return

        if len(set(new_axes)) != len(new_axes) or not all(c in valid_letters for c in new_axes):
            QMessageBox.warning(None, "Invalid Axes",
                "Axes must contain unique characters from XYZCT.")
            self.axes_input.setText(self.meta["axes"])
            return

        self.meta["axes"] = new_axes

    def update_scale(self):
        self.meta["scale"] = [spin.value() for spin in self.scale_inputs]

class NodeFunction(Node):
    def __init__(self, meta, flow_widget, pos=(0, 0)):

        super().__init__(meta, flow_widget, "yellow", (-100, -50, 100, 50), pos=pos)

class Edge(QGraphicsLineItem):
    def __init__(self, source_node, dest_node, color="lightgray", width=4):
        super().__init__()
        self.source = source_node
        self.dest = dest_node
        self.pen = QPen(QColor(color), width)
        self.setPen(self.pen)
        self.setZValue(-1)
        self.update_position()
        self.source.add_edge(self)
        self.dest.add_edge(self)

    def update_position(self):
        src_pos = self.source.scenePos()
        dst_pos = self.dest.scenePos()
        self.setLine(src_pos.x(), src_pos.y(), dst_pos.x(), dst_pos.y())

def get_help_text(obj):
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        help(obj)
        return buf.getvalue()

class WidgetData(QWidget):
    def __init__(self, main, node: dict):
        super().__init__()
        self.main = main
        self.node = node
        self.setWindowTitle(f"Filter: {node.meta['name']}")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        title = QLabel(f"Dataset: {node.meta['name']}")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.layout.addWidget(title)

        shape_label = QLabel(f"Shape: {node.meta['shape']}")
        dtype_label = QLabel(f"Dtype: {node.meta['dtype']}")
        
        self.layout.addWidget(shape_label)
        self.layout.addWidget(dtype_label)

class WidgetFunction(QWidget):
    def __init__(self, main, node: dict):
        super().__init__()
        self.main = main
        self.filter_func = node["function"]
        self.node = node
        self.changed = False
        self.setWindowTitle(f"Filter: {node['name']}")
        self.node_added = self.node["name"] in self.main.nodes

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
            
        self.help_text = get_help_text(self.filter_func)
        if "help" in node:
            self.help_link = node["help"]
        else:
            self.help_link = f"Documentation for {node['name']} not available."

        # --- Name and help ---
        name_row = QHBoxLayout()
        # name_row.addWidget(QLabel(node["name"]))
        self.name0 = node["name"]
        self.name = QLineEdit()
        self.name.setText(self.name0)
        name_row.addWidget(self.name)
        self.help_button = QPushButton("?")
        self.help_button.setMinimumHeight(10)
        self.help_button.setToolTip(self.help_text)
        self.help_button.clicked.connect(lambda: webbrowser.open(self.help_link))
        name_row.addWidget(self.help_button)
        self.layout.addLayout(name_row)

        # --- Input ---
        self.image_selector = {}
        if "inputs" in node and node["inputs"] is not None:
            for i,(input,input_args) in enumerate(node["inputs"].items()):
                image_row = QHBoxLayout()
                image_row.addWidget(QLabel(f"Input {i} ({input_args['type']}):"))
                self.image_selector[input] = QComboBox()
                self.update_node_menu(self.image_selector[input], input_args)
                image_row.addWidget(self.image_selector[input])
                self.layout.addLayout(image_row)

        # --- Filter parameters ---
        self.form = QFormLayout()
        self.param_widgets = {}

        for name, param in node["parameters"].items():

            widget = None

            annotation = param["type"]
            default = param["default"]

            # if param["default"] is not inspect.Parameter.empty:
            # elif (filter_name, param.name) in default_map.keys():
            #     default = default_map[(filter_name, name)]    
            #     annotation = type(default)
            # else:
            #     default = None

            # if annotation == inspect._empty:
            #     annotation = type(default)

            # print(param.name, param.annotation, default)
            if annotation == float:
                widget = QLineEdit()
                widget.setText(str(default))
                self.form.addRow(name, widget)
                # widget = QDoubleSpinBox()
                # widget.setSingleStep(0.1)
                # widget.setRange(-1e6, 1e6)
                # widget.setValue(float(default))
                # self.form.addRow(name, widget)
            elif annotation == int:
                widget = QLineEdit()
                widget.setText(str(default))
                self.form.addRow(name, widget)
                # widget = QSpinBox()
                # widget.setRange(-2147483647, 2147483647)
                # widget.setValue(int(default))
                # self.form.addRow(name, widget)
            elif annotation == bool:
                widget = QLineEdit()
                widget.setText(str(default))
                self.form.addRow(name, widget)
                # widget = QCheckBox()
                # widget.setChecked(bool(default))
                # self.form.addRow(name, widget)
            elif annotation == str:
                widget = QLineEdit()
                widget.setText(str(default))
                self.form.addRow(name, widget)
            elif annotation == type:
                widget = QLineEdit()
                widget.setText(str(default))
                self.form.addRow(name, widget)
                # widget = QComboBox()
                # widget.addItems(["uint8", "uint32", "int8", "int16", "int32", "float16", "float32", "float64"])
                # widget.setCurrentText(default)
                # self.form.addRow(name, widget)
            elif annotation == "image":
                widget = QComboBox()
                self.update_node_menu(widget, param)
                # widget.addItems(["uint8", "uint32", "int8", "int16", "int32", "float16", "float32", "float64"])
                widget.setCurrentText(default)
                self.form.addRow(name, widget)
            else:
                widget = QLineEdit()
                widget.setText(str(default))
                self.form.addRow(name, widget)
                # print(f"Unsupported parameter type for {name}: {annotation} {default}")
                # continue  # skip unsupported types

            self.param_widgets[name] = widget

            for name, widget in self.param_widgets.items():
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.valueChanged.connect(self.mark_node_changed)
                elif isinstance(widget, QCheckBox):
                    widget.stateChanged.connect(self.mark_node_changed)
                elif isinstance(widget, QLineEdit):
                    widget.textChanged.connect(self.mark_node_changed)
                elif isinstance(widget, tuple):  # range
                    for w in widget:
                        w.valueChanged.connect(self.mark_node_changed)

        self.layout.addLayout(self.form)

        # --- Output ---
        if "outputs" in node and node["outputs"] is not None:
            for i,(output,output_args) in enumerate(node["outputs"].items()):
                image_row = QHBoxLayout()
                image_row.addWidget(QLabel(f"Output {i} ({output_args['type']})"))
                self.layout.addLayout(image_row)

        # --- Apply button ---
        self.action_btn = QPushButton()
        self.action_btn.setMinimumHeight(30)
        self.layout.addWidget(self.action_btn)

        if self.node_added:
            self.setup_as_apply()
        else:
            self.setup_as_add()
        # # --- Save button ---
        # self.apply_btn = QPushButton("Save")
        # self.apply_btn.clicked.connect(self.save)
        # self.layout.addWidget(self.apply_btn)

    def setup_as_add(self):
        self.action_btn.setText("Add to Pipeline")
        try:
            self.action_btn.clicked.disconnect()
        except TypeError:
            pass
        self.action_btn.clicked.connect(self.add_and_switch)

    def setup_as_apply(self):
        self.action_btn.setText("Apply and Save")
        try:
            self.action_btn.clicked.disconnect()
        except TypeError:
            pass
        self.action_btn.clicked.connect(self.apply_filter)

    def add_and_switch(self):
        node = self.get_node()
        self.main.add_node(node)
        self.node_added = True
        self.setup_as_apply()

    def mark_node_changed(self):

        self.changed = True

    def mark_nodes_dirty(self, source_name):
        visited = set()

        def dfs(node_name):
            if node_name in visited:
                return
            visited.add(node_name)

            node = self.main.nodes[node_name]
            if node and node.meta["type"] == "function":
                node.mark_dirty()

            # Recursively update downstream nodes
            for dest_node in self.main.nodes.values():
                inputs = dest_node.meta.get("parameters", {})
                for input_data in inputs.values():
                    if input_data.get("default") == node_name:
                        dfs(dest_node.meta["name"])
                    elif input_data.get("default").split(":")[0] == node_name:
                        dfs(input_data["default"].split(":")[0])

        dfs(source_name)

    def mark_node_dirty(self):
        node_name = self.node["name"]
        if node_name in self.main.nodes:
            self.main.nodes[node_name].mark_dirty()

    def update_node_menu(self, selector, input):
        selector.clear()
        image_layers = [None]
        
        downstream_nodes = set()
        if self.main and hasattr(self.main, "get_downstream_nodes"):
            downstream_nodes = self.main.get_downstream_nodes(self.name0)

        for layer in self.main.viewer.layers:
            if layer.name not in self.main.nodes:
                image_layers.append(layer.name)

        for name, node in self.main.nodes.items():
            if name == self.name0 or name in downstream_nodes:
                continue

            if node.meta["type"] == "image" and input["type"] == "image":
                image_layers.append(name)

            if node.meta["type"] == "function":
                if input["type"] in [i["type"] for i in node.meta["outputs"].values()]:
                    image_layers.append(name)

        selector.addItems(image_layers)
        selector.setCurrentText(input["default"])
            
    def apply_filter(self):
        meta = self.get_meta()

        # Update node metadata
        updated_meta = self.get_meta()
        self.main.nodes[updated_meta["name"]].meta = updated_meta
        self.main.selected_node.meta = updated_meta

        # --- Gather inputs (images or values) ---
        input_data = {}
        for input_name, selector in self.image_selector.items():
            selected_source = selector.currentText()
            input_type = self.node["inputs"][input_name]["type"]

            # Fetch from napari layer (for image-type)
            if input_type == "image":
                if selected_source in self.main.viewer.layers:
                    input_data[input_name] = self.main.viewer.layers[selected_source].data
                else:
                    print(f"Missing image input '{input_name}' from layer: {selected_source}")
                    return

            # Fetch from upstream node (value-type)
            elif input_type == "points":
                if selected_source in self.main.viewer.layers:
                    input_data[input_name] = self.main.viewer.layers[selected_source].data
                else:
                    print(f"Missing image input '{input_name}' from layer: {selected_source}")
                    return
            
            elif input_type == "vectors":
                if selected_source in self.main.viewer.layers:
                    input_data[input_name] = self.main.viewer.layers[selected_source].data
                else:
                    print(f"Missing image input '{input_name}' from layer: {selected_source}")
                    return

            else:
                if selected_source in self.main.nodes:
                    source_node = self.main.nodes[selected_source]
                    if hasattr(source_node, "results"):
                        input_data[input_name] = source_node.results.get("0", None)
                    else:
                        raise ValueError(f"No result attribute in source node '{selected_source}' for input '{input_name}'")
                        # print(f"No result attribute in source node '{selected_source}' for input '{input_name}'")
                        # return
                else:
                    print(f"Missing input node for: {selected_source}")
                    return

        # --- Gather parameters ---
        params = {}
        for k, v in meta.get("parameters", {}).items():
            if v["type"] == str:
                params[k] = v["default"]
            elif v["type"] == "image":
                selected_source = v["default"]
                if selected_source in self.main.viewer.layers:
                    params[k] = self.main.viewer.layers[selected_source].data
            elif isinstance(v["default"], str) and v["default"].split(":")[0] in self.main.nodes:
                source_node = self.main.nodes[v["default"].split(":")[0]]
                if hasattr(source_node, "results"):
                    params[k] = source_node.results.get(v["default"].split(":")[1], "0")
                else:
                    raise ValueError(f"No result attribute in source node '{v['default']}' for parameter '{k}'")
            elif isinstance(v["default"], str):
                params[k] = eval(v["default"])
            else:
                params[k] = v["default"]

        # --- Run the function ---
        try:
            result = self.filter_func(**input_data, **params)
        except Exception as e:
            print(f"Error applying function {meta['name']}: {e}")
            return

        # --- Handle outputs ---
        outputs = meta.get("outputs", {})

        multioutput = len(outputs.values())>1
        for count, output_data in outputs.items():
            out_type = output_data["type"]
            if multioutput:
                out_name = f"{meta['name']}_{count}"
            else:
                out_name = meta["name"]

            if out_type == "image":
                if out_name in self.main.viewer.layers:
                    self.main.viewer.layers.remove(out_name)
                self.main.viewer.add_image(result, name=out_name, colormap="gray")
            elif out_type == "points":
                if out_name in self.main.viewer.layers:
                    self.main.viewer.layers.remove(out_name)
                shape = result.shape[1]
                for format in output_data["formats"]:
                    if len(format) == shape:
                        break                
                pos = result[:, [i for i in range(shape) if format[i] == "Pos"]]
                size = result[:, [i for i in range(shape) if format[i] == "Size"]].flatten()
                self.main.viewer.add_points(pos, name=out_name)
            elif out_type == "vectors":
                if out_name in self.main.viewer.layers:
                    self.main.viewer.layers.remove(out_name)
                self.main.viewer.add_vectors(result, name=out_name)
            else:
                self.main.nodes[meta["name"]].results[count] = result

        # Update graph
        self.main.update_edges()

        if self.changed:
            self.mark_nodes_dirty(meta["name"])

        if meta["name"] in self.main.nodes:
            self.main.nodes[meta["name"]].mark_clean()

        self.main.saved = False

    def rename_node(self):

        meta = self.get_meta()

        node_name = self.node["name"]
        if node_name in self.main.nodes:
            self.main.nodes[node_name].mark_clean()

        if meta["name"] != self.name0:
            for i,j in self.main.nodes.items():
                if j.meta["inputs"] is not None:
                    for name,data in  j.meta["inputs"].items():
                        if data["default"] == self.name0:
                            data["default"] = meta["name"]

        self.main.delete_selected_node()
        self.main.add_node()

    def get_meta(self):
        """
        Returns a dictionary representation of the node with updated inputs, outputs, and parameters.
        """
        # --- Parameters ---
        updated_params = self.node["parameters"].copy()
        for name, widget in self.param_widgets.items():
            # if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            #     updated_params[name]["default"] = widget.value()
            # elif isinstance(widget, QCheckBox):
            #     updated_params[name]["default"] = widget.isChecked()
            # elif isinstance(widget, QLineEdit):
            #     updated_params[name]["default"] = widget.text()
            # elif isinstance(widget, QComboBox):
            #     if widget.currentText() in ["uint8", "uint32", "int8", "int16", "int32", "float16", "float32", "float64"]:
            #         updated_params[name]["default"] = eval(widget.currentText())
            #     else:
            #         updated_params[name]["default"] = widget.currentText()
            # else:
            #     print(f"Unsupported widget type for parameter '{name}': {widget}")
            if updated_params[name]["type"] == "image":
                updated_params[name]["default"] = widget.currentText()
            else:
                updated_params[name]["default"] = widget.text()

        # --- Inputs ---
        updated_inputs = {}
        for input_name, selector in self.image_selector.items():
            selected_input = selector.currentText()
            updated_inputs[input_name] = {
                "type": self.node["inputs"][input_name]["type"],
                "default": selected_input
            }

        # # --- Outputs ---
        # output_name = self.output_name_edit.text().strip()
        # updated_outputs ={"image": {"type": "image", "default": output_name}} if output_name else {"image": {"type": "image", "default": None}}

        return {
            "name": self.name.text(),
            "function_name": self.node["function_name"],
            "type": self.node["type"],
            "inputs": updated_inputs,
            "outputs": self.node["outputs"],
            "function": self.filter_func,
            "parameters": updated_params,
            "help": self.help_link
        }

    def get_node(self):
        
        return NodeFunction(self.get_meta(), self.main)

class MenuWidget():
    def __init__(self, main_widget):

        # Use QPushButton instead of QToolButton (more reliable)
        self.menu_button = QPushButton("Nodes")
        self.menu_button.setMinimumHeight(30)
        self.menu_button.clicked.connect(self.show_menu)
        main_widget.controls.addWidget(self.menu_button)

        # Build the nested menu
        self.menu = QMenu(main_widget)

        self.menu.addAction("add image node", main_widget.add_layer_to_node)
        for menu in NODE_TYPES.keys():
            data_menu = QMenu(f"{menu}", self.menu)
            for name, node in NODE_TYPES[menu].items():
                if node["type"] == "image":
                    data_menu.addAction(name, lambda node=node: main_widget.load_data(node))
                elif node["type"] == "function":
                    data_menu.addAction(name, lambda node=node: main_widget.load(node))
            self.menu.addMenu(data_menu)

    def show_menu(self):
        # Position the menu just below the button
        pos = self.menu_button.mapToGlobal(QPoint(0, self.menu_button.height()))
        self.menu.exec_(pos)

class AutoFitGraphicsView(QGraphicsView):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # self.scene().setSceneRect(self.scene().itemsBoundingRect())
        # self.fitInView(self.sceneRect().adjusted(-20, -20, 20, 20), Qt.KeepAspectRatio)

class DataLoaderWidget(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.csv_data = None
        self.current_row = 0
        self.column_selectors = {}

        layout = QVBoxLayout(self)

        # Row layout: Load button + preview
        top_row = QHBoxLayout()

        self.csv_path_label = QLabel("No file selected")
        self.csv_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # allow copy
        top_row.addWidget(self.csv_path_label)

        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_csv)
        top_row.addWidget(self.load_btn)

        layout.addLayout(top_row)

        self.mapping_form = QFormLayout()
        layout.addLayout(self.mapping_form)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("←")
        self.prev_btn.clicked.connect(self.prev_row)
        self.next_btn = QPushButton("→")
        self.next_btn.clicked.connect(self.next_row)
        self.row_selector = QComboBox()
        self.row_selector.currentIndexChanged.connect(self.update_images)

        nav.addWidget(self.prev_btn)
        nav.addWidget(self.row_selector)
        nav.addWidget(self.next_btn)
        layout.addLayout(nav)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv);;All Files (*)")
        if not path:
            return

        try:
            self.csv_data = pd.read_csv(path)
            relative_path = os.path.relpath(path)
            self.csv_path_label.setText(relative_path)
        except Exception as e:
            QMessageBox.critical(self, "CSV Load Error", str(e))
            return

        self.setup_mapping()

    def setup_mapping(self):
        # Reset mappings
        for i in reversed(range(self.mapping_form.count())):
            item = self.mapping_form.takeAt(i)
            if item.widget():
                item.widget().deleteLater()

        self.column_selectors.clear()

        image_nodes = [n for n in self.main.nodes.values() if n.meta["type"] == "image"]
        for node in image_nodes:
            selector = QComboBox()
            selector.addItems([""] + list(self.csv_data.columns))
            self.mapping_form.addRow(f"{node.meta['name']}:", selector)
            self.column_selectors[node.meta["name"]] = selector

        self.row_selector.clear()
        first_column = self.csv_data.columns[0]
        self.row_selector.addItems([str(val) for val in self.csv_data[first_column]])
        self.row_selector.setCurrentIndex(0)
        self.current_row = 0
        self.update_images()

    def prev_row(self):
        if self.current_row > 0:
            self.current_row -= 1
            self.row_selector.setCurrentIndex(self.current_row)  # This will trigger update_images()

    def next_row(self):
        if self.current_row < len(self.csv_data) - 1:
            self.current_row += 1
            self.row_selector.setCurrentIndex(self.current_row)  # This will trigger update_images()

    def update_images(self):
        if self.csv_data is None:
            return

        self.current_row = self.row_selector.currentIndex()
        row = self.csv_data.iloc[self.current_row]

        for node_name, combo in self.column_selectors.items():
            column = combo.currentText()
            if not column:
                continue

            path = row[column]
            if isinstance(path, str) and path.strip():
                try:
                    img = imread(path)
                    if node_name in self.main.viewer.layers:
                        self.main.viewer.layers[node_name].data = img
                    else:
                        self.main.viewer.add_image(img, name=node_name, colormap="gray")
                    if node_name in self.main.nodes:
                        node = self.main.nodes[node_name]
                        node.meta["shape"] = img.shape
                        node.meta["dtype"] = str(img.dtype)
                    if self.main.selected_node and self.main.selected_node.meta["name"] == node_name:
                        self.main.load(node.meta)
                except Exception as e:
                    print(f"Could not load image from {path}: {e}")

    def sync_node_mapping(self):
        if self.csv_data is None:
            return  # nothing to do yet

        current_nodes = {n.meta["name"]: n for n in self.main.nodes.values() if n.meta["type"] == "image"}

        # Clear old widgets
        for i in reversed(range(self.mapping_form.count())):
            item = self.mapping_form.takeAt(i)
            if item.widget():
                item.widget().deleteLater()

        self.column_selectors.clear()

        for name, node in current_nodes.items():
            selector = QComboBox()
            selector.addItems([""] + list(self.csv_data.columns))
            self.mapping_form.addRow(f"{name}:", selector)
            self.column_selectors[name] = selector

        # Refresh image preview for current row
        self.update_images()

class AnalysisWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.edge_pairs = set()
        self.selected_node = None
        self.nodes = {}
        self.saved = True

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Buttons and input
        controls = QHBoxLayout()
        self.controls = controls
        self.node_input = MenuWidget(self)

        # --- Menu Button ---
        self.menu_button = QPushButton("Menu")
        self.menu_button.setMinimumHeight(30)
        self.menu = QMenu(self.menu_button)
        # Add actions
        save_action = self.menu.addAction("New Pipeline")
        save_action.triggered.connect(self.new_pipeline)
        # Add actions
        save_action = self.menu.addAction("Save Pipeline")
        save_action.triggered.connect(self.save_pipeline)
        # Load action
        load_action = self.menu.addAction("Load Pipeline")
        load_action.triggered.connect(self.load_pipeline)
        # Load action
        load_action = self.menu.addAction("Dataloader")
        load_action.triggered.connect(self.toggle_data_loader)
        # Connect menu to button
        self.menu_button.setMenu(self.menu)
        # Add to layout
        controls.addWidget(self.menu_button)

        add_edge_btn = QPushButton("Remove from pipeline")
        add_edge_btn.clicked.connect(self.delete_selected_node)
        controls.addWidget(add_edge_btn)

        self.layout.addLayout(controls)

        # Buttons and input
        self.function_box = QWidget()
        self.function_box.setObjectName("FunctionBox")
        self.functions = QVBoxLayout()
        self.functions.setContentsMargins(10, 10, 10, 10)
        self.function_box.setLayout(self.functions)
        self.function_box.setStyleSheet("""
            QWidget#FunctionBox {
                border: 2px solid #cccccc;
                border-radius: 8px;
            }
        """)
        self.layout.addWidget(self.function_box)
        self.set_function_widget(None)

        # Data loader
        self.data_loader_widget = DataLoaderWidget(self)
        self.data_loader_widget.setVisible(False)
        self.layout.addWidget(self.data_loader_widget)

        # Graphical scene and view
        controls_scene = QHBoxLayout()
        self.fit_scene_btn = QPushButton("Fit to Scene")
        self.fit_scene_btn.clicked.connect(self.fit_scene)
        controls_scene.addWidget(self.fit_scene_btn)
        self.layout.addLayout(controls_scene)

        self.scene = QGraphicsScene()
        self.view = AutoFitGraphicsView(self.scene)
        self.layout.addWidget(self.view)

    def set_function_widget(self, widget):
        # Clear all previous widgets
        while self.functions.count():
            old = self.functions.takeAt(0)
            if old.widget():
                old.widget().deleteLater()

        if widget is not None:
            self.functions.addWidget(widget)
            self.function_box.show()
        else:
            self.function_box.hide()

    def on_node_click(self, label):
        node = self.nodes[label]
        meta = node.meta

        if meta["type"] == "image":
            # Show basic image info instead of FunctionWidget
            layer = self.viewer.layers[label] if label in self.viewer.layers else None
            if layer:
                info_widget = self.make_image_info_widget(layer)
                self.set_function_widget(info_widget)
        else:
            self.set_function_widget(WidgetFunction(self, meta))

    def add_node(self, node):

        if node.meta["name"] in self.nodes:
            raise ValueError(f"Node with name '{node.meta['name']}' already exists.")
        else:
            self.nodes[node.meta["name"]] = node
            self.scene.addItem(node)
            if self.selected_node is not None:
                self.selected_node.setPen(QPen(Qt.black))
            self.selected_node = node
            if self.selected_node is not None:
                self.selected_node.setPen(QPen(Qt.red))

        self.saved = False
        self.data_loader_widget.sync_node_mapping()

    # def add_function(self):
        
    #     node = self.functions.itemAt(0).widget().get_node()  # Clear previous widget
    #     # node = self.functions.at.get_node()
    #     self.add_node(node)

    def add_node_metadata(self, node):

        widget_item = self.functions.itemAt(0)
        if widget_item is not None:
            widget = widget_item.widget()
            if widget is not None:
                node_meta = widget.get_node()
                if node_meta["name"] not in self.nodes:
                    y = max([i[1].y() for i in self.nodes.items()], default=0)
                    node = Node(0, y+100, node_meta["name"], self.on_node_click, self, node_meta)
                    self.nodes[node_meta["name"]] = node
                    self.scene.addItem(node)
                    self.selected_node = node
                else:
                    raise ValueError(f"Node with name '{node_meta['name']}' already exists.")
                
        self.update_edges()

    def add_node_from_input(self):
        label = self.node_input.text().strip()
        if label:
            self.add_node(label)
            self.node_input.clear()

    def delete_selected_node(self):
        node = self.selected_node
        if node:

            if node.meta["name"] in self.viewer.layers:
                self.viewer.layers.remove(node.meta["name"])

            # Remove all connected edges
            for edge in list(node.edges):
                self.scene.removeItem(edge)
                if edge.source:
                    edge.source.edges = [e for e in edge.source.edges if e != edge]
                if edge.dest:
                    edge.dest.edges = [e for e in edge.dest.edges if e != edge]
            self.scene.removeItem(node)
            self.nodes = {k: v for k, v in self.nodes.items() if v != node}
            self.selected_node = None

        self.update_edges()

        self.saved = False
        self.data_loader_widget.sync_node_mapping()

    def load(self, meta):
        """
        Load a node based on its metadata.
        If the node is a dataset, display its info.
        If it's a function, display the function widget.
        """
        # if self.selected_node:
        #     self.selected_node.setPen(QPen(Qt.black))

        if meta["type"] == "image":
            # Load dataset and show its info
            if meta["name"] in self.viewer.layers:
                node = NodeData(meta["name"], self.viewer.layers[meta["name"]].data, self)
            else:
                node = NodeData(meta["name"], [], self)
            self.set_function_widget(node.display_widget())
        elif meta["type"] == "function":
            # Load function and show its widget
            node = NodeFunction(meta, self)
            self.set_function_widget(WidgetFunction(self, meta))

    def load_data(self, meta):
        # 1. Load image into Napari
        image = meta["function"]()
        self.viewer.add_image(image, name=meta["name"], colormap="gray")

        # node = NodeData(meta["name"], image, self)
        # self.add_node(node)

        # self.load(node.meta)

        return
    
    def add_layer_to_node(self):
        """
        Add a layer to the current node.
        If no node is selected, create a new one.
        """
        layer = self.viewer.layers.selection.active
        if layer is None:
            name = "image"
            count = 1
            while name in self.nodes:   
                name = f"image_{count}"
                count += 1
            
            data = []
        else:
            name = layer.name
            data = layer.data

        # Create a new NodeData with the selected layer's data
        node = NodeData(name, data, self)
        self.add_node(node)

        self.load(node.meta)

    def update_edges(self):
        # --- Remove all edges from the scene and from nodes ---
        for node in self.nodes.values():
            for edge in list(node.edges):  # copy since we’re modifying in-loop
                self.scene.removeItem(edge)
            node.edges.clear()  # reset node's edge list

        # --- Rebuild edges from inputs ---
        for dest_node in self.nodes.values():
            # if "inputs" not in dest_node.meta:
            #     continue
            for input_name, input_info in dest_node.meta.get("parameters", {}).items():
                if input_info.get("type") == "image":
                    source_name = input_info.get("default")
                    if source_name and source_name in self.nodes:
                        source_node = self.nodes[source_name]
                        edge = Edge(source_node, dest_node)
                        self.scene.addItem(edge)
                
                if input_info.get("default").split(":")[0] in self.nodes:
                    source_node = self.nodes[input_info.get("default").split(":")[0]]
                    edge = Edge(source_node, dest_node)
                    self.scene.addItem(edge)

    def fit_scene(self):
        items_rect = self.scene.itemsBoundingRect()
        self.scene.setSceneRect(items_rect)
        self.view.fitInView(items_rect.adjusted(-20, -20, 20, 20), Qt.KeepAspectRatio)

    def get_downstream_nodes(self, start_node_name):
        visited = set()

        def dfs(node_name):
            if node_name in visited:
                return
            visited.add(node_name)
            node = self.nodes.get(node_name)
            if not node:
                return
            for dest_node in self.nodes.values():
                inputs = dest_node.meta.get("inputs", {})
                for input_data in inputs.values():
                    if input_data.get("default") == node_name:
                        dfs(dest_node.meta["name"])

        dfs(start_node_name)
        visited.discard(start_node_name)
        return visited

    def new_pipeline(self):
        """
        Clear the current pipeline and reset the scene.
        """
        if not self.saved:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save before clearing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel
            )
            if reply == QMessageBox.Yes:
                self.save_pipeline()
            elif reply == QMessageBox.Cancel:
                return

        self.clear_pipeline()

        self.saved = True

    def save_pipeline(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Pipeline", "", "JSON Files (*.json);;All Files (*)")
        if not path:
            return  # User cancelled

        import json

        pipeline_data = {}
        for name, node in self.nodes.items():
            meta_new = {}
            meta = node.meta.copy()
            meta_new["name"] = meta["name"]
            meta_new["type"] = meta["type"]
            if "function_name" in meta:
                meta_new["function_name"] = meta["function_name"]
            if "parameters" in meta:
                meta_new["parameters"] = {}
                for k, v in meta["parameters"].items():
                    meta_new["parameters"][k] = {
                            "default": v["default"],
                        }
                
            pipeline_data[name] = meta_new

        try:
            with open(path, "w") as f:
                json.dump(pipeline_data, f, indent=2)
            print(f"Pipeline saved to: {path}")
        except Exception as e:
            print(f"Failed to save pipeline: {e}")

        self.saved = True

    def load_pipeline(self):

        self.new_pipeline()

        path, _ = QFileDialog.getOpenFileName(self, "Load Pipeline", "", "JSON Files (*.json);;All Files (*)")
        if not path:
            return  # User cancelled

        try:
            with open(path, "r") as f:
                pipeline_data = json.load(f)
        except Exception as e:
            raise TypeError(f"Failed to load pipeline: {e}")

        for name, meta in pipeline_data.items():
            node_type = meta.get("type")
            pos = meta.get("pos", (0, 0))

            if node_type == "image":
                # Reload image from viewer or dummy placeholder
                if name in self.viewer.layers:
                    image = self.viewer.layers[name].data
                else:
                    image = np.zeros(meta.get("shape", (100, 100)), dtype=eval(meta.get("dtype", "float32")))
                    self.viewer.add_image(image, name=name, colormap="gray")

                node = NodeData(name, image, self, pos=pos)

            elif node_type == "function":
                # You'll need to rehydrate the function object
                func_name = meta.get("function_name")
                func_obj = None
                for d in NODE_TYPES.values():
                    for sub in d.values():
                        if func_name in sub:
                            func_obj = sub[func_name]["function"]

                if func_obj is None:
                    print(f"Could not find function {func_name} in NODE_TYPES")
                    continue

                meta["function"] = func_obj
                node = NodeFunction(meta, self, pos=pos)

            else:
                print(f"Unknown node type: {node_type}")
                continue

            self.add_node(node)

        self.update_edges()
        self.fit_scene()

        self.saved = True

    def clear_pipeline(self):
        for node in list(self.nodes.values()):
            self.scene.removeItem(node)
        self.nodes.clear()
        self.selected_node = None
        self.scene.clear()
        self.viewer.layers.clear()

    def toggle_data_loader(self):
        self.data_loader_widget.setVisible(not self.data_loader_widget.isVisible())
