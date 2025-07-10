import os
from pathlib import Path
import numpy as np
import nd2
import tifffile
import napari
import sys
import zarr
import pandas as pd
from itertools import chain, permutations, product
from scipy.ndimage import map_coordinates
from qtpy.QtWidgets import (
    QWidget, QPushButton, QComboBox,
    QMessageBox, QFileDialog, QLabel,
    QLineEdit, QHBoxLayout, QVBoxLayout, QDialog, QDoubleSpinBox, QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem, QInputDialog, QCheckBox, QHeaderView, QSizePolicy
)
from magicgui.widgets import Container, FloatRangeSlider, FloatSlider
from qtpy.QtCore import Qt, QTimer
from PIL import Image

def get_permutations(s):
    return [''.join(p) for p in permutations(s)]

AXIS = np.sort(get_permutations("XY")+get_permutations("XYZ")+get_permutations("XYT")+get_permutations("XYC")+get_permutations("XYZC")+get_permutations("XYZT")+get_permutations("XYCT")+get_permutations("XYZTC"))

class PreprocessingWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        axis = AXIS[[len(i) == 4 for i in AXIS]]

        # Parameters
        self.viewer = viewer
        self.dropdown_file_menu_files = []
        self.name_files = {}
        self.axis = axis[0]
        self.scale = (1.,1.,1.,1.)
        self.cut_sliders = {}
        self.cuts = []
        self.cut_count = 0
        self.load_in_advance = 0
        self.metadata_files = None
        self.metadata_file = None
        self.metadata_cuts = None
        self.metadata_cut = None
        self.making_cut = False
        self.selected_file = None
        self.selected_file_pos = 0
        self.selected_file_name = None
        self.selected_cut = None
        self.selected_cut_preview = None
        self._block = False
        self.create()

        # Layout
        self.layout = QVBoxLayout()

        # File navigation buttons and dropdown
        self.button_file_previous = QPushButton("Previous file")
        self.button_file_previous.clicked.connect(self.file_previous)
        self.button_file_next = QPushButton("Next file")
        self.button_file_next.clicked.connect(self.file_next)
        self.dropdown_file_menu = QComboBox()
        self.dropdown_file_menu.addItems(self.dropdown_file_menu_files)
        self.dropdown_file_menu.currentTextChanged.connect(self.file_select)
        self.dropdown_file_menu.setCurrentText("")
        self.dropdown_file_menu.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.dropdown_file_menu.setMinimumContentsLength(1)
        # Layout
        self.file_nav_layout = QHBoxLayout()
        self.file_nav_layout.addWidget(self.button_file_previous)
        self.file_nav_layout.addWidget(self.button_file_next)
        self.file_nav_layout.addWidget(self.dropdown_file_menu)
        self.layout.addLayout(self.file_nav_layout)

        # Axis and Scale Settings
        self.axis_dropdown = QComboBox()
        self.axis_dropdown.addItems(axis)
        self.axis_dropdown.setCurrentText(self.axis)
        self.axis_dropdown.currentTextChanged.connect(self.axis_dropdown_edit)
        self.scale_line_edit = QLineEdit(f"{self.scale}")
        self.scale_line_edit_value = self.scale_line_edit.text()
        self.scale_line_edit.editingFinished.connect(self.scale_line_edit_check)
        # Create two-column table
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(2)
        self.file_table.setHorizontalHeaderLabels(["Label", "Value"])
        self.file_table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.SelectedClicked)
        self.file_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.file_table.setSelectionMode(QTableWidget.SingleSelection)
        self.file_table.setMinimumHeight(20)
        self.file_table.horizontalHeader().setStretchLastSection(True)
        self.file_table.cellChanged.connect(self.save_metadata_file)
        self.file_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.file_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Add row and save buttons
        self.add_file_row_button = QPushButton("+ Add Row")
        self.add_file_row_button.clicked.connect(self.new_file_table_row)
        self.delete_file_row_button = QPushButton("- Delete Row")
        self.delete_file_row_button.clicked.connect(self.remove_file_table_row)
        # self.add_file_metadata_save_button = QPushButton("Save")
        # self.add_file_metadata_save_button.clicked.connect(self.save_metadata_file)
        # Layout
        axis_scale_group = QGroupBox("Axis-Scale")
        axis_scale_layout = QVBoxLayout()
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("Axis:"))
        axis_layout.addWidget(self.axis_dropdown)
        axis_widget = QWidget()
        axis_widget.setLayout(axis_layout)
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Axis Scale:"))
        scale_layout.addWidget(self.scale_line_edit)
        scale_widget = QWidget()
        scale_widget.setLayout(scale_layout)
        axis_scale_layout.addWidget(axis_widget)
        axis_scale_layout.addWidget(scale_widget)
        self.widget_file_metadata = QWidget()
        self.layout_file_medata = QVBoxLayout()
        self.layout_file_medata.addWidget(self.file_table)
        self.layout_file_medata.addWidget(self.add_file_row_button)
        self.layout_file_medata.addWidget(self.delete_file_row_button)
        self.widget_file_metadata.setLayout(self.layout_file_medata)
        # self.layout_file_medata.addWidget(self.add_file_metadata_save_button)
        axis_scale_layout.addWidget(self.widget_file_metadata)
        axis_scale_group.setLayout(axis_scale_layout)
        self.layout.addWidget(axis_scale_group)

        # Cut Start/End Buttons and Settings
        self.cut_start = QPushButton("Start Cut")
        self.cut_start.clicked.connect(self.start_cut)
        self.cut_end = QPushButton("End Cut")
        self.cut_end.clicked.connect(self.end_cut)
        self.cut_load = QPushButton("Load Cut")
        self.cut_load.clicked.connect(self.load_cut)
        self.save_format = QComboBox()
        self.save_format.addItems(["tif"])
        # Layout
        self.cut_start_end_group = QGroupBox("Cut Start/End")
        self.cut_start_end_layout = QVBoxLayout()
        self.cut_layout = QHBoxLayout()
        self.cut_layout.addWidget(self.cut_start)
        self.cut_layout.addWidget(self.cut_end)
        self.cut_layout.addWidget(self.cut_load)
        self.cut_widget = QWidget()
        self.cut_widget.setLayout(self.cut_layout)
        self.end_layout = QHBoxLayout()
        self.end_layout.addWidget(QLabel("Save Format:"))
        self.end_layout.addWidget(self.save_format)
        self.end_widget = QWidget()
        self.end_widget.setLayout(self.end_layout)
        self.cut_start_end_layout.addWidget(self.cut_widget)
        self.cut_start_end_layout.addWidget(self.end_widget)
        self.cut_start_end_group.setLayout(self.cut_start_end_layout)
        self.layout.addWidget(self.cut_start_end_group)

        # Cut Preview Layer
        # Create two-column table
        self.cut_table = QTableWidget()
        self.cut_table.setColumnCount(2)
        self.cut_table.setHorizontalHeaderLabels(["Label", "Value"])
        self.cut_table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.SelectedClicked)
        self.cut_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cut_table.setSelectionMode(QTableWidget.SingleSelection)
        self.cut_table.setMinimumHeight(20)
        self.cut_table.horizontalHeader().setStretchLastSection(True)
        self.cut_table.cellChanged.connect(self.save_metadata_cut)
        self.cut_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cut_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Add row and save buttons
        self.add_row_button = QPushButton("+ Add Row")
        self.add_row_button.clicked.connect(self.new_table_row)
        self.delete_row_button = QPushButton("- Delete Row")
        self.delete_row_button.clicked.connect(self.remove_table_row)
        # self.add_metadata_save_button = QPushButton("Save")
        # self.add_metadata_save_button.clicked.connect(self.save_metadata_cut)
        # Layout
        self.group_cut = QGroupBox("Cut Metadata")
        self.layout_cut = QVBoxLayout()
        self.widget_cut_sliders = QWidget()
        self.layout_cut_sliders = QVBoxLayout()
        self.widget_cut_sliders.setLayout(self.layout_cut_sliders)
        self.layout_cut.addWidget(self.widget_cut_sliders)
        self.widget_cut_metadata = QWidget()
        self.layout_cut_medata = QVBoxLayout()
        self.widget_cut_metadata.setLayout(self.layout_cut_medata)
        self.layout_cut_medata.addWidget(self.cut_table)
        self.layout_cut_medata.addWidget(self.add_row_button)
        self.layout_cut_medata.addWidget(self.delete_row_button)
        # self.layout_cut_medata.addWidget(self.add_metadata_save_button)
        self.layout_cut.addWidget(self.widget_cut_metadata)
        self.group_cut.setLayout(self.layout_cut)
        self.layout.addWidget(self.group_cut)

        # Set App layout
        self.setLayout(self.layout)

        self.widget_file_metadata.setVisible(False)
        self.widget_cut_sliders.setVisible(False)
        self.widget_cut_metadata.setVisible(False)

    def create(self):

        # Define dialog inline
        dialog = QDialog(self)
        dialog.setWindowTitle("Load or Create Project")

        raw_edit = QLineEdit()
        raw_edit.setReadOnly(True)
        raw_btn = QPushButton("Browse")
        def select_raw():
            path = QFileDialog.getExistingDirectory(dialog, "Select ND2 Raw Data Folder")
            if path:
                raw_edit.setText(path)
        raw_btn.clicked.connect(select_raw)

        raw_layout = QHBoxLayout()
        raw_layout.addWidget(raw_edit)
        raw_layout.addWidget(raw_btn)

        ending_edit = QLineEdit()
        ending_edit.setText("*.tif,*.tiff")
        ending_layout = QHBoxLayout()
        ending_layout.addWidget(ending_edit)

        project_edit = QLineEdit()
        project_edit.setReadOnly(True)
        project_btn = QPushButton("Browse")
        def select_project():
            path = QFileDialog.getExistingDirectory(dialog, "Select Project Output Folder")
            if path:
                project_edit.setText(path)
        project_btn.clicked.connect(select_project)

        project_layout = QHBoxLayout()
        project_layout.addWidget(project_edit)
        project_layout.addWidget(project_btn)

        # Buttons
        load_btn = QPushButton("Load")
        cancel_btn = QPushButton("Cancel")

        def load_project():
            if not raw_edit.text() or not project_edit.text():
                QMessageBox.warning(dialog, "Missing Input", "Please select both folders.")
                return
            dialog.accept()

        def cancel_project():
            QMessageBox.information(dialog, "Cancelled", "Project setup was cancelled.")
            dialog.reject()

        load_btn.clicked.connect(load_project)
        cancel_btn.clicked.connect(cancel_project)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(load_btn)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select Raw Data Folder:"))
        layout.addLayout(raw_layout)
        layout.addWidget(QLabel("File Ending:"))
        layout.addLayout(ending_layout)
        layout.addWidget(QLabel("Select Project Folder:"))
        layout.addLayout(project_layout)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)

        # Show dialog and handle result
        if dialog.exec_() != QDialog.Accepted:
            self.setDisabled(True)
            return

        # Assign selected paths
        self.root_dir = raw_edit.text()
        self.root_dir_cuts = project_edit.text()

        # self.root_dir = "/Users/gatocor/Documents/Academic/EMBL Postdoc/gastrulation_actin_cable/PIV_actin_cable/input_toby"#REMOVE
        # self.root_dir_cuts = "/Users/gatocor/Documents/Academic/EMBL Postdoc/gastrulation_actin_cable/PIV_actin_cable/input_toby_processed"#REMOVE
        self.root_path = Path(self.root_dir)
        self.root_path_cuts = Path(self.root_dir_cuts)
        self.root_path_cuts.mkdir(exist_ok=True)

        if not (self.root_path_cuts / "dataset_files.csv").exists():
            self.metadata_files = pd.DataFrame({
                "raw_file": [],
                "file": [],
                "axis": [],
                "scale": [],
                "shape": [],
            })
            self.metadata_files.to_csv(self.root_path_cuts / "dataset_files.csv", index=False)
        else:
            self.metadata_files = pd.read_csv(self.root_path_cuts / "dataset_files.csv")

        if not (self.root_path_cuts / "dataset_cuts.csv").exists():
            self.metadata_cuts = pd.DataFrame({
                "raw_file": [],
                "file": [],
                "cut": [],
                "axis": [],
                "scale": [],
                "shape": [],
                "rawX": [],
                "rawY": [],
                "rawZ": [],
                "rawT": [],
                "rawC": [],
            })
            self.metadata_cuts.to_csv(self.root_path_cuts / "dataset_cuts.csv", index=False)
        else:
            self.metadata_cuts = pd.read_csv(self.root_path_cuts / "dataset_cuts.csv")

        (self.root_path_cuts / "data").mkdir(exist_ok=True)
        (self.root_path_cuts / "screenshot").mkdir(exist_ok=True)

        # self.files_list = list(chain.from_iterable(
        #     self.root_path.rglob(i.strip()) for i in ending_edit.text().split(",")
        # ))
        ending_edit = "*.tif,*.tiff" #REMOVE
        self.files_list = list(chain.from_iterable(
            self.root_path.rglob(i.strip()) for i in ending_edit.split(",")
        ))
        self.dropdown_file_menu_files = [str(file_name) for file_name in self.files_list]
        stem = [file_name.stem for file_name in self.files_list]
        if len(self.dropdown_file_menu_files) != len(np.unique(stem)):
            self.name_files = {str(i):Path(i.split(self.root_dir)[-1][1:].replace("/", "_").replace("\ ", "_")).stem for i in self.dropdown_file_menu_files}
        else:
            self.name_files = {str(file_name):file_name.stem for file_name in self.files_list}
        self.files_list = [""] + self.files_list
        self.dropdown_file_menu_files = [""] + self.dropdown_file_menu_files
        self.name_files[""] = ""

    def scale_layers(self):
        for layer in self.viewer.layers:
            layer.scale = self.scale

    def reset_axis_dropdown(self):
        l = len(self.viewer.layers[self.selected_file_name].data.shape)
        if l == len(self.axis_dropdown.currentText()):
            return

        axis = AXIS[[len(i) == l for i in AXIS]]
        self.axis_dropdown.clear()
        self.axis_dropdown.addItems(axis)
        self.axis = axis[0]
        self.axis_dropdown.setCurrentText(self.axis)
        self.scale = tuple([1.0]*l)
        self.scale_line_edit.setText(f"{self.scale}")  # Reset scale input

    def load_cuts(self):

        metadata_cuts = self.metadata_cuts[self.metadata_cuts["raw_file"] == self.selected_file]

        data = []
        self.cuts = []
        for _, row in metadata_cuts.iterrows():
            axis = row["axis"]
            rectangle = np.zeros((4, len(axis)), dtype=float)
            rectangle[:,-1] = eval(row[f"raw{axis[-1]}"])
            rectangle[:,-2] = eval(row[f"raw{axis[-2]}"])
            ranges = {}
            for id, ax in enumerate(row["axis"][:-2]):
                val = eval(row[f"raw{ax}"])
                ranges[id] = range(int(val[0]), int(val[1] + 1))            
            rectangles = self.expand_shape_over_dims(rectangle, ranges)
            data += rectangles
            self.cuts += [row["cut"]]*len(rectangles)

        # Update internal state
        self.cut_count = metadata_cuts["cut"].max() if not metadata_cuts.empty else -1
        self.axis = self.metadata_file["axis"]
        self.scale = eval(self.metadata_file["scale"]) if isinstance(self.metadata_file["scale"], str) else self.metadata_file["scale"]

        # Update interactive elements
        self.axis_dropdown.setCurrentText(self.axis)
        self.scale_line_edit.setText(f"{self.scale}")

        # Update viewer layers
        self.viewer.layers["Cuts"].data = data
        self.viewer.layers["Cuts"].mode = "select"
        
        self.reset_axis_dropdown()
        self.scale_layers()

    def go2file(self, pos):

        # Prevent modifictions during cutsorutsid  file list
        if self.making_cut:
            self.dropdown_file_menu.setCurrentText(self.selected_file)
            QMessageBox.warning(self, "Cut in progress", "Please finish the current cut before navigating to another file.")
            return
        elif pos < 0 or pos >= len(self.files_list):
            return

        # Load the selected file
        file_name = self.files_list[pos]
        if file_name == "":
            self.selected_file_pos = pos
            self.selected_file = self.dropdown_file_menu_files[self.selected_file_pos]
            self.selected_file_name = self.name_files[self.selected_file]
            self.viewer.layers.clear()
            self.widget_file_metadata.setVisible(False)
            self.widget_cut_sliders.setVisible(False)
            self.widget_cut_metadata.setVisible(False)
            return

        try:
            if file_name.suffix.lower() == '.nd2':
                with nd2.ND2File(file_name) as nd2_file:
                    image_stack = nd2_file.asarray()
            elif file_name.suffix.lower() == '.zarr':
                image_stack = zarr.open(file_name, mode='r')
            else:
                image_stack = tifffile.imread(file_name)
        except Exception:
            QMessageBox.warning(self, "File Load Error", f"Could not load file: {file_name}")
            return

        # Update parameters
        self.selected_file_pos = pos
        self.selected_file = self.dropdown_file_menu_files[self.selected_file_pos]
        self.selected_file_name = self.name_files[self.selected_file]
        if self.selected_file in self.metadata_files["raw_file"].values:
            self.metadata_file = self.metadata_files[self.metadata_files["raw_file"] == self.selected_file].iloc[0]
        else:
            self.metadata_file = pd.Series({
                "raw_file": self.selected_file,
                "file": self.selected_file_name,
                "axis": self.axis,
                "scale": self.scale,
                "shape": image_stack.shape
            })
            self.metadata_files = pd.concat([self.metadata_files, self.metadata_file.to_frame().T], ignore_index=True)
            self.save_metadata()
            self.metadata_files = pd.read_csv(self.root_path_cuts / "dataset_files.csv")
            self.metadata_file = self.metadata_files[self.metadata_files["raw_file"] == self.selected_file].iloc[0]
        self.metadata_cut = None
        self.cut_count = -1
        self.cuts = []

        # Hide cut
        self.widget_file_metadata.setVisible(True)
        self.widget_cut_sliders.setVisible(False)
        self.widget_cut_metadata.setVisible(False)

        # Update viewer layers
        self.viewer.layers.clear()
        self.viewer.add_image(image_stack, name=self.selected_file_name)
        cut_layer = self.viewer.add_shapes([], ndim=image_stack.ndim, name="Cuts", edge_color="blue", face_color="transparent", edge_width=5, opacity=1)            
        self.viewer.dims.current_step = (0, 0, 0)
        cut_layer.remove_selected = self.confirm_and_remove_selected.__get__(cut_layer)

        # Load auxiliar metadata
        self.update_file_table_from_metadata()
        self.load_cuts()

    def file_previous(self):
        self.go2file(self.selected_file_pos-1)
        self._block = True
        self.dropdown_file_menu.setCurrentText(self.selected_file)
        self._block = False

    def file_next(self):
        self.go2file(self.selected_file_pos+1)
        self._block = True
        self.dropdown_file_menu.setCurrentText(self.selected_file)
        self._block = False

    def file_select(self, event):
        if self._block:
            return
        pos = np.where(np.array(self.dropdown_file_menu_files) == event)[0][0]
        self.go2file(pos)

    def clear_metadata_cut_table(self):
        self._block = True
        while self.cut_table.rowCount() > 0:
            self.cut_table.removeRow(0)
        self._block = False

    def clear_metadata_file_table(self):
        self._block = True
        while self.file_table.rowCount() > 0:
            self.file_table.removeRow(0)
        self._block = False

    def name_cut(self):

        if self.selected_cut is None:
            return            
            
        return f"{self.selected_file_name}_cut_{self.selected_cut:04d}"

    def add_table_row(self, name, content, content_editable=True):

        self._block = True
        row = self.cut_table.rowCount()
        self.cut_table.insertRow(row)

        # First column: non-editable label
        item_label = QTableWidgetItem(name)
        item_label.setFlags(item_label.flags() & Qt.ItemIsEditable)  # Make it non-editable
        self.cut_table.setItem(row, 0, item_label)

        # Second column: editable cell, optional default value
        item_value = QTableWidgetItem(content)
        if content_editable:
            item_value.setFlags(item_value.flags() | Qt.ItemIsEditable)
        else:
            item_value.setFlags(item_value.flags() & Qt.ItemIsEditable)
        self.cut_table.setItem(row, 1, item_value)
        self._block = False

    def add_file_table_row(self, name, content, content_editable=True):

        self._block = True
        row = self.file_table.rowCount()
        self.file_table.insertRow(row)

        # First column: non-editable label
        item_label = QTableWidgetItem(name)
        item_label.setFlags(item_label.flags() & Qt.ItemIsEditable)  # Make it non-editable
        self.file_table.setItem(row, 0, item_label)

        # Second column: editable cell, optional default value
        item_value = QTableWidgetItem(content)
        if content_editable:
            item_value.setFlags(item_value.flags() | Qt.ItemIsEditable)
        else:
            item_value.setFlags(item_value.flags() & Qt.ItemIsEditable)
        self.file_table.setItem(row, 1, item_value)
        self._block = False

    def new_table_row(self):
        # Prompt user for row label (first column)
        label, ok = QInputDialog.getText(self, "Add Row", "Enter label for new row:")
        if not ok or not label.strip():
            return  # Cancelled or empty

        if label.strip() in self.metadata_cuts.columns:
            QMessageBox.warning(self, "Duplicate Label", f"The label '{label.strip()}' already exists.")
            return

        self.metadata_cuts[label.strip()] = ""  # Add new column to metadata
        self.metadata_cut[label.strip()] = ""  # Add new column to cut metadata
        self.add_table_row(label.strip(), "")

    def new_file_table_row(self):
        # Prompt user for row label (first column)
        label, ok = QInputDialog.getText(self, "Add Row", "Enter label for new row:")
        if not ok or not label.strip():
            return  # Cancelled or empty

        if label.strip() in self.metadata_files.columns:
            QMessageBox.warning(self, "Duplicate Label", f"The label '{label.strip()}' already exists.")
            return

        if label.strip() in self.metadata_cuts.columns:
            QMessageBox.warning(self, "Cut Label", f"The label '{label.strip()}' already exists in cuts. If you consider this label to be of all the file, please remove it before from cuts and then add it here.")
            return

        self.metadata_file[label.strip()] = ""  # Add new column to metadata
        self.metadata_files[label.strip()] = ""  # Add new column to cut metadata
        if self.metadata_cut is not None:
            self.metadata_cut[label.strip()] = ""
            self.update_cut_table_from_metadata()
        self.metadata_cuts[label.strip()] = ""  # Add new column to cuts metadata
        self.add_file_table_row(label.strip(), "")

    def remove_table_row(self):
        selected = self.cut_table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "No Selection", "Please select a row to remove.")
            return

        label = self.cut_table.item(selected, 0).text()
        if label in ["raw_file", "file", "axis", "scale", "shape", "cut", "rawX", "rawY", "rawZ", "rawT", "rawC"] or label in self.metadata_files.columns:
            QMessageBox.warning(self, "Protected Row", "This row cannot be deleted.")
            return
        else:
            reply = QMessageBox.question(
                self,
                "Confirm Row Deletion",
                f"Are you sure you want to delete the row '{label}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            
        self.metadata_cuts.drop(columns=[label], inplace=True, errors='ignore')
        if self.metadata_cut is not None:
            self.metadata_cut.drop(labels=[label], inplace=True, errors='ignore')

        self.cut_table.removeRow(selected)
        self.save_metadata()

    def remove_file_table_row(self):
        selected = self.file_table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "No Selection", "Please select a row to remove.")
            return

        label = self.file_table.item(selected, 0).text()
        if label in ["raw_file", "file", "axis", "scale", "shape"]:
            QMessageBox.warning(self, "Protected Row", "This row cannot be deleted.")
            return
        else:
            reply = QMessageBox.question(
                self,
                "Confirm Row Deletion",
                f"Are you sure you want to delete the row '{label}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            
        self.metadata_files.drop(columns=[label], inplace=True, errors='ignore')
        if self.metadata_file is not None:
            self.metadata_file.drop(labels=[label], inplace=True, errors='ignore')

        self.file_table.removeRow(selected)
        self.save_metadata()

    def update_cut_sliders(self):

        current_layer = self.viewer.layers[self.selected_file_name]
        shape = current_layer.data.shape

        axis_string = self.axis_dropdown.currentText()

        # Clear existing sliders
        while self.layout_cut_sliders.count():
            item = self.layout_cut_sliders.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.cut_sliders = {}
        self.axis = axis_string  # e.g., ['T','Z','Y','X']

        for i, axis_char in enumerate(self.axis):
            if axis_char in "XY":
                continue
            slider = FloatRangeSlider(
                name=axis_char,
                min=0,
                max=shape[i] - 1,
                step=0.1,
                value=(0, shape[i] - 1),
                orientation="horizontal"
            )
            slider.native.setDecimals(0) 
            self.cut_sliders[axis_char] = slider

            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(f"{axis_char}:"))
            row_layout.addWidget(slider.native)
            container = QWidget()
            container.setLayout(row_layout)
            self.layout_cut_sliders.addWidget(container)

            # self.layout_cut_sliders.addRow(f"{axis_char} Range:", slider.native)

        # Add a button to update the cut preview
        self.update_preview_button = QPushButton("Update")
        self.update_preview_button.clicked.connect(self.update_cut_rectangle)
        self.layout_cut_sliders.addWidget(self.update_preview_button)

    def update_cut_table_from_metadata(self):

        if self.metadata_cut is None:
            return

        self.clear_metadata_cut_table()

        for label, value in self.metadata_cut.items():
            if label in ["raw_file", "file", "cut", "axis", "scale", "rawX", "rawY", "rawZ", "rawT", "rawC"] or label in self.metadata_files.columns:
                self.add_table_row(label, str(value), content_editable=False)
            else:
                self.add_table_row(label, str(value), content_editable=True)

    def update_file_table_from_metadata(self):

        if self.metadata_file is None:
            return

        self.clear_metadata_file_table()

        for label, value in self.metadata_file.items():
            if label in ["raw_file", "file", "axis", "scale", "shape"]:
                self.add_file_table_row(label, str(value), content_editable=False)
            else:
                self.add_file_table_row(label, str(value), content_editable=True)

    def update_cut_metadata_from_table(self):
        if self.metadata_cut is None:
            return
        
        for row in range(self.cut_table.rowCount()):
            label = self.cut_table.item(row, 0).text()
            item = self.cut_table.item(row, 1)
            if label in ["raw_file", "file", "cut", "axis", "scale", "rawX", "rawY", "rawZ", "rawT", "rawC"]:
                continue
            if item is not None:
                value = item.text()
                self.metadata_cut[label] = value

    def update_file_metadata_from_table(self):
        if self.metadata_file is None:
            return
        
        for row in range(self.file_table.rowCount()):
            label = self.file_table.item(row, 0).text()
            item = self.file_table.item(row, 1)
            if label in ["raw_file", "file", "cut", "axis", "scale", "shape"]:
                continue
            if item is not None:
                value = item.text()
                self.metadata_file[label] = value

    def expand_shape_over_dims(self, base_shape, dim_values_dict):
        dims_to_expand = sorted(dim_values_dict.keys())
        value_combinations = list(product(
            *[dim_values_dict[dim] for dim in dims_to_expand]
        ))

        shapes = []
        for vals in value_combinations:
            shape_copy = np.array(base_shape, dtype=float)
            for dim, val in zip(dims_to_expand, vals):
                shape_copy[:, dim] = val
            shapes.append(shape_copy)

        return shapes

    def update_cut_rectangle(self):
        if "CutPreview" not in self.viewer.layers:
            return
        
        selected = list(self.viewer.layers["CutPreview"].selected_data)
        if len(selected) == 0 and self.selected_cut_preview is None:
            QMessageBox.warning(self, "No Cut Selected", "Please select a cut preview to update.")
            return
        elif len(selected) > 1:
            QMessageBox.warning(self, "Multiple Selections", "Please select only one cut preview to update.")
            return

        if len(selected) == 1:
            self.selected_cut_preview = selected[0]

        cut_preview_layer = self.viewer.layers["CutPreview"].data[self.selected_cut_preview]
        ids = []
        ranges = {}
        for ax,val in self.cut_sliders.items():
            id = self.axis.index(ax)
            ids.append(id)
            ranges[id] = range(int(val.value[0]), int(val.value[1] + 1))
        
        cuts_preview_layer = self.expand_shape_over_dims(cut_preview_layer, ranges)
        self.viewer.layers["CutPreview"].data = cuts_preview_layer

    def save_metadata_cut(self):

        if self.metadata_cut is None or self._block:
            return

        self.update_cut_metadata_from_table()
        if self.metadata_cut["file"] in self.metadata_cuts["file"].values:
            loc = np.argwhere(self.metadata_cuts["file"] == self.metadata_cut["file"])[0, 0]
            self.metadata_cuts.iloc[loc,:] = self.metadata_cut
        else:
            self.metadata_cuts = pd.concat([self.metadata_cuts, self.metadata_cut.to_frame().T], ignore_index=True)

        self.save_metadata()

    def save_metadata_file(self):

        if self.metadata_file is None or self._block:
            return

        self.update_file_metadata_from_table()
        if self.metadata_file["file"] in self.metadata_files["file"].values:
            loc = np.argwhere(self.metadata_files["file"] == self.metadata_file["file"])[0, 0]
            self.metadata_files.iloc[loc,:] = self.metadata_file
        else:
            self.metadata_files = pd.concat([self.metadata_files, self.metadata_file.to_frame().T], ignore_index=True)

        for col in self.metadata_file.index:
            if col not in ["raw_file", "file", "axis", "scale", "shape"]:
                if self.metadata_cut is not None:
                    self.metadata_cut[col] = self.metadata_file[col]
                    self.update_cut_table_from_metadata()                
                self.metadata_cuts.loc[self.metadata_cuts["raw_file"] == self.selected_file, col] = self.metadata_file[col]

        self.save_metadata()

    def save_metadata(self):

        self.metadata_cuts.to_csv(self.root_path_cuts / "dataset_cuts.csv", index=False)
        self.metadata_files.to_csv(self.root_path_cuts / "dataset_files.csv", index=False)

    # def save_auxiliar(self):
    #     data = self.viewer.layers["Cuts"].data.copy()
    #     save_path = self.root_path_cuts / f"auxiliar/{self.selected_file_name}_cuts_data.npz"
    #     np.savez(save_path, cuts=np.array(self.cuts), data=data, counts=self.cut_count, axis=self.axis, scale=self.scale, allow_pickle=True)

    def axis_dropdown_edit(self):

        if self._block:
            return

        if self.making_cut:
            self._block = True
            QMessageBox.warning(self, "Cut in progress", "Please finish the current cut before changing the axis.")
            self.axis_dropdown.setCurrentText(self.axis)
            self._block = False
            return

        axis_map = {f"raw{i}": f"raw{j}" for i, j in zip(self.axis, self.axis_dropdown.currentText())}        
        self.axis = self.axis_dropdown.currentText()

        if self.metadata_cut is not None:
            columns_to_rename = [axis_map[col] if col in axis_map else col for col in self.metadata_cut.index]
            self.metadata_cut.index = columns_to_rename
        columns_to_rename = [axis_map[col] if col in axis_map else col for col in self.metadata_cuts.columns]
        self.metadata_cuts.columns = columns_to_rename
        self.metadata_files.loc[self.metadata_files["raw_file"] == self.selected_file,"axis"] = self.axis
        self.metadata_cuts.loc[self.metadata_cuts["raw_file"] == self.selected_file,"axis"] = self.axis
        if self.metadata_file is not None:
            self.metadata_file["axis"] = self.axis
        if self.metadata_cut is not None:
            self.metadata_cut["axis"] = self.axis
            
        self.update_cut_sliders()
        self.update_file_table_from_metadata()
        self.update_cut_table_from_metadata()
        self.save_metadata()
        # self.save_auxiliar()

    def scale_line_edit_check(self):

        if self._block:
            return

        if self.making_cut:
            self._block = True
            QMessageBox.warning(self, "Cut in progress", "Please finish the current cut before changing the axis.")
            self.scale_line_edit.setText(f"{self.scale}")
            self._block = False
            return

        def show_warning():
            QMessageBox.warning(self, "Invalid Scale Format", "Please enter a valid scale format, the tuple must be the same length as axis, e.g., axis='YXTC' and scale=(1.0, 1.0, 1.0, 1.0).")
            self.scale_line_edit.setText(f"{self.scale}")
        
        scale = eval(self.scale_line_edit.text())
        if isinstance(scale, tuple) and len(scale) == len(self.axis):
            self.scale = scale
            self.metadata_files.loc[self.metadata_files["raw_file"] == self.selected_file,"scale"] = str(self.scale)
            self.metadata_cuts.loc[self.metadata_cuts["raw_file"] == self.selected_file,"scale"] = str(self.scale)

            if self.metadata_file is not None:
                self.metadata_file["scale"] = str(self.scale)
            if self.metadata_cut is not None:
                self.metadata_cut["scale"] = str(self.scale)

            self.scale_layers()
            self.update_file_table_from_metadata()
            self.update_cut_table_from_metadata()
            self.save_metadata()
            # self.save_auxiliar()
        else:
            QTimer.singleShot(0, show_warning)

    def save_screenshot(self):

        pos = []
        for i,ax in enumerate(self.axis):
            if ax in "ZYX":
                continue
            if ax == "Z":
                self.viewer.dims.ndim = 3
            pos.append(self.cut_sliders[ax].value[0])            
            self.viewer.dims.set_current_step(i, int(self.cut_sliders[ax].value[0]))

        screenshot = self.viewer.screenshot()
        cut_path = self.root_path_cuts / f"screenshot/{self.name_cut()}.png"
        screenshot_image = Image.fromarray(screenshot)
        screenshot_image.save(str(cut_path))

    def save_cut_image(self):

        image_cut = self.crop_rotated_rectangle()
        if self.save_format.currentText() == "tif":
            image_cut = np.array(image_cut, dtype=np.float32)
            cut_path = self.root_path_cuts / f"data/{self.name_cut()}.tif"
            tifffile.imwrite(cut_path, image_cut)
        else:
            QMessageBox.warning(self, "Unsupported Format", f"The format {self.save_format.currentText()} is not supported for saving cuts.")
            return
        
        return image_cut.shape

    def crop_rotated_rectangle(self):

        image = self.viewer.layers[self.selected_file_name].data.copy()
        corners = self.viewer.layers["CutPreview"].data[0][:,-2:]

        width = np.linalg.norm(corners[1] - corners[0])
        height = np.linalg.norm(corners[3] - corners[0])
        out_h, out_w = int(round(height)), int(round(width))

        dst = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1]
        ], dtype=np.float32)

        A = self.compute_perspective_transform(dst, corners.astype(np.float32))
        yy, xx = np.indices((out_h, out_w))
        coords = np.stack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()], axis=0)
        src_coords = A @ coords
        src_coords /= src_coords[2]
        x_src = src_coords[0].reshape(out_h, out_w)
        y_src = src_coords[1].reshape(out_h, out_w)

        if image.ndim == 2:
            cropped = map_coordinates(image, [x_src, y_src], order=1, mode='reflect')
        else:
            # Determine order of axes and ranges
            axis_ranges = {}
            output_shape = []
            iter_axes = []
            axis_min = []
            
            for i, ax in enumerate(self.axis):
                if ax in "YX":
                    continue
                axis_min.append(int(self.cut_sliders[ax].value[0]))
                r = range(int(self.cut_sliders[ax].value[0]), int(self.cut_sliders[ax].value[1]) + 1)
                axis_ranges[ax] = r
                output_shape.append(len(r))
                iter_axes.append(ax)

            # Add spatial dimensions (Y, X)
            output_shape += [out_h, out_w]
            cropped = np.zeros(output_shape, dtype=image.dtype)

            # Loop through all non-spatial axes
            for idx, idx_values in enumerate(product(*axis_ranges.values())):
                slicer = [slice(None)] * image.ndim
                for ax, val in zip(axis_ranges.keys(), idx_values):
                    axis_idx = self.axis.index(ax)
                    slicer[axis_idx] = val

                # Extract 2D frame to crop
                frame = image[tuple(slicer)]
                frame_cropped = map_coordinates(frame, [x_src, y_src], order=1, mode='reflect')

                # Place in output
                output_index = tuple(np.array(idx_values)-np.array(axis_min)) + (slice(None), slice(None))
                cropped[output_index] = frame_cropped

        return cropped

    def compute_perspective_transform(self, dst, src):
        A = []
        for (x_dst, y_dst), (x_src, y_src) in zip(dst, src):
            A.append([x_dst, y_dst, 1, 0, 0, 0, -x_src * x_dst, -x_src * y_dst])
            A.append([0, 0, 0, x_dst, y_dst, 1, -y_src * x_dst, -y_src * y_dst])
        A = np.array(A)
        b = src.flatten()
        h = np.linalg.solve(A, b)
        H = np.append(h, 1).reshape(3, 3)
        return H

    def start_cut(self):

        # Disable everything
        self.button_file_next.setDisabled(True)
        self.button_file_previous.setDisabled(True)
        self.dropdown_file_menu.setDisabled(True)
        self.axis_dropdown.setDisabled(True)
        self.scale_line_edit.setDisabled(True)
        self.file_table.setDisabled(True)
        self.add_file_row_button.setDisabled(True)
        self.delete_file_row_button.setDisabled(True)
        self.cut_start.setDisabled(True)
        self.cut_end.setDisabled(False)
        self.cut_load.setDisabled(True)
        self.save_format.setDisabled(False)
        self.cut_table.setDisabled(True)
        self.add_row_button.setDisabled(True)
        self.delete_row_button.setDisabled(True)

        if self.making_cut:
            QMessageBox.warning(self, "Cut in progress", "Please finish the current cut before starting a new one.")
            return
        
        self.making_cut = True
        self.cut_count += 1
        self.selected_cut = self.cut_count
        self.selected_cut_preview = 0
        rectangle = np.zeros((4, self.viewer.layers[self.selected_file_name].data.ndim), dtype=float)
        rectangle[:, self.axis.index("Y")] = np.array([0.25, 0.75, 0.75, 0.25])*self.viewer.layers[self.selected_file_name].data.shape[self.axis.index("X")]  # X coordinates
        rectangle[:, self.axis.index("X")] = np.array([0.25, 0.25, 0.75, 0.75])*self.viewer.layers[self.selected_file_name].data.shape[self.axis.index("Y")]  # Y coordinates
        if "CutPreview" in self.viewer.layers:
            del self.viewer.layers["CutPreview"]
        self.viewer.add_shapes([rectangle], name="CutPreview", edge_color="red", face_color="transparent", edge_width=2, opacity=1)
        self.viewer.layers["CutPreview"].mode = "select"
        if self.metadata_cut is None:
            self.metadata_cut = pd.Series({
                "raw_file": self.selected_file,
                "file": f"{self.name_cut()}",
                "cut": self.selected_cut,
                "axis": self.axis_dropdown.currentText(),
                "scale": self.scale_line_edit.text(),
                "shape": self.viewer.layers[self.selected_file_name].data.shape,
                "rawX": None,
                "rawY": None,
                "rawZ": None,
                "rawT": None,
                "rawC": None,
            })
            for i in self.metadata_cuts.columns:
                if i not in self.metadata_cut.index.values:
                    if i in self.metadata_file.index:
                        self.metadata_cut[i] = self.metadata_file[i]
                    else:
                        self.metadata_cut[i] = ""
        else:
            self.metadata_cut["raw_file"] = self.selected_file
            self.metadata_cut["file"] = f"{self.name_cut()}"
            self.metadata_cut["cut"] = self.selected_cut
            self.metadata_cut["axis"] = self.axis_dropdown.currentText()
            self.metadata_cut["scale"] = self.scale_line_edit.text()
            self.metadata_cut["shape"] = self.viewer.layers[self.selected_file_name].data.shape
            self.metadata_cut["rawX"] = None
            self.metadata_cut["rawY"] = None
            self.metadata_cut["rawZ"] = None
            self.metadata_cut["rawT"] = None
            self.metadata_cut["rawC"] = None

        self.metadata_cuts = pd.concat([self.metadata_cuts, self.metadata_cut.to_frame().T], ignore_index=True)
            
        self.update_cut_sliders()
        self.update_cut_table_from_metadata()
        self.update_cut_rectangle()
        self.scale_layers()

        self.widget_cut_sliders.setVisible(True)
        self.widget_cut_metadata.setVisible(True)

    def end_cut(self):

        if not self.making_cut:
            QMessageBox.warning(self, "No Cut in Progress", "Please start a cut before ending it.")
            return

        self.update_cut_rectangle()
        self.making_cut = False
        self.selected_cut_preview = None
        data = self.viewer.layers["CutPreview"].data.copy()
        cut_id = [self.cut_count] * len(data)

        self.cuts += cut_id
        data_ = self.viewer.layers["Cuts"].data.copy()
        if len(data_) == 0:
            data_ = np.zeros((0, 4, self.viewer.layers[self.selected_file_name].data.ndim), dtype=float)
        data_ = np.concatenate((data_, data), axis=0)
        self.viewer.layers["Cuts"].data = data_

        shape = self.save_cut_image()
        self.save_screenshot()
        self.update_cut_metadata_from_table()
        self.metadata_cut[f"raw{self.axis[-1]}"] = str(data[0][:, -1].tolist())
        self.metadata_cut[f"raw{self.axis[-2]}"] = str(data[0][:, -2].tolist())
        for i in range(0, len(self.axis)-2):
            self.metadata_cut[f"raw{self.axis[i]}"] = str(list(np.round(self.cut_sliders.get(self.axis[i], FloatRangeSlider()).value, decimals=0).astype(int))) if self.axis[i] in self.cut_sliders else str([])
        self.metadata_cut["shape"] = shape
        self.update_cut_table_from_metadata()
        self.save_metadata_cut()
        # self.save_auxiliar()
        self.viewer.layers.selection.active = self.viewer.layers["Cuts"]
        self.viewer.layers["Cuts"].mode = "select"

        del self.viewer.layers["CutPreview"]

        self.widget_cut_sliders.setVisible(False)
        self.widget_cut_metadata.setVisible(True)

        for widget in self.findChildren(QWidget):
            widget.setDisabled(False)

    def load_cut(self):

        if self.making_cut:
            QMessageBox.warning(self, "Cut in progress", "Please finish the current cut before loading a new one.")
            return

        if len(list(self.viewer.layers["Cuts"].selected_data)) == 0:
            QMessageBox.warning(self, "No Cut Selected", "Please select a cut to load.")
            return

        if self.viewer.layers["Cuts"].selected_data is not None:
            selected = list(self.viewer.layers["Cuts"].selected_data)[0]
            self.selected_cut = self.cuts[selected]
            self.metadata_cut = self.metadata_cuts[self.metadata_cuts["file"] == f"{self.name_cut()}"].iloc[0]

        self.widget_cut_sliders.setVisible(False)
        self.widget_cut_metadata.setVisible(True)

        self.update_cut_table_from_metadata()

    def delete_cut(self):
        layer = self.viewer.layers["Cuts"]
        if layer is None or layer.__class__.__name__ != "Shapes":
            QMessageBox.warning(self, "Unknown Deletion Cut", "No cut has been selected.")
            return
        if len(layer.selected_data) > 1:
            QMessageBox.warning(self, "Multiple deletions", "Multiple cuts selected at the same time, please, just select one.")
            return
        selected = self.cuts[list(layer.selected_data)[0]]
        keep = np.array(self.cuts) != selected
        self.cuts = np.array(self.cuts)[keep].tolist()
        layer.data = np.array(layer.data)[keep].tolist()
        self.metadata_cuts = self.metadata_cuts[self.metadata_cuts["file"] != f"{self.name_cut()}"]
        self.metadata_cuts.to_csv(self.root_path_cuts / "dataset_cuts.csv", index=False)
        if os.path.exists(self.root_path_cuts / f"data/{self.name_cut()}.tif"):
            os.remove(self.root_path_cuts / f"data/{self.name_cut()}.tif")
        if os.path.exists(self.root_path_cuts / f"screenshot/{self.name_cut()}.png"):
            os.remove(self.root_path_cuts / f"screenshot/{self.name_cut()}.png")

        self.widget_cut_sliders.setVisible(False)
        self.widget_cut_metadata.setVisible(False)

    def confirm_and_remove_selected(self):
        indices = list(self.viewer.layers["Cuts"].selected_data)
        if not indices:
            QMessageBox.warning(self, "Unknown Deletion Cut", "No cut has been selected.")
            return
        msg = QMessageBox()
        msg.setWindowTitle("Confirm Shape Deletion")
        msg.setText("Do you want to delete selected cut?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        if msg.exec_() == QMessageBox.Yes:
            self.delete_cut()
