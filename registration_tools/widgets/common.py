from qtpy.QtWidgets import QVBoxLayout, QWidget, QSlider, QLabel, QPushButton, QLineEdit, QHBoxLayout, QDialog, QVBoxLayout, QLineEdit, QPushButton, QFileDialog, QMessageBox, QHBoxLayout
from qtpy.QtCore import Qt, QTimer
import os

def create_button(self, label_text, callback):
    button = QPushButton(label_text)
    button.clicked.connect(callback)
    self.layout.addWidget(button)
    return button

def create_slider(self, label_text, min_value, max_value, set_value, connect=None):
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

    if connect is not None:
        slider.valueChanged.connect(connect)

    return widget


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
