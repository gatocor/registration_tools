from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QMenu, QAction, QApplication
)
from qtpy.QtCore import Qt, QPoint
import sys
import napari

from .data import get_skimage_dataset_functions, load_data
from .exposure import get_skimage_exposure_functions, ExposureWidget
from .feature import get_skimage_feature_functions, FeatureWidget
from .filters import get_skimage_filters_functions, FilterWidget
from .morphology import get_skimage_morphology_functions, MorphologyWidget

class SkimageMenuWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.setWindowTitle("Skimage Menu Example")
        self.viewer = viewer

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Use QPushButton instead of QToolButton (more reliable)
        self.menu_button = QPushButton("skimage")
        self.menu_button.setMinimumHeight(30)
        self.menu_button.clicked.connect(self.show_menu)
        layout.addWidget(self.menu_button)

        # Build the nested menu
        self.menu = QMenu(self)

        data_menu = QMenu("Data", self.menu)
        for name, func in get_skimage_dataset_functions().items():
            data_menu.addAction(name, lambda f=func: load_data(self.viewer, f.__name__))
        self.menu.addMenu(data_menu)

        filter_menu = QMenu("Exposure", self.menu)
        for name, func in get_skimage_exposure_functions().items():
            filter_menu.addAction(name, lambda f=func: self.viewer.window.add_dock_widget(ExposureWidget(self.viewer, f.__name__)))
        self.menu.addMenu(filter_menu)

        filter_menu = QMenu("Feature", self.menu)
        for name, func in get_skimage_feature_functions().items():
            filter_menu.addAction(name, lambda f=func: self.viewer.window.add_dock_widget(FeatureWidget(self.viewer, f.__name__)))
        self.menu.addMenu(filter_menu)

        filter_menu = QMenu("Filters", self.menu)
        for name, func in get_skimage_filters_functions().items():
            filter_menu.addAction(name, lambda f=func: self.viewer.window.add_dock_widget(FilterWidget(self.viewer, f.__name__)))
        self.menu.addMenu(filter_menu)

        filter_menu = QMenu("Morphology", self.menu)
        for name, func in get_skimage_morphology_functions().items():
            filter_menu.addAction(name, lambda f=func: self.viewer.window.add_dock_widget(MorphologyWidget(self.viewer, f.__name__)))
        self.menu.addMenu(filter_menu)

    def show_menu(self):
        # Position the menu just below the button
        pos = self.menu_button.mapToGlobal(QPoint(0, self.menu_button.height()))
        self.menu.exec_(pos)
