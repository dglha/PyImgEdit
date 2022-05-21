import sys

import numpy as np
from PIL.ImageQt import ImageQt
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget,
    QMainWindow,
    QDialog,
    QFileDialog,
    QGraphicsScene,
    QErrorMessage,
)
from PyQt5.QtGui import QPixmap, QImage
from matplotlib import pyplot as plt
from functools import wraps, partial

from PIL import Image

from main import Ui_MainWindow
from models.image_operation import ImageOperation
from models.effect_filter import EffectFilter
import pathlib


def is_image_loaded(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if type(self.current_image) == list:
            return self.display_error_message("Please choose an image first!")
        return func(self, *args, **kwargs)

    return wrapper


class ImageEditor(QMainWindow, Ui_MainWindow):

    # Define properties
    original_image = [0]
    current_image = [0]
    temp_img = [0]
    previous_image = [0]

    _image_blur = [0]
    _image_sharpen = [0]
    _image_color = [0]
    _image_contrast = [0]
    _image_bright = [0]

    image_height = 0
    image_width = 0

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.set_slider_enabled(False)

        # Connect signals to slots
        # Page 1
        self.open_button.clicked.connect(self.open_image)
        self.histogram_equal_button.clicked.connect(self.histogram_equalization)
        self.view_histogram_button.clicked.connect(self.view_histogram)
        self.invert_image_button.clicked.connect(self.invert_image)
        self.log_transform_button.clicked.connect(self.log_transform)
        self.gamma_transform_button.clicked.connect(self.gamma_transform)

        # Page 1 slider
        self.blur_slider.valueChanged.connect(self.blur_image)
        self.bright_slider.valueChanged.connect(self.bright_image)
        self.color_slider.valueChanged.connect(self.color_image)
        self.contrast_slider.valueChanged.connect(self.contrast_image)
        self.sharpen_slider.valueChanged.connect(self.sharpen_image)

        # Page 2
        self.flip_updown_button.clicked.connect(
            partial(self.transpose_image, Image.Transpose.FLIP_TOP_BOTTOM)
        )
        self.flip_leftright_button.clicked.connect(
            partial(self.transpose_image, Image.Transpose.FLIP_LEFT_RIGHT)
        )

        self.rotate_left_button.clicked.connect(
            partial(self.transpose_image, Image.Transpose.ROTATE_90)
        )
        self.rotate_right_button.clicked.connect(
            partial(self.transpose_image, Image.Transpose.ROTATE_270)
        )

        # Page 3 - Filter
        self.pink_dream_filter_button.clicked.connect(self.apply_pink_dream)
        self.cyperpunk_filter_button.clicked.connect(self.apply_cyperpunk)
        self.snowy_filter_button.clicked.connect(self.apply_snowy)
        self.pastel_filter_button.clicked.connect(self.apply_pastel)
        self.firestorm_filter_button.clicked.connect(self.apply_firestorm)
        self.ice_filter_button.clicked.connect(self.apply_ice)
        self.darkness_filter_button.clicked.connect(self.apply_darkness)
        self.gray_nostalgia_filter_button.clicked.connect(self.apply_gray_nos)
        self.sweet_dream_filter_button.clicked.connect(self.apply_sweet_dream)
        self.cartoon_filter_button.clicked.connect(self.apply_cartoon)

        self.next_page_button.clicked.connect(self.to_next_page)
        self.prev_page_button.clicked.connect(self.to_prev_page)
        self.undo_button.clicked.connect(self.undo_action)
        self.original_image_button.clicked.connect(self.undo_to_original)

    def show_image_info_status_bar(self):
        info = ImageOperation.get_information(self.original_image)
        file_path = pathlib.Path(info["name"])
        msg = f"{file_path.name} MODE: {info['mode']} SIZE: {info['size'] } FORMAT: {info['format']}"
        self.statusbar.showMessage(msg)

    def set_slider_enabled(self, enabled: bool):
        self.blur_slider.setEnabled(enabled)
        self.sharpen_slider.setEnabled(enabled)
        self.color_slider.setEnabled(enabled)
        self.bright_slider.setEnabled(enabled)
        self.contrast_slider.setEnabled(enabled)

    def reset_slider_value(self):
        self.blur_slider.setValue(0)
        self.sharpen_slider.setValue(10)
        self.color_slider.setValue(10)
        self.bright_slider.setValue(10)
        self.contrast_slider.setValue(10)

    def open_image(self):
        open_image_dialog = QFileDialog()
        open_image_dialog.setMimeTypeFilters({"image/jpeg", "image/png"})
        image_path = QFileDialog.getOpenFileName(open_image_dialog, "Select image", "/")

        if image_path[0]:
            self.current_image = Image.open(image_path[0])

            self.original_image = self.previous_image = self.current_image
            self.display_image()
            self.show_image_info_status_bar()
            self.set_slider_enabled(True)

        else:
            pass

    def display_image(self):
        """
        Set display size to the size of the image display (Graphic view)
        """
        image_scene = QGraphicsScene()
        self.temp_img = ImageQt(self.current_image)
        # self.temp_img = QImage(self.temp_img)
        pixmap = QPixmap.fromImage(self.temp_img)
        w, h = self.scale_image(pixmap.width(), pixmap.height())
        pixmap = pixmap.copy().scaled(
            int(w), int(h), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        image_scene.addPixmap(pixmap.copy())
        self.graphicsView.setScene(image_scene)

    def scale_image(self, width, height):
        k = self.graphicsView.frameGeometry().height() / height
        if width * k <= self.graphicsView.frameGeometry().width():
            w = width * k
            h = self.graphicsView.frameGeometry().height()
        else:
            k = self.graphicsView.frameGeometry().width() / width
            w = self.graphicsView.frameGeometry().width()
            h = height * k

        return w, h

    def set_previous_image(self):
        self.undo_button.setEnabled(True)
        self.original_image_button.setEnabled(True)
        self.previous_image = self.current_image

    def display_error_message(self, msg):
        e = QErrorMessage(self)
        e.setWindowTitle("Error")
        e.showMessage(msg)

    """
        Funcs
    """

    @pyqtSlot()
    @is_image_loaded
    def histogram_equalization(self):
        # Store current image to previous
        self.set_previous_image()

        self.current_image = ImageOperation.histogram_equalization(self.current_image)
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def view_histogram(self):
        image = np.array(self.current_image)
        if len(image.shape) > 2:
            self.display_error_message("Histogram equalization first!!")
            return
        # histogram = np.bincount(image[:, :, 2].ravel(), minlength=256)
        plt.figure(num="Image Histogram")
        plt.hist(image)
        plt.xlabel("Intensity levels")
        plt.ylabel("No. of pixels")
        # show the stem plot
        plt.show()

    @pyqtSlot()
    @is_image_loaded
    def log_transform(self):
        self.set_previous_image()
        self.current_image = ImageOperation.log_transform(self.current_image)
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def gamma_transform(self):
        self.set_previous_image()

        # Create input dialog
        gamma_value, is_done = QtWidgets.QInputDialog.getDouble(
            self, "Input dialog", "Enter gamma value:"
        )
        if not is_done:
            return

        try:
            gamma_value = float(gamma_value)
        except Exception:
            self.display_error_message("Please input right format for gamma value!")
            return

        self.current_image = ImageOperation.gamma_transform(
            self.current_image, gamma_value
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def invert_image(self):
        self.set_previous_image()
        self.current_image = ImageOperation.invert_image(self.current_image)
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def transpose_image(self, direction: Image.Transpose, *args):
        self.set_previous_image()
        self.current_image = ImageOperation.transpose_image(
            self.current_image, direction
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def blur_image(self):
        if type(self._image_blur) == list:
            self.set_previous_image()

        blur_value = self.blur_slider.value()

        # Avoid calling again @@
        self.blur_slider.valueChanged.disconnect()
        self.blur_slider.setValue(blur_value)
        self.blur_slider.valueChanged.connect(self.blur_image)

        if blur_value == 0:
            return

        self._image_blur = self.current_image
        self.current_image = ImageOperation.blur_image(
            self.previous_image, radius=blur_value
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def bright_image(self):
        if type(self._image_bright) == list:
            self.set_previous_image()

        value = self.bright_slider.value()
        bright_value = value / 10

        self.bright_slider.valueChanged.disconnect()
        self.bright_slider.setValue(value)
        self.bright_slider.valueChanged.connect(self.bright_image)

        if bright_value == 0:
            return

        self._image_bright = self.current_image
        self.current_image = ImageOperation.brightness_image(
            self.previous_image, factor=bright_value
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def color_image(self):
        if type(self._image_bright) == list:
            self.set_previous_image()

        value = self.color_slider.value()
        color_value = value / 10

        self.color_slider.valueChanged.disconnect()
        self.color_slider.setValue(value)
        self.color_slider.valueChanged.connect(self.color_image)

        if color_value == 0:
            return

        self._image_bright = self.current_image
        self.current_image = ImageOperation.color_image(
            self.previous_image, factor=color_value
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def contrast_image(self):
        if type(self._image_contrast) == list:
            self.set_previous_image()

        value = self.contrast_slider.value()
        contrast_value = value / 10

        self.contrast_slider.valueChanged.disconnect()
        self.contrast_slider.setValue(value)
        self.contrast_slider.valueChanged.connect(self.contrast_image)

        if contrast_value == 0:
            return

        self._image_contrast = self.current_image
        self.current_image = ImageOperation.color_image(
            self.previous_image, factor=contrast_value
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def sharpen_image(self):
        if type(self._image_sharpen) == list:
            self.set_previous_image()

        value = self.sharpen_slider.value()
        sharpen_value = value / 10

        self.sharpen_slider.valueChanged.disconnect()
        self.sharpen_slider.setValue(value)
        self.sharpen_slider.valueChanged.connect(self.sharpen_image)

        if sharpen_value == 0:
            return

        self._image_contrast = self.current_image
        self.current_image = ImageOperation.sharpen_image(
            self.previous_image, factor=sharpen_value
        )
        self.display_image()

    """
    Filter
    """

    @pyqtSlot()
    @is_image_loaded
    def apply_pink_dream(self):
        self.set_previous_image()
        self.current_image = Image.fromarray(
            EffectFilter.pink_dream(self.current_image)
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def apply_cyperpunk(self):
        self.set_previous_image()
        self.current_image = Image.fromarray(
            EffectFilter.cyperpunk_2077(self.current_image)
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def apply_snowy(self):
        self.set_previous_image()
        self.current_image = Image.fromarray(
            EffectFilter.snowy(self.current_image)
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def apply_pastel(self):
        self.set_previous_image()
        self.current_image = Image.fromarray(
            EffectFilter.pastel(self.current_image)
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def apply_firestorm(self):
        self.set_previous_image()
        self.current_image = Image.fromarray(
            EffectFilter.firestorm(self.current_image)
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def apply_ice(self):
        self.set_previous_image()
        self.current_image = Image.fromarray(
            EffectFilter.ice(self.current_image)
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def apply_darkness(self):
        self.set_previous_image()
        self.current_image = Image.fromarray(
            EffectFilter.darkness(self.current_image)
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def apply_gray_nos(self):
        self.set_previous_image()
        self.current_image = Image.fromarray(
            EffectFilter.gray_nostalgia(self.current_image)
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def apply_sweet_dream(self):
        self.set_previous_image()
        self.current_image = Image.fromarray(
            EffectFilter.sweet_dream(self.current_image)
        )
        self.display_image()

    @pyqtSlot()
    @is_image_loaded
    def apply_cartoon(self):
        self.set_previous_image()
        self.current_image = Image.fromarray(
            EffectFilter.cartoon(self.current_image)
        )
        self.display_image()

    """
    App
    """

    @pyqtSlot()
    def to_next_page(self):
        next_index = self.stackedWidget.currentIndex() + 1
        if self.stackedWidget.count() > next_index:
            self.stackedWidget.setCurrentIndex(next_index)

    @pyqtSlot()
    def to_prev_page(self):
        prev_index = (
            (self.stackedWidget.currentIndex() - 1)
            if self.stackedWidget.currentIndex() > 0
            else 0
        )
        self.stackedWidget.setCurrentIndex(prev_index)

    @pyqtSlot()
    def undo_action(self):
        self.reset_slider_value()
        self.undo_button.setEnabled(True)
        self.current_image = self.previous_image
        self.display_image()

    @pyqtSlot()
    def undo_to_original(self):
        self.reset_slider_value()
        self.original_image_button.setEnabled(True)
        self.current_image = self.original_image
        self.display_image()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImageEditor()
    window.show()
    sys.exit(app.exec())
