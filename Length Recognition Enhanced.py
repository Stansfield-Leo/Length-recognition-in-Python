import cv2
import numpy as np
import os
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QFileDialog, QScrollArea, QLineEdit, QFormLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_files = []
        self.current_index = -1
        self.pixel_to_um = 1.0  # 默认的像素到微米转换系数

    def initUI(self):
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(1080, 810)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidget(self.label)
        self.scroll_area.setWidgetResizable(True)

        self.prev_button = QPushButton('Previous', self)
        self.prev_button.clicked.connect(self.show_prev_image)
        self.next_button = QPushButton('Next', self)
        self.next_button.clicked.connect(self.show_next_image)
        self.load_button = QPushButton('Load Folder', self)
        self.load_button.clicked.connect(self.load_folder)

        self.pixel_edit = QLineEdit(self)
        self.pixel_edit.setPlaceholderText("Enter pixels")
        self.um_edit = QLineEdit(self)
        self.um_edit.setPlaceholderText("Enter micrometers")
        self.coefficient_edit = QLineEdit(self)
        self.coefficient_edit.setPlaceholderText("Enter pixel-to-um coefficient")

        self.coefficient_edit.returnPressed.connect(self.update_coefficient)
        self.pixel_edit.returnPressed.connect(self.convert_to_um)
        self.um_edit.returnPressed.connect(self.convert_to_pixels)

        form_layout = QFormLayout()
        form_layout.addRow("Pixels:", self.pixel_edit)
        form_layout.addRow("Micrometers:", self.um_edit)
        form_layout.addRow("Coefficient:", self.coefficient_edit)

        hbox = QHBoxLayout()
        hbox.addWidget(self.prev_button)
        hbox.addWidget(self.next_button)
        hbox.addWidget(self.load_button)

        vbox = QVBoxLayout()
        vbox.addWidget(self.scroll_area)
        vbox.addLayout(hbox)
        vbox.addLayout(form_layout)

        self.setLayout(vbox)
        self.setWindowTitle('Image Viewer')
        self.setFixedSize(1100, 1000)
        self.show()

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder_path:
            self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if self.image_files:
                self.current_index = 0
                self.show_image()

    def show_image(self):
        if 0 <= self.current_index < len(self.image_files):
            image_path = self.image_files[self.current_index]
            image = cv2.imread(image_path)

            if image is not None:
                self.detect_and_display(image)
            else:
                print(f"Failed to read the image: {image_path}")

    def show_prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def show_next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image()

    def update_coefficient(self):
        try:
            self.pixel_to_um = float(self.coefficient_edit.text())
        except ValueError:
            self.coefficient_edit.setText("Invalid input")

    def convert_to_um(self):
        try:
            pixels = float(self.pixel_edit.text())
            micrometers = pixels * self.pixel_to_um
            self.um_edit.setText(f"{micrometers:.2f}")
        except ValueError:
            self.um_edit.setText("Invalid input")

    def convert_to_pixels(self):
        try:
            micrometers = float(self.um_edit.text())
            pixels = micrometers / self.pixel_to_um
            self.pixel_edit.setText(f"{pixels:.2f}")
        except ValueError:
            self.pixel_edit.setText("Invalid input")

    def detect_and_display(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)

        # Apply a Gaussian blur to the grayscale image
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Apply edge detection using Canny
        edges = cv2.Canny(blurred, 30, 150)

        # Perform morphological operations to close small gaps in the edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour, which should correspond to the liquid droplet
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Calculate the minimum width within the bounding box
            min_width = w
            min_width_row = y

            for row in range(y, y + h):
                nonzero_cols = np.where(eroded[row, x:x + w] > 0)[0]
                if len(nonzero_cols) > 1:
                    width = nonzero_cols[-1] - nonzero_cols[0]
                    if width < min_width:
                        min_width = width
                        min_width_row = row

            # Convert the minimum width to micrometers
            min_width_um = min_width * self.pixel_to_um

            # Find points for the minimum width
            pt1 = (x + np.where(eroded[min_width_row, x:x + w] > 0)[0][0], min_width_row)
            pt2 = (x + np.where(eroded[min_width_row, x:x + w] > 0)[0][-1], min_width_row)
            center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

            # Draw the minimum width line and label it
            cv2.line(image, pt1, pt2, (0, 0, 255), 2)
            cv2.putText(image, f"Width: {min_width:.2f} px / {min_width_um:.2f} µm", 
                        (center[0] - 100, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Update the UI with the detected width
            self.pixel_edit.setText(f"{min_width:.2f}")
            self.um_edit.setText(f"{min_width_um:.2f}")

        # Convert the processed image to RGB for display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimg = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg).scaled(1080, 810, Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageWindow()
    sys.exit(app.exec_())
