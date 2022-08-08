import os
from os import path
import subprocess

from tokenize import group

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QUrl, QProcess, pyqtSignal, QThread
from PyQt5.QtGui import QIcon, QIntValidator, QFont, QTextCursor, QPixmap, QImage
from PyQt5.QtWidgets import QPushButton, QLabel, QRadioButton, QMenuBar, QMenu, QCheckBox, QWidget, QFileDialog, \
    QMainWindow, QMessageBox, QLineEdit, QGroupBox, QVBoxLayout, QButtonGroup, QTextBrowser, QStatusBar, QApplication, QSlider, QProgressBar
from PyQt5.QtCore import Qt

from gui_utils import yolact_video_segmentation, yolact_single_image_segmentation, yolact_images_segmentation, yolar_video_detection, yolar_single_image_detection, yolar_images_detection, \
    retinaface_video_detection, retinaface_single_image_detection, retinaface_images_detection

import json
import sys
from threading import Thread
from PyQt5.QtWidgets import QApplication
import time

class Ui_Deidentification(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video = ""
        self.input_image = ""
        self.images_input_path = ""
        self.output_image = ""
        self.images_output_path = ""
        self.video_output = ""
        self.method = ""
        self.classes = ""

    def see_film(self):
        os.system('python videowindow.py')

    def choose_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, 'Choose a video file', '', 'Video files | (*.avi *.mp4);')
        url = QUrl.fromLocalFile(video_path)
        print("Selected video: ", url.fileName())
        self.video = video_path
        self.video_output = "../output/video/masking/" + url.fileName()
        return video_path

    def choose_images(self):
        if self.checkBox.isChecked():
            input_image, _ = QFileDialog.getOpenFileName(self, 'Choose an input image file', '', 'Image files | (*.jpg *.png);')
            url_input = QUrl.fromLocalFile(input_image)
            print("Selected input image: ", url_input.fileName())
            self.input_image = input_image
            self.output_image = "../output/image/masking/" + url_input.fileName()
            return input_image
        else:
            images_input_path = QFileDialog.getExistingDirectory(self, 'Choose an input folder')
            url_input = QUrl.fromLocalFile(images_input_path)
            print("Selected input images: ", url_input.fileName())
            self.images_input_path = images_input_path
            self.images_output_path = "../output/image"
            return images_input_path

    def start_segmentation(self):
        if self.label_on_off.isChecked():
            label_dis = 'display'    
        else:
            label_dis = 'undisplay'

        self.classes = ''
        if self.pedestrain_class.isChecked():
            self.classes += "0"
        if self.vehicle_class.isChecked():
            self.classes += "1"

        class_index = {}
        class_index['class'] = []
        class_index['class'].append({
            'index' : self.classes
        })

        basepath = path.dirname(__file__)
        filepath = path.abspath(path.join(basepath,  "class.json"))
        with open(filepath, 'w') as outfile:
            json.dump(class_index, outfile, indent=4)

        if self.score_threshold.value() and self.topk.text():
            score_threshold = float(self.score_threshold.value() / 100)
            topk = int(self.topk.text())
            deid_level = int(self.deid_level.value())
            if self.video and self.video_multiframe.text():
                video_multiframe = int(self.video_multiframe.text())
                yolact_video_segmentation(self.video, self.video_output, score_threshold, topk, video_multiframe, label_dis, self.classes, deid_level)
            elif self.input_image:
                yolact_single_image_segmentation(self.input_image, self.output_image, score_threshold, topk, label_dis, self.classes, deid_level)
            elif self.images_input_path:
                yolact_images_segmentation(self.images_input_path, self.images_output_path, score_threshold, topk, label_dis, self.classes, deid_level)
            else:
                QMessageBox.warning(self, 'Warning', 'Method or Data not specified', QMessageBox.Ok)
        else:
            QMessageBox.warning(self, 'Warning', 'Method or Data not specified', QMessageBox.Ok)
        
    def start_detection(self):
        if self.label_on_off.isChecked():
            label_dis = 'display'    
        else:
            label_dis = 'undisplay'

        self.classes = ''            
        if self.pedestrain_class.isChecked():
            self.classes += "0 "
        if self.vehicle_class.isChecked():
            self.classes += "1 2 3 5 7"

        if self.score_threshold.value():
            score_threshold = float(self.score_threshold.value() / 100)
            deid_level = int(self.deid_level.value())
            if self.video and self.video_multiframe.text():
                yolar_video_detection(self.video, self.video_output, score_threshold, self.method, label_dis, self.classes, deid_level)
            elif self.input_image:
                yolar_single_image_detection(self.input_image, self.output_image, score_threshold, self.method, label_dis, self.classes, deid_level)
            elif self.images_input_path:
                yolar_images_detection(self.images_input_path, self.images_output_path, score_threshold, self.method, label_dis, self.classes, deid_level)
            else:
                QMessageBox.warning(self, 'Warning', 'Method or Data not specified', QMessageBox.Ok)
        else:
            QMessageBox.warning(self, 'Warning', 'Method or Data not specified', QMessageBox.Ok)

    def start_face_detection(self):
        if self.label_on_off.isChecked():
            label_dis = 'display'    
        else:
            label_dis = 'undisplay'

        if self.score_threshold.value():
            score_threshold = float(self.score_threshold.value() / 100)
            deid_level = int(self.deid_level.value())        
            if self.video and self.video_multiframe.text():
                retinaface_video_detection(self.video, self.video_output, self.method, score_threshold, label_dis, deid_level)
            elif self.input_image:
                retinaface_single_image_detection(self.input_image, self.output_image, self.method, score_threshold, label_dis, deid_level)
            elif self.images_input_path:
                retinaface_images_detection(self.images_input_path, self.images_output_path, self.method, score_threshold, label_dis, deid_level)
            else:
                QMessageBox.warning(self, 'Warning', 'Model or data not specified', QMessageBox.Ok)
        else:
            QMessageBox.warning(self, 'Warning', 'Model or data not specified', QMessageBox.Ok)
        
    def radio_masking_clicked(self):

        self.group_1 = QButtonGroup(self.centralwidget)
        self.group_1.addButton(self.radio_images)
        self.group_1.addButton(self.radio_video)
        self.group_1.removeButton(self.radio_images)
        self.group_1.removeButton(self.radio_video)
        self.radio_images.setChecked(False)
        self.radio_video.setChecked(False)
        self.checkBox.setChecked(False)
        self.label_on_off.setChecked(False)
        self.group_1.addButton(self.radio_images)
        self.group_1.addButton(self.radio_video)

        self.score_threshold_label.show()
        self.score_threshold.show()
        self.score_threshold_value.show()
        self.deid_level_label.show()
        self.deid_level.show()
        self.deid_level_value.show()
        self.pedestrain_class.show()
        self.vehicle_class.show()
        self.topk_label.show()
        self.topk.show()
        self.images_or_video.show()
        self.radio_images.show()
        self.radio_video.show()
        self.segmentation.show()

        self.checkBox.hide()
        self.label_on_off.hide()
        self.upload_images.hide()
        self.upload_video_button.hide()
        self.video_multiframe_label.hide()
        self.video_multiframe.hide()
        self.detection.hide()
        self.face_detection.hide()
        self.face_class.hide()

    def radio_detection_clicked(self):

        if self.radio_blurring.isChecked():
            self.method = "blur"
        elif self.radio_mosaic.isChecked():
            self.method = "mosaic"
        elif self.radio_shuffle.isChecked():
            self.method = "shuffle"
        elif self.radio_distortion.isChecked():
            self.method = "distortion"

        self.group_1 = QButtonGroup(self.centralwidget)
        self.group_1.addButton(self.radio_images)
        self.group_1.addButton(self.radio_video)
        self.group_1.removeButton(self.radio_images)
        self.group_1.removeButton(self.radio_video)
        self.radio_images.setChecked(False)
        self.radio_video.setChecked(False)
        self.checkBox.setChecked(False)
        self.label_on_off.setChecked(False)
        self.group_1.addButton(self.radio_images)
        self.group_1.addButton(self.radio_video)

        self.score_threshold_label.show()
        self.score_threshold.show()
        self.score_threshold_value.show()
        self.deid_level_label.show()
        self.deid_level.show()
        self.deid_level_value.show()
        self.pedestrain_class.show()
        self.vehicle_class.show()
        self.face_class.show()
        self.images_or_video.show()
        self.radio_images.show()
        self.radio_video.show()

        self.topk_label.hide()
        self.topk.hide()
        self.checkBox.hide()
        self.label_on_off.hide()
        self.upload_images.hide()
        self.upload_video_button.hide()
        self.video_multiframe_label.hide()
        self.video_multiframe.hide()
        self.segmentation.hide()
        self.detection.show()
        print("detection on")

        # if self.face_class.isChecked():
        #     self.detection.hide()
        #     self.face_detection.show()
        # elif self.pedestrain_class.isChecked() or self.vehicle_class.isChecked():
        #     self.detection.show()
        #     self.face_detection.hide()

    def radio_images_clicked(self):

        self.checkBox.show()
        self.label_on_off.show()
        self.upload_images.show()
        self.upload_video_button.hide()
        self.video_multiframe_label.hide()
        self.video_multiframe.hide()

        self.video = ''
        self.input_image = ''

    def radio_video_clicked(self):

        self.checkBox.hide()
        self.upload_images.hide()
        self.label_on_off.show()
        self.upload_video_button.show()
        self.video_multiframe_label.show()
        self.video_multiframe.show()

        self.video = ''
        self.input_image = ''

    def value_changed(self, value):
        value = value / 100
        if value == 0:
            value = 0.01
        if value > 0.99:
            value = 0.99
        self.score_threshold_value.setText(str(value))

    def level_value_changed(self, value):
        self.deid_level_value.setText(str(value))

    def check_box_changed(self, check_box):
        if check_box is True:
            self.vehicle_class.setEnabled(False)
            self.vehicle_class.setChecked(False)
            self.pedestrain_class.setEnabled(False)
            self.pedestrain_class.setChecked(False)
            self.detection.hide()
            self.face_detection.show()
            print("detection off")
        elif check_box is False:
            self.vehicle_class.setEnabled(True)
            self.vehicle_class.setChecked(True)
            self.pedestrain_class.setEnabled(True)
            self.pedestrain_class.setChecked(True)
            self.detection.show()
            self.face_detection.hide()
            print("detection on")

    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(360, 400)
        mainWindow.setMinimumSize(360, 400)
        mainWindow.setWindowTitle("Demo")
        # mainWindow.setWindowIcon(QIcon('ci.jpg'))

        self.centralwidget = QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # select model
        self.title = QLabel("De-identification Application", self.centralwidget)
        self.title.setGeometry(60, 20, 400, 20)
        self.title.setFont(QFont('Arial', 14))
        self.title.setObjectName("title")

        # select method
        self.de_identification_methods = QLabel("Methods :", self.centralwidget)
        self.de_identification_methods.setGeometry(30, 70, 120, 20)
        self.de_identification_methods.setObjectName("de_identification_methods")

        self.radio_masking = QRadioButton("Masking", self.centralwidget)
        self.radio_masking.setGeometry(100, 70, 70, 20)
        self.radio_masking.setObjectName("radio_masking")
        self.radio_masking.toggled.connect(self.radio_masking_clicked)

        self.radio_blurring = QRadioButton("Blurring", self.centralwidget)
        self.radio_blurring.setGeometry(180, 70, 70, 20)
        self.radio_blurring.setObjectName("radio_blurring")
        self.radio_blurring.toggled.connect(self.radio_detection_clicked)

        self.radio_mosaic = QRadioButton("Mosaic", self.centralwidget)
        self.radio_mosaic.setGeometry(260, 70, 70, 20)
        self.radio_mosaic.setObjectName("radio_mosaic")
        self.radio_mosaic.toggled.connect(self.radio_detection_clicked)

        self.radio_shuffle = QRadioButton("Shuffle", self.centralwidget)
        self.radio_shuffle.setGeometry(100, 100, 100, 20)
        self.radio_shuffle.setObjectName("radio_shuffle")
        self.radio_shuffle.toggled.connect(self.radio_detection_clicked)

        self.radio_distortion = QRadioButton("Distortion", self.centralwidget)
        self.radio_distortion.setGeometry(180, 100, 100, 20)
        self.radio_distortion.setObjectName("radio_distortion")
        self.radio_distortion.toggled.connect(self.radio_detection_clicked)

        self.score_threshold_value = QLabel(self.centralwidget)
        self.score_threshold_value.setGeometry(260, 130, 50, 20)
        self.score_threshold_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.score_threshold_value.setText('0.3')
        self.score_threshold_value.hide()

        self.score_threshold_label = QLabel("Score threshold :", self.centralwidget)
        self.score_threshold_label.setGeometry(30, 130, 120, 20)
        self.score_threshold_label.setObjectName("score_threshold_label")
        self.score_threshold_label.hide()

        self.score_threshold = QSlider(Qt.Horizontal, self.centralwidget)
        self.score_threshold.setGeometry(160, 130, 120, 20)
        self.score_threshold.setRange(0, 100)
        self.score_threshold.setValue(30)
        self.score_threshold.setTickInterval(1)
        self.score_threshold.valueChanged.connect(self.value_changed)
        self.score_threshold.hide()

        self.deid_level_value = QLabel(self.centralwidget)
        self.deid_level_value.setGeometry(260, 160, 50, 20)
        self.deid_level_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.deid_level_value.setText('1')
        self.deid_level_value.hide()

        self.deid_level_label = QLabel("Deid level :", self.centralwidget)
        self.deid_level_label.setGeometry(30, 160, 120, 20)
        self.deid_level_label.setObjectName("deid_level_label")
        self.deid_level_label.hide()

        self.deid_level = QSlider(Qt.Horizontal, self.centralwidget)
        self.deid_level.setGeometry(160, 160, 120, 20)
        self.deid_level.setRange(1, 3)
        self.deid_level.setValue(1)
        self.deid_level.setTickInterval(1)
        self.deid_level.setTickPosition(QSlider.TicksBothSides)
        self.deid_level.valueChanged.connect(self.level_value_changed)
        self.deid_level.hide()

        self.pedestrain_class = QCheckBox("Pedestrian", self.centralwidget)
        self.pedestrain_class.setGeometry(30, 190, 100, 20)
        self.pedestrain_class.setObjectName("pedestrain_class")
        self.pedestrain_class.setChecked(True)    
        self.pedestrain_class.hide()

        self.vehicle_class = QCheckBox("Vehicle", self.centralwidget)
        self.vehicle_class.setGeometry(130, 190, 100, 20)
        self.vehicle_class.setObjectName("vehicle_class")
        self.vehicle_class.setChecked(True)
        self.vehicle_class.hide()

        self.face_class = QCheckBox("Face", self.centralwidget)
        self.face_class.setGeometry(230, 190, 100, 20)
        self.face_class.setObjectName("face_class")
        self.face_class.setChecked(False)
        self.face_class.clicked.connect(self.check_box_changed)
        self.face_class.hide()

        self.topk_label = QLabel("Top k :", self.centralwidget)
        self.topk_label.setGeometry(210, 190, 50, 20)
        self.topk_label.setObjectName("topk_label")
        self.topk_label.hide()

        self.topk = QLineEdit(self.centralwidget)
        self.topk.setObjectName("topk")
        self.topk.setGeometry(260, 190, 50, 20)
        self.topk.setValidator(QIntValidator())
        self.topk.setText("15")
        self.topk.hide()

        # images or video
        self.images_or_video = QLabel("Type of input :", self.centralwidget)
        self.images_or_video.setGeometry(30, 220, 90, 20)
        self.images_or_video.setObjectName("images_or_video")
        self.images_or_video.hide()

        self.radio_images = QRadioButton("Images", self.centralwidget)
        self.radio_images.setGeometry(150, 220, 70, 20)
        self.radio_images.setObjectName("radio_images")
        self.radio_images.toggled.connect(self.radio_images_clicked)
        self.radio_images.hide()

        self.radio_video = QRadioButton("Video", self.centralwidget)
        self.radio_video.setGeometry(240, 220, 70, 20)
        self.radio_video.setObjectName("radio_video")
        self.radio_video.toggled.connect(self.radio_video_clicked)
        self.radio_video.hide()

        self.upload_images = QPushButton("Upload images", self.centralwidget)
        self.upload_images.setGeometry(30, 250, 300, 20)
        self.upload_images.setObjectName("upload_images")
        self.upload_images.clicked.connect(self.choose_images)
        self.upload_images.hide()
        
        self.upload_video_button = QPushButton("Upload video", self.centralwidget)
        self.upload_video_button.setGeometry(30, 250, 300, 20)
        self.upload_video_button.setObjectName("upload_video")
        self.upload_video_button.clicked.connect(self.choose_video)
        self.upload_video_button.hide()

        self.checkBox = QCheckBox("Single image", self.centralwidget)
        self.checkBox.setGeometry(30, 280, 120, 20)
        self.checkBox.setObjectName("checkBox")
        self.checkBox.hide()

        self.video_multiframe_label = QLabel("Video multiframe:", self.centralwidget)
        self.video_multiframe_label.setGeometry(30, 280, 120, 20)
        self.video_multiframe_label.setObjectName("video_multiframe_label")
        self.video_multiframe_label.hide()

        self.video_multiframe = QLineEdit(self.centralwidget)
        self.video_multiframe.setObjectName("video_multiframe")
        self.video_multiframe.setGeometry(150, 280, 50, 20)
        self.video_multiframe.setValidator(QIntValidator())
        self.video_multiframe.setText("10")
        self.video_multiframe.hide()

        self.label_on_off = QCheckBox("Label On/Off", self.centralwidget)
        self.label_on_off.setGeometry(230, 280, 120, 20)
        self.label_on_off.setObjectName("label_on_off")
        self.label_on_off.hide()

        # run segmentation module
        self.segmentation = QPushButton("Start", self.centralwidget)
        self.segmentation.setGeometry(30, 310, 300, 20)
        self.segmentation.setObjectName("segmentation")
        self.segmentation.clicked.connect(self.start_segmentation)

        # run detection module
        self.detection = QPushButton("Start", self.centralwidget)
        self.detection.setGeometry(30, 310, 300, 20)
        self.detection.setObjectName("detection")
        self.detection.clicked.connect(self.start_detection)

        # run face detection module
        self.face_detection = QPushButton("Start", self.centralwidget)
        self.face_detection.setGeometry(30, 310, 300, 20)
        self.face_detection.setObjectName("face_detection")
        self.face_detection.clicked.connect(self.start_face_detection)

        # visualization
        self.visualization = QPushButton("Play result", self.centralwidget)
        self.visualization.setGeometry(30, 340, 300, 20)
        self.visualization.setObjectName("Play result")
        self.visualization.clicked.connect(self.see_film)

        # self.pbar = QProgressBar(self.centralwidget)
        # self.pbar.setGeometry(30, 370, 330, 20)

        # menu bar
        self.menubar = QMenuBar(mainWindow)
        self.menubar.setGeometry(0, 0, 711, 21)
        self.menubar.setObjectName("menubar")

        # menu help
        self.menuHelp = QMenu("Help", self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menubar.addAction(self.menuHelp.menuAction())
        mainWindow.setMenuBar(self.menubar)

        # status bar
        self.statusbar = QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        mainWindow.setCentralWidget(self.centralwidget)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_Deidentification()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
