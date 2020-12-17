# ai_interview_gui_opencv_qt

import numpy as np
import cv2
import sys
import threading # py 기본 thread 모듈
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QMessageBox, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
#from PyQt5 import QtCore


# main.py에서 cam 영상을 처리한 ndarray 변수를 받아와서 기존 cvui 말고 qt의 GUI로 출력
# main.py에서 감정 분류-감정별 확률 변수 import
# main.py의 모듈화?

# https://blog.xcoda.net/104 # opencv와 qt 연동해서 실시간 cam 영상 출력하는 예제


# threading
running = False


class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        widget = MyWidget()
        self.setCentralWidget(widget)

        self.setWindowTitle('AI interview program')
        #self.setGeometry(300,300,400,300) # window 크기 설정
        self.show()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Quit', 'Do you want to quit AI interview?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout()

        self.lbl_cam = QLabel() # cam 영상이 들어갈 label 할당
        btn_start = QPushButton('Start AI interview', self)
        btn_stop = QPushButton('Pause AI interview', self)

        btn_start.clicked.connect(self.start_cam) # cam 시작
        btn_stop.clicked.connect(self.stop_cam) # cam 일시정지

        vbox.addWidget(self.lbl_cam)
        vbox.addWidget(btn_start)
        vbox.addWidget(btn_stop)

        self.setLayout(vbox)


    def run(self):
        global running # global 변수 선언

        camera = cv2.VideoCapture(0) # opencv cam 켜기

        height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.lbl_cam.resize(width, height) # cam lbl 크기 설정

        while running:
            ret, frame = camera.read() # Capture image from camera

            if ret:
                color_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # color: RGB
                h, w, c = color_img.shape
                qimg = QImage(color_img.data, w, h, w * c, QImage.Format_RGB888) # qimage 생성
                pixmap = QPixmap.fromImage(qimg)
                self.lbl_cam.setPixmap(pixmap)

            else:
                message = QMessageBox.about(self, 'Error', 'Cannot read frame.')
                break

        camera.release() # opencv cam 끄기


    def start_cam(self): # cam 시작
        '''global running # global 변수 선언
        running = True
        th = threading.Thread(target=self.run)
        th.start()'''
        pass


    def stop_cam(self): # cam 일시정지
        '''global running # global 변수 선언
        running = False'''
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())