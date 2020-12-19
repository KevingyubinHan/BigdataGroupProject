# GUI_Qthread.py

# https://blog.xcoda.net/104
# https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv
# https://blog.naver.com/townpharm/220959370280
# https://opentutorials.org/module/544/18723
# https://riptutorial.com/ko/pyqt5/example/29500/%EA%B8%B0%EB%B3%B8-pyqt-%EC%A7%84%ED%96%89%EB%A5%A0-%EB%A7%89%EB%8C%80


import cv2
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QMessageBox, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot


# 기존 cvui 말고 qt의 GUI로 출력
# 감정별 확률 변수: 어떻게 프레임당 변화하는 값을 불러올 수 있을까? thread 이용?


# 실시간 camera 영상을 받아오는 thread
class Worker(QThread):
    changePixmap = pyqtSignal(QImage)
    running = True # run 중인지 판별할 변수

    def run(self):
        camera = cv2.VideoCapture(0)  # opencv cam 켜기

        while self.running:
            ret, frame = camera.read()  # Capture image from camera

            if ret:
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # color: RGB
                h, w, c = rgb_img.shape
                qt_img = QImage(rgb_img.data, w, h, w * c, QImage.Format_RGB888)  # Qimage 생성
                p = qt_img.scaled(640, 480, Qt.KeepAspectRatio) # 화면비 scaling
                self.changePixmap.emit(p)

            else:
                message = QMessageBox.about(self, 'Error', 'Cannot read frame.')
                break

        camera.release()  # opencv cam 끄기

    def stop(self):
        self.running = False
        self.quit()


# window
class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        widget = MyWidget()
        self.setCentralWidget(widget)

        self.setWindowTitle('AI interview program')

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Quit', 'Do you want to quit AI interview?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


# widget UI setting
class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.lbl_cam = QLabel(self)  # cam 영상이 들어갈 label 할당
        self.lbl_cam.resize(640, 480) # label resize(cam)

        self.btn_toggle = QPushButton(self)
        self.btn_toggle.setText('Start your AI interview')
        self.btn_toggle.clicked.connect(self.onButtonClick) # start/pause button

        vbox = QVBoxLayout() # vertical layout
        vbox.addWidget(self.lbl_cam)
        vbox.addWidget(self.btn_toggle)
        self.setLayout(vbox)

        self.th = Worker() # thread

    @pyqtSlot()
    def onButtonClick(self):
        if self.th.isRunning():
            self.th.stop()
            self.btn_toggle.setText('Start your AI interview')

        else:
            self.th = Worker()
            self.th.changePixmap.connect(self.set_image)
            self.th.start()

            self.btn_toggle.setText('Pause')

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.lbl_cam.setPixmap(QPixmap.fromImage(image))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
