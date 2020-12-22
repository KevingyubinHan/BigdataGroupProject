# GUI_Qthread.py

# https://blog.xcoda.net/104
# https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv
# https://blog.naver.com/townpharm/220959370280
# https://opentutorials.org/module/544/18723
# https://riptutorial.com/ko/pyqt5/example/29500/%EA%B8%B0%EB%B3%B8-pyqt-%EC%A7%84%ED%96%89%EB%A5%A0-%EB%A7%89%EB%8C%80


import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QMessageBox, QVBoxLayout, QHBoxLayout, QInputDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTime, QTimer


'''!if not exist "./files" mkdir files
# Download Face detection XML
!curl - L - o. / files / haarcascade_frontalface_default.xml
https: // raw.githubusercontent.com / opencv / opencv / master / data / haarcascades / haarcascade_frontalface_default.xml
'''

# Face detection XML load and trained model loading
face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('files/ai_interview_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surprising", "Neutral"]


# 실시간 camera 영상을 받아오는 thread
class Worker(QThread):
    changePixmap = pyqtSignal(QImage)
    running = True  # thread가 현재 run 중인지 판별하는 변수 선언

    def run(self):
        camera = cv2.VideoCapture(0)  # opencv cam 켜기

        while self.running:
            ret, frame = camera.read()  # Capture image from camera

            if ret:
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # color: RGB
                h, w, c = rgb_img.shape
                qt_img = QImage(rgb_img.data, w, h, w * c, QImage.Format_RGB888)  # Qimage 생성
                p = qt_img.scaled(640, 480, Qt.KeepAspectRatio)  # 화면비 scaling

                self.changePixmap.emit(p)

            else:
                message = QMessageBox.about(self, 'Error', 'Cannot read frame.')
                self.running = False
                break

        camera.release()  # opencv cam 끄기

    def stop(self):
        self.running = False
        self.quit()


# 감정 인식 thread
class Worker2(QThread):
    emotion_value = pyqtSignal(list)
    running = True  # thread가 현재 run 중인지 판별하는 변수 선언

    def run(self):
        camera = cv2.VideoCapture(0)  # opencv cam 켜기

        while self.running:
            ret, frame = camera.read()  # Capture image from camera
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection in frame
            faces = face_detection.detectMultiScale(gray,
                                                    scaleFactor=1.1,
                                                    minNeighbors=5,
                                                    minSize=(30, 30))

            # Perform emotion recognition only when face is detected
            if len(faces) > 0:
                # For the largest image
                face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = face
                # Resize the image to 48x48 for neural network
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Emotion predict
                preds = emotion_classifier.predict(roi)[0]  # 감정 분류 인식 percentage
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]

                self.emotion_value.emit(preds)

            # q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
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
        vbox = QVBoxLayout()  # vertical layout
        hbox = QHBoxLayout()  # horizontal layout
        subvbox_1 = QVBoxLayout()  # vertical layout
        subvbox_2 = QVBoxLayout()  # vertical layout

        self.lbl_cam = QLabel(self)  # cam 영상이 들어갈 label 할당
        self.lbl_cam.resize(640, 480)  # label resize(cam)
        vbox.addWidget(self.lbl_cam)

        self.lbls_emotions = []
        for i in range(len(EMOTIONS)):  # 감정 이름이 들어갈 label 할당
            self.lbls_emotions.append(QLabel())
            self.lbls_emotions[i].setText(EMOTIONS[i])
            subvbox_1.addWidget(self.lbls_emotions[i])

        self.lbls_probs = []
        for i in range(len(EMOTIONS)):  # 감정별 확률이 들어갈 label 할당
            self.lbls_probs.append(QLabel())
            subvbox_2.addWidget(self.lbls_probs[i])

        self.lbl_timer = QLabel('Countdown')  # timer가 들어갈 label 할당
        self.flag = False # timer가 작동중인지 판별할 flag
        self.count = QTime(0, 0, 0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.showTime)
        vbox.addWidget(self.lbl_timer)

        set_countdown = QPushButton("Set Countdown", self)
        set_countdown.clicked.connect(self.get_seconds)
        vbox.addWidget(set_countdown)

        self.btn_toggle = QPushButton(self)  # 시작/일시정지 button 할당
        self.btn_toggle.setText('Start your AI interview')
        self.btn_toggle.clicked.connect(self.onButtonClick)  # start/pause button
        vbox.addWidget(self.btn_toggle)

        hbox.addLayout(subvbox_1)
        hbox.addLayout(subvbox_2)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        self.th = Worker()  # 실시간 cam 영상 받아오는 thread
        self.th_emotion = Worker2()  # 감정 인식 thread

    @pyqtSlot()
    def onButtonClick(self):
        if self.th.isRunning():
            self.th.stop()
            self.th_emotion.stop()
            self.flag = False
            #self.timer.stop()
            self.btn_toggle.setText('Start your AI interview')

        else:
            self.th = Worker()
            self.th.changePixmap.connect(self.set_image)
            self.th.start()

            self.th_emotion = Worker2()
            self.th_emotion.emotion_value.connect(self.set_lbls_probs)
            self.th_emotion.start()

            self.flag = True
            self.timer.start(1000)  # every seconds

            if self.count == QTime(0, 0, 0):
                self.flag = False

            self.btn_toggle.setText('Pause')

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.lbl_cam.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(list)
    def set_lbls_probs(self, emotion_value_list):
        for i in range(len(EMOTIONS)):
            self.lbls_probs[i].setText(str(int(emotion_value_list[i] * 100)))

    def get_seconds(self):
        self.flag = False

        seconds, done = QInputDialog.getInt(self, 'Countdown', '초를 입력하세요.', min=1)

        if done:
            self.count = QTime(0, 0, seconds)
            text = self.count.toString('hh:mm:ss')
            self.lbl_timer.setText(text)

    def showTime(self):
        if self.flag:
            self.count = self.count.addSecs(-1)

            text = self.count.toString('hh:mm:ss')
            self.lbl_timer.setText(text)

            if self.count == QTime(0,0,0):
                self.flag = False
                self.lbl_timer.setText("countdown")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
