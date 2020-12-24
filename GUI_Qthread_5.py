# GUI_Qthread
import sys
import random
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QMessageBox, QVBoxLayout, QHBoxLayout, QInputDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QTime, QTimer

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Face detection XML load and trained model loading
face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('files/ai_interview_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surprising", "Neutral"]

# load interview question examples
question_txt = open('files/InterviewQuestion.txt', 'r', encoding='utf-8')
question_lines = question_txt.readlines()


# Qthread
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_preds_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        camera = cv2.VideoCapture(0)

        while self._run_flag:
            ret, frame = camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detection.detectMultiScale(gray,
                                                    scaleFactor=1.1,
                                                    minNeighbors=5,
                                                    minSize=(30, 30))
            canvas = np.zeros((250, 300, 3), dtype="uint8")

            if len(faces) > 0:
                face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = face

                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Emotion predict
                preds = emotion_classifier.predict(roi)[0]  # array
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]

                cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

                # Label printing
                for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            self.change_pixmap_signal.emit(frame)
            self.change_preds_signal.emit(canvas)

        camera.release()
        cv2.destroyAllWindows()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


# widget
class MyWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('AI interview program')

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        vbox_2 = QVBoxLayout()

        self.image_label = QLabel(self)  # cam->image 들어갈 label 할당
        self.image_label.resize(640, 480)
        vbox.addWidget(self.image_label)

        self.preds_label = QLabel(self) # emotion preds 들어갈 label 할당
        hbox.addWidget(self.preds_label)

        self.btn_question = QPushButton('Set Question')
        self.btn_question.clicked.connect(self.get_question)
        vbox_2.addWidget(self.btn_question)

        self.lbl_question = QLabel('Question')
        vbox_2.addWidget(self.lbl_question)

        self.btn_countdown = QPushButton('Set Countdown')
        self.btn_countdown.setEnabled(False)
        self.btn_countdown.clicked.connect(self.get_seconds)
        vbox_2.addWidget(self.btn_countdown)

        self.lbl_timer = QLabel('Countdown')  # timer가 들어갈 label 할당
        vbox_2.addWidget(self.lbl_timer)

        self.flag = False  # timer가 작동중인지 판별할 flag
        self.count = QTime(0, 0, 0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_time)

        self.btn_start = QPushButton('Start')  # 시작 button 할당
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.click_start)
        vbox_2.addWidget(self.btn_start)

        self.btn_pause = QPushButton('Pause')  # 일시정지 button 할당
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.click_pause)
        vbox_2.addWidget(self.btn_pause)

        hbox.addLayout(vbox_2)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        self.thread = VideoThread() # thread
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_preds_signal.connect(self.update_preds)
        self.thread.start()  # auto start

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img, 640, 480)
        self.image_label.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def update_preds(self, cv_canvas):
        qt_img = self.convert_cv_qt(cv_canvas, 300, 300)
        self.preds_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img, width, height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        converted_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = converted_img.scaled(width, height, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)

    def get_question(self):
        line = random.choice(question_lines)
        self.lbl_question.setText(line)
        self.btn_countdown.setEnabled(True)

    def get_seconds(self):
        self.flag = False

        seconds, done = QInputDialog.getInt(self, 'Countdown', '초를 입력하세요.', min=1)

        if done:
            self.count = QTime(0, 0, seconds)
            text = self.count.toString('hh:mm:ss')
            self.lbl_timer.setText(text)
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)

    def show_time(self):
        if self.flag:
            self.count = self.count.addSecs(-1)

            text = self.count.toString('hh:mm:ss')
            self.lbl_timer.setText(text)

            if self.count == QTime(0, 0, 0):
                self.flag = False
                self.lbl_timer.setText('Countdown')
                self.btn_start.setEnabled(False)
                self.btn_pause.setEnabled(False)

    def click_start(self):
        if not self.flag:
            self.flag = True
            self.timer.start(1000)  # every seconds
            self.btn_start.setEnabled(False)
            self.btn_pause.setEnabled(True)

    def click_pause(self):
        if self.flag:
            self.flag = False
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(False)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Quit', 'Do you want to quit AI interview program?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.thread.stop()
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWidget()
    ex.show()
    sys.exit(app.exec_())
