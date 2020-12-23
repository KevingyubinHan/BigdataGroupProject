import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMessageBox, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Face detection XML load and trained model loading
face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('files/ai_interview_model.hdf5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surprising", "Neutral"]


# Qthread
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)

        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


# widget
class MyWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.display_width = 640
        self.display_height = 480

        self.preds = np.array([0,0,0,0,0,0,0])

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        vbox2 = QVBoxLayout()
        vbox3 = QVBoxLayout()

        self.image_label = QLabel(self)  # cam->image 들어갈 label 할당
        self.image_label.resize(self.display_width, self.display_height)
        vbox.addWidget(self.image_label)

        lbls_emotions = []
        for i in range(len(EMOTIONS)):  # 감정 이름이 들어갈 label 할당
            lbls_emotions.append(QLabel())
            lbls_emotions[i].setText(EMOTIONS[i])
            vbox2.addWidget(lbls_emotions[i])

        self.lbls_probs = []
        for i in range(len(EMOTIONS)):  # 감정별 확률이 들어갈 label 할당
            self.lbls_probs.append(QLabel())
            self.lbls_probs[i].setText(str(self.preds[i]))
            vbox3.addWidget(self.lbls_probs[i])

        hbox.addLayout(vbox2)
        hbox.addLayout(vbox3)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        self.thread = VideoThread() # thread
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()  # auto start

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

        emotion_value = self.detect_emotions(cv_img) # preds
        for i in range(len(EMOTIONS)):
            self.lbls_probs[i].setText(str(int(emotion_value[i] * 100)))

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        converted_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = converted_img.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)

    def detect_emotions(self, cv_img):
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray,
                                                scaleFactor=1.1,
                                                minNeighbors=5,
                                                minSize=(30, 30))

        if len(faces) > 0:
            face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = face

            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            self.preds = emotion_classifier.predict(roi)[0]  # array

        return self.preds

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Quit', 'Do you want to quit AI interview?',
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
