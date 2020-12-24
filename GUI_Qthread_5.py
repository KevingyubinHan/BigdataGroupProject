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

        hbox = QHBoxLayout()

        self.image_label = QLabel(self)  # cam->image 들어갈 label 할당
        self.image_label.resize(self.display_width, self.display_height)
        hbox.addWidget(self.image_label)

        self.preds_label = QLabel(self)
        hbox.addWidget(self.preds_label)

        self.setLayout(hbox)

        self.thread = VideoThread() # thread
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_preds_signal.connect(self.update_preds)
        self.thread.start()  # auto start

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def update_preds(self, cv_canvas):
        qt_img = self.convert_cv_qt(cv_canvas) # 수정 필요
        self.preds_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        converted_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = converted_img.scaled(640, 480, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)

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
