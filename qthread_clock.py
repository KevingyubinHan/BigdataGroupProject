# qthread_clock.py

import sys
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class WorkerClock(QThread):
    sig_clock = pyqtSignal(str)
    running = True

    def run(self):
        while self.running:
            current_time = time.strftime('%X', time.localtime(time.time())) # HH:mm:ss
            self.sig_clock.emit(current_time)
            self.sleep(1)

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
        self.btn_toggle = QPushButton(self)
        self.btn_toggle.setText('clock start')
        self.btn_toggle.clicked.connect(self.onButtonClick)

        self.lbl_clock = QLabel('clock', self) # 현재 시각 label

        self.th = WorkerClock() # thread

        vbox = QVBoxLayout() # vertical layout
        vbox.addWidget(self.btn_toggle)
        vbox.addWidget(self.lbl_clock)
        self.setLayout(vbox)

    @pyqtSlot()
    def onButtonClick(self):
        if self.th.isRunning():
            self.th.stop()

            self.btn_toggle.setText('clock start')
            self.lbl_clock.setText('clock')

        else:
            self.th = WorkerClock()
            self.th.sig_clock.connect(self.update_clock)
            self.th.start()

            self.btn_toggle.setText('clock stop')

    def update_clock(self, clock_time):
        self.lbl_clock.setText(clock_time)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
