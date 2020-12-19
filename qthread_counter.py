#qthread_counter.py

#https://opentutorials.org/module/544/18723
#https://riptutorial.com/ko/pyqt5/example/29500/%EA%B8%B0%EB%B3%B8-pyqt-%EC%A7%84%ED%96%89%EB%A5%A0-%EB%A7%89%EB%8C%80

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class Worker(QThread):
    sig_cnt = pyqtSignal(int)
    running = True

    def run(self):
        count = 0

        while self.running:
            count += 1
            self.sig_cnt.emit(count)
            self.sleep(1)

    def stop(self):
        self.running = False
        self.quit()


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.btn_toggle = QPushButton(self)
        self.btn_toggle.setText('counter start/stop')
        self.btn_toggle.clicked.connect(self.onButtonClick)

        self.lbl_num = QLabel('count', self)

        self.th = Worker()

        vbox = QVBoxLayout()
        vbox.addWidget(self.btn_toggle)
        vbox.addWidget(self.lbl_num)
        self.setLayout(vbox)

    @pyqtSlot()
    def onButtonClick(self):
        if self.th.isRunning():
            self.th.stop()

            self.btn_toggle.setText('counter start')
            self.lbl_num.setText('num')

        else:
            self.th = Worker()
            self.th.sig_cnt.connect(self.onCountChanged)
            self.th.start()

            self.btn_toggle.setText('counter stop')

    def onCountChanged(self, num):
        self.lbl_num.setText(str(num))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyWidget()
    ex.show()
    sys.exit(app.exec_())