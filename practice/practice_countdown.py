# practice_countdown.py

# https://www.geeksforgeeks.org/timer-application-using-pyqt5/

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QInputDialog
from PyQt5.QtCore import QTimer, QTime


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.count = 0
        self.flag = False

        button = QPushButton("Set time(s)", self)
        button.clicked.connect(self.get_seconds)

        self.label = QLabel("//TIMER//", self)

        start_button = QPushButton("Start", self)
        start_button.clicked.connect(self.start_action)

        pause_button = QPushButton("Pause", self)
        pause_button.clicked.connect(self.pause_action)

        reset_button = QPushButton("Reset", self)
        reset_button.clicked.connect(self.reset_action)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.showTime)

        vbox = QVBoxLayout()
        vbox.addWidget(button)
        vbox.addWidget(self.label)
        vbox.addWidget(start_button)
        vbox.addWidget(pause_button)
        vbox.addWidget(reset_button)
        self.setLayout(vbox)

    def showTime(self):
        if self.flag:
            self.count = self.count.addSecs(-1)

            text = self.count.toString('hh:mm:ss')
            self.label.setText(text)

            if self.count == QTime(0,0,0):
                self.flag = False
                self.label.setText("countdown")

    def get_seconds(self):
        self.flag = False

        seconds, done = QInputDialog.getInt(self, 'Countdown', '초를 입력하세요.', min=1)

        if done:
            self.count = QTime(0, 0, seconds)
            text = self.count.toString('hh:mm:ss')
            self.label.setText(text)

    def start_action(self):
        self.flag = True
        self.timer.start(1000)

        if self.count == QTime(0,0,0):
            self.flag = False

    def pause_action(self):
        self.flag = False

    def reset_action(self):
        self.start = False
        self.count = QTime(0,0,0)
        self.label.setText("//TIMER//")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Widget()
    window.show()
    sys.exit(app.exec())