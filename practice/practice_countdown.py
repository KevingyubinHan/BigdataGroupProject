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

        timer = QTimer(self)
        timer.timeout.connect(self.showTime)
        timer.start(1000)

        vbox = QVBoxLayout()
        vbox.addWidget(button)
        vbox.addWidget(self.label)
        vbox.addWidget(start_button)
        vbox.addWidget(pause_button)
        vbox.addWidget(reset_button)
        self.setLayout(vbox)

    def showTime(self):
        if self.flag:
            self.count -= 1

            if self.count == 0:
                self.flag = False
                self.label.setText("countdown")

        if self.flag:
            text = str(self.count)
            self.label.setText(text)

    def get_seconds(self):
        self.flag = False
        second, done = QInputDialog.getInt(self, 'Seconds', 'Enter Seconds:')

        if done:
            self.count = second
            self.label.setText(str(second))

    def start_action(self):
        self.flag = True

        if self.count == 0:
            self.flag = False

    def pause_action(self):
        self.flag = False

    def reset_action(self):
        self.start = False
        self.count = 0
        self.label.setText("//TIMER//")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Widget()
    window.show()
    sys.exit(app.exec())