import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer, QTime, Qt

# http://www.gisdeveloper.co.kr/?p=8345#comment-29388
# https://learndataanalysis.org/how-to-create-a-digital-clock-with-pyqt5-pyqt5-tutorial/

class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(250, 150)

        layout = QVBoxLayout()

        fnt = QFont('Open Sans', 120, QFont.Bold)

        self.lbl = QLabel()
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setFont(fnt)
        layout.addWidget(self.lbl)

        self.btnStart = QPushButton("시작")
        self.btnStart.clicked.connect(self.onStartButtonClicked)

        self.btnStop = QPushButton("멈춤")
        self.btnStop.clicked.connect(self.onStopButtonClicked)

        layout.addWidget(self.btnStart)
        layout.addWidget(self.btnStop)

        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.showTime)

        self.zero_time = QTime(0, 0, 0)


    def showTime(self):
        self.zero_time = self.zero_time.addSecs(1)
        displayTxt = self.zero_time.toString('hh:mm:ss')
        print(displayTxt)

        self.lbl.setText(displayTxt)

    def onStartButtonClicked(self):
        self.timer.start(1000)
        self.btnStop.setEnabled(True)
        self.btnStart.setEnabled(False)

    def onStopButtonClicked(self):
        self.timer.stop()
        self.btnStop.setEnabled(False)
        self.btnStart.setEnabled(True)


app = QApplication(sys.argv)

demo = AppDemo()
demo.show()

app.exit(app.exec_())