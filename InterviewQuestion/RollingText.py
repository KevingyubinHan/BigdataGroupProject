# -*- coding: utf-8 -*-
"""
1. random으로 질문 리스트
2. Pyqt에 보이게(timer 시간과 연동하여)
"""

import random
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt


Question = open('C:/Users/Swan/Documents/GitHub/Text/InterviewQuestion.txt','r',encoding='UTF8')


lines = Question.readlines()
for line in lines:
    print(line)
Question.close()

Q_randomlist = random.choice(lines)



class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        label1 = QLabel(Q_randomlist, self)
        label1.setAlignment(Qt.AlignCenter)
        font1 = label1.font()
        font1.setPointSize(20)
        label1.setFont(font1)
        layout = QVBoxLayout()
        layout.addWidget(label1)
       
        self.setLayout(layout)
        self.setWindowTitle('QLabel')
        self.setGeometry(300, 300, 300, 200)
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())