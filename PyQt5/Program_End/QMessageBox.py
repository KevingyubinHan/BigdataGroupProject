# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:25:23 2020

@author: Swan
https://www.jbmpa.com/pyside2/8
"""
#!/usr/bin/env python3
import sys
from PyQt5.QtWidgets import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 윈도우 설정
        self.setGeometry(300, 300, 400, 300)  # x, y, w, h
        self.setWindowTitle('Status Window')

        # QButton 위젯 생성
        self.button = QPushButton('Restart', self)
        self.button.clicked.connect(self.messagebox_open)
        self.button.setGeometry(10, 10, 200, 50)

        # QDialog 설정
        self.msg = QMessageBox()

    # 버튼 이벤트 함수
    def messagebox_open(self):
        self.msg.setIcon(QMessageBox.Information)
        self.msg.setWindowTitle('AI 면접 Test')
        self.msg.setText('좀 더 미소를 지어 보세요')
        self.msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        retval = self.msg.exec_()

        # 반환값 판단
        print('QMessageBox 리턴값 ', retval)
        if retval == QMessageBox.Ok :
            print('messagebox ok : ', retval)
        elif retval == QMessageBox.Cancel :
            print('messagebox cancel : ', retval)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())