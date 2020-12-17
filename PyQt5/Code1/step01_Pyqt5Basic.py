# -*- coding: utf-8 -*-
"""
Pyqt5를 이용하여 image 출력
"""

from PyQt5 import QtWidgets
from PyQt5 import QtGui

app = QtWidgets.QApplication([])
label = QtWidgets.QLabel()

'''
video가 보이려면
Qpixmqp 객체 생성시 파일 경로 전달 -> 이 부분을 OpenCV에서 읽는 ndarray를 통해 읽게 해줌
이때 QtGui.Qlmage객체 필요 : 객체 생성시 생성자에 ndarray.data 전달 -> Qimage를 Qpixmap으로 변환
'''
pixmap = QtGui.QPixmap('tkv.jpg')
label.setPixmap(pixmap)
label.resize(pixmap.width(), pixmap.height())
label.show()

app.exec_()

