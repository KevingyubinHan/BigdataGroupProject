# -*- coding: utf-8 -*-
"""
OpenCV로 읽어서 PyQt로 보여줌
"""

import cv2
from PyQt5 import QtWidgets
from PyQt5 import QtGui

app = QtWidgets.QApplication([])
label = QtWidgets.QLabel()

img = cv2.imread('tkv.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
h,w,c = img.shape

'''
QtGui.Qlmag 객체 생성시 생성자에 ndarray.data 전달 -> Qimage를 Qpixmap으로 변환
카메라로 캡처한 화면을 실시간으로 처리 가능
-> 이 화면을 실시간으로 보여주려면 반복문 필요-> Python 기본 모듈인 threading 모듈을 가지고 구현해야함
'''
qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
pixmap = QtGui.QPixmap.fromImage(qImg)
label.setPixmap(pixmap)
label.resize(pixmap.width(), pixmap.height())
label.show()

app.exec_()




