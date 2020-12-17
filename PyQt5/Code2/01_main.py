# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 00:14:35 2020

@author: Swan
"""
import sys
from PyQt5.QtWidgets import *
from video import *
 
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
 
class CWidget(QWidget):
 
    def __init__(self):
        super().__init__()    
        size = QSize(600,500)
        self.initUI(size)
        self.video = video(self, QSize(self.frm.width(), self.frm.height()))
 
    def initUI(self, size):
        vbox = QVBoxLayout()        
        # cam on, off button
        self.btn = QPushButton('start cam', self)
        self.btn.setCheckable(True)
        self.btn.clicked.connect(self.onoffCam)
        vbox.addWidget(self.btn)
 
        # kind of detection
        txt = ['full body', 'upper body', 'lower body', 'face', 'eye', 'eye glass', 'smile']       
        self.grp = QButtonGroup(self)
        self.grp = QButtonGroup(self)
        for i in range(len(txt)):
            btn = QCheckBox(txt[i], self)
            self.grp.addButton(btn, i)
            vbox.addWidget(btn)   
        vbox.addStretch(1)
        self.grp.setExclusive(False)
        self.grp.buttonClicked[int].connect(self.detectOption)
        self.bDetect = [False for i in range(len(txt))]
                 
        # video area
        self.frm = QLabel(self)     
        self.frm.setFrameShape(QFrame.Panel)
         
        hbox = QHBoxLayout()
        hbox.addLayout(vbox)       
        hbox.addWidget(self.frm, 1)        
        self.setLayout(hbox)
        
        self.setFixedSize(size)
        self.move(100,100)
        self.setWindowTitle('OpenCV + PyQt5')
        self.show()
 
    def onoffCam(self, e):
        if self.btn.isChecked():
            self.btn.setText('stop cam')
            self.video.startCam()
        else:
            self.btn.setText('start cam')
            self.video.stopCam()            
 
    def detectOption(self, id):
        if self.grp.button(id).isChecked():
            self.bDetect[id] = True
        else:
            self.bDetect[id] = False
        #print(self.bDetect)
        self.video.setOption(self.bDetect)
 
    def recvImage(self, img):        
        self.frm.setPixmap(QPixmap.fromImage(img))
 
    def closeEvent(self, e):
        self.video.stopCam()  
 
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = CWidget()
    sys.exit(app.exec_())

 
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
 
class CWidget(QWidget):
 
    def __init__(self):
        super().__init__()    
        size = QSize(600,500)
        self.initUI(size)
        self.video = video(self, QSize(self.frm.width(), self.frm.height()))
 
    def initUI(self, size):
        vbox = QVBoxLayout()        
        # cam on, off button
        self.btn = QPushButton('start cam', self)
        self.btn.setCheckable(True)
        self.btn.clicked.connect(self.onoffCam)
        vbox.addWidget(self.btn)
 
        # kind of detection
        txt = ['full body', 'upper body', 'lower body', 'face', 'eye', 'eye glass', 'smile']       
        self.grp = QButtonGroup(self)
        self.grp = QButtonGroup(self)
        for i in range(len(txt)):
            btn = QCheckBox(txt[i], self)
            self.grp.addButton(btn, i)
            vbox.addWidget(btn)   
        vbox.addStretch(1)
        self.grp.setExclusive(False)
        self.grp.buttonClicked[int].connect(self.detectOption)
        self.bDetect = [False for i in range(len(txt))]
                 
        # video area
        self.frm = QLabel(self)     
        self.frm.setFrameShape(QFrame.Panel)
         
        hbox = QHBoxLayout()
        hbox.addLayout(vbox)       
        hbox.addWidget(self.frm, 1)        
        self.setLayout(hbox)
        
        self.setFixedSize(size)
        self.move(100,100)
        self.setWindowTitle('OpenCV + PyQt5')
        self.show()
 
    def onoffCam(self, e):
        if self.btn.isChecked():
            self.btn.setText('stop cam')
            self.video.startCam()
        else:
            self.btn.setText('start cam')
            self.video.stopCam()            
 
    def detectOption(self, id):
        if self.grp.button(id).isChecked():
            self.bDetect[id] = True
        else:
            self.bDetect[id] = False
        #print(self.bDetect)
        self.video.setOption(self.bDetect)
 
    def recvImage(self, img):        
        self.frm.setPixmap(QPixmap.fromImage(img))
 
    def closeEvent(self, e):
        self.video.stopCam()  
 
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = CWidget()
    sys.exit(app.exec_())