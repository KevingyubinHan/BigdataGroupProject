# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:17:28 2020

@author: Swan
"""
#!/usr/bin/env python3
from PyQt5.QtCore import *

time = 0

def printTime() :
  time += 1
  print(time)

timerVar = QTimer()
timerVar.setInterval(1000)
timerVar.timeout.connect(printTime)
timerVar.start()