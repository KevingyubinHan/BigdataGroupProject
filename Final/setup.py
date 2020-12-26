# -*- coding: utf-8 -*-
"""
exe 실행파일 생성

@author: Swan
"""

from distutils.core import setup
import py2exe, sys, os

sys.argv.append('py2exe')

setup(
    options = {'py2exe': {'bundle_files': 1}},
    windows = [{'script': "GUI_Qthread_5.py"}],
    zipfile = None,
)

import sys
print(sys.path)
