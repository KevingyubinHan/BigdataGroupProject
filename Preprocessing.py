# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:11:11 2020

"""

import dlib # face detection/face landmark
from skimage import io # image read/save
from glob import glob # dir 패턴검색(*jpg)
import cv2
from PIL import Image # image file read
import numpy as np

# 경로 지정
base_dir = 'C:/ITWILL/AI_Interview_App/Data/'
path_raw = base_dir + "Raw" # raw image 위치 
path_cropped = base_dir + "Cropped" # cropped image 저장 위치
path_reshaped = base_dir + "Reshaped"

# 68 landmark 학습 data
predictor = base_dir + "shape_predictor_68_face_landmarks.dat"

# hog 얼굴 인식기(알고리즘)
face_detector = dlib.get_frontal_face_detector()

# 68 landmark 인식기
face_68_landmark = dlib.shape_predictor(predictor)



#for file in glob(fpath+"/*.jpg") : # 한 명 얼굴 인식  
i = 0
for file in glob(path_raw+"/*.jpg") : # [추가] 여러 명 얼굴 인식
    image = io.imread(file)
    print(image.shape) # (928, 650, 3)
    
    # image 출력장 표시 
    win = dlib.image_window()
    win.set_image(image)
        
    #detected = face_detector(image)#, 1)
    detected = face_detector(image, 1) # [추가] 여러 명 얼굴 인식
    print('인식한 face size =', len(detected))
    # 인식한 face size = 1
        
    i += 1
    for face in detected : # 한 명 얼굴 인식
    #for i, face in enumerate(detected) : # person.jpg   
        # 감지된 image 사각점 좌표 
        print(face) # [(141, 171) : 왼쪽 위  (409, 439) : 오른쪽 아래]
        
        print(f'왼쪽 : {face.left()}, 위 : {face.top()}, 오른쪽 : {face.right()}, 아래 : {face.bottom()}')
        # 왼쪽 : 141, 위 : 171, 오른쪽 : 409, 아래 : 439
        
        
        # 이미지 출력장에 face 사각점 좌표 겹치기 
        win.add_overlay(face)
        
        # 이미지 face 사각점안에 68 point 겹치기
        face_landmark = face_68_landmark(image, face)
        win.add_overlay(face_landmark)
        
        # 크롭(crop) : 얼굴 부분만 자르기 : image[h, w]
        crop = image[face.top():face.bottom(), face.left():face.right()]
        
        # croped image save
        io.imsave(path_cropped() + "/cropped"+str(i)+".jpg", crop)


# Data Preprocessing
def imageReshape() : 
    img_h = img_w = 48
    
    image_reshape = [] # 빈 리스트 : image save 
    
    for file in glob(path_cropped + "/*.jpg") :
        img = Image.open(file) 
        
        # image 규격화 
        img = img.resize( (img_h, img_w) )
        # PIL -> numpy
        img_data = np.array(img)
        print(img_data.shape)
        
        image_reshape.append(img_data)
    
    return np.array(image_reshape) # list -> numpy
        
image_reshape = imageReshape()    

print(image_reshape.shape) # (1, 48, 48, 1) 

import matplotlib.pyplot as plt

img = image_reshape[0]
plt.imshow(img)

# Test : 데이터 입력 가능한지 
from tensorflow.keras.models import load_model
from skimage import io
import os

os.chdir("C:/ITWILL/AI_Interview_App/Scripts")

model_original = load_model('files/emotion_model.hdf5', compile=False)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = gray.reshape(1, 48, 48, 1)  # input shape 중요
gray3 = gray2.astype("float") / 255.0

preds = model_original.predict(gray3)
print(preds)
print(preds.max())
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surprising", "Neutral"]
