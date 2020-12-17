# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:15:17 2020

@author: Swan
"""

def faceDetect():
    eye_detect = False
    face_cascade = cv2.CascadeClassifier("C:/Users/user/jupyter_works/opencv_python/xml/haarcascade_frontface.xml")
    print(face_cascade)
    
    try:
        cap = cv2.VideoCapture(0)
    except:
        print('fail')
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
                        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        color = (0, 0, 0)
        
        for (x, y, w, h) in faces:
       #     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.rectangle(frame,(x,y+int(h/28*10)),(x+w,y+int(h/28*14)),color,-1)
                    
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30)
        if k == ord('i'):
            eye_detect = not eye_detect
        if k == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    faceDetect()
    