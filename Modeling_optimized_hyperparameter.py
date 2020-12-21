# -*- coding: utf-8 -*-
"""
Modeling_desktop.py

1. 기존 모델 로딩
2. Fine Tuning
3. Model Training
4. Model Evaluation
5. Save
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np

#### HyperParameters ####
dnn_layers = 2
optimizer = 'adam'
lr = 0.1
batch_size = 20
steps_per_epoch = 40
epochs = 100
validation_steps = 20
#########################

###########
# Dataset  
###########

# new_data 이용
#train_dir = "C:/Users/Soo/Desktop/Project_Backups/dataset_new/train"
#val_dir = "C:/Users/Soo/Desktop/Project_Backups/dataset_new/test"

# original_data 이용
train_dir = "C:/Users/Soo/Desktop/Project_Backups/dataset_original/train"
val_dir = "C:/Users/Soo/Desktop/Project_Backups/dataset_original/test"

# original_data_six 이용
#train_dir = "C:/Users/Soo/Desktop/Project_Backups/dataset_original_six/train"
#val_dir = "C:/Users/Soo/Desktop/Project_Backups/dataset_original_six/test"

# validation만 new_data 이용
#val_dir = "C:/Users/Soo/Desktop/Project_Backups/validation"

## 3) Preparing Dataset using ImageGenerator

### (1) Training data : 이미지 증식 -> overfitting solution
train_data = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 20, # image 회전 각도 범위(+, - 범위)
        width_shift_range = 0.3, # image 수평 이동 범위
        height_shift_range = 0.3, # image 수직 이동 범위  
        shear_range = 0.2, # image 전단 각도 범위
        zoom_range=0.3, # image 확대 범위
        horizontal_flip=True # image 수평 뒤집기 범위 
        ) 

train_generator = train_data.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size, 
        class_mode='categorical',
        color_mode='grayscale')

### (2)) Validation data : 이미지 증식 사용 안 함
val_data = ImageDataGenerator(rescale=1./255)

validation_generator = val_data.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale')

## 4) Model Training

# Setting EarlyStopping
callback = EarlyStopping(monitor='val_loss', patience=4)


# 10번 반복 학습

max_list = []

for num in range(5):
    
    input_shape=(48, 48, 1)

    ## 1) 모델 생성
    model = Sequential()
    
    ## 2) Layer 추가
    
    ### (4) emotion_model.hdf5
    model_original = load_model('C:/Users/Soo/Desktop/Project_Backups/emotion_model.hdf5', compile=False)
    model_original.trainable = False

    model.add(Sequential(model_original))
    
    if dnn_layers == 2 :
        ### (5) Flatten Layer
        model.add(Flatten())
        
        ### (6) Fully Connected Layer_01
        model.add(Dense(14, activation = 'relu'))
        model.add(Dropout(0.2))
        
        ### (9) Fully Connected Layer_04 - output layer
        model.add(Dense(7, activation = 'softmax'))
    
        
    elif dnn_layers == 4 :
        ### (5) Flatten Layer
        model.add(Flatten())
        
        ### (6) Fully Connected Layer_01
        model.add(Dense(14, activation = 'relu'))
        model.add(Dropout(0.2))
        
        ### (6) Fully Connected Layer_01
        model.add(Dense(28, activation = 'relu'))
        model.add(Dropout(0.2))
        
        ### (6) Fully Connected Layer_01
        model.add(Dense(14, activation = 'relu'))
        model.add(Dropout(0.2))
        
        ### (9) Fully Connected Layer_04 - output layer
        model.add(Dense(7, activation = 'softmax'))
    

    # Model Compile
    model.compile(optimizer=optimizer,
                  lr=lr,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Training
    model_fit = model.fit_generator(
              train_generator, 
              steps_per_epoch=steps_per_epoch, 
              epochs=epochs, 
              validation_data=validation_generator,
              validation_steps=validation_steps,
              callbacks=[callback])
    
    
    val_loss_list = model_fit.history['val_accuracy']
    max_list.append(max(val_loss_list))
    
print('Model Accuracy :', max(max_list))     

######################
# 4. Model Evaluation - history 기능 이용
######################

# 1) loss, acc, val_loss, val_acc 변수 할당
loss = model_fit.history['loss'] # train
acc = model_fit.history['accuracy']
val_loss = model_fit.history['val_loss'] # validation
val_acc = model_fit.history['val_accuracy']

epochs = range(1, len(acc) + 1)

# 2) Model History Graph : acc vs val_acc   
plt.plot(epochs, acc, 'b', label='Training Acc.')
plt.plot(epochs, val_acc, 'r', label='Validation Acc.')
plt.title('Training vs Validation Accuracy')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

# Model History Graph : loss vs val_loss 
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()


# test
test_img = io.imread('C:/Users/Soo/Desktop/Project_Backups/for_test_happy.jpg')
test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) / 255.
test_img_gray_reshaped = test_img_gray.reshape(1, 48, 48, 1)
model.predict(test_img_gray_reshaped)


########################################
# 5. Save Model - HDF5 형식 
########################################

model.save('C:/Users/Soo/Desktop/Project_Backups/new_model')
print('model saved...')












































































































































































