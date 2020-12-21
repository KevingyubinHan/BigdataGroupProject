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
from tensorflow.keras.layers import Dropout, Dense, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from skimage import io
import matplotlib.pyplot as plt
import cv2


####################
# 1. 기존 모델 로딩
####################

# 기존 Model Loading : "emotion_model.hdf5" - 경로 주의!
model_original = load_model('C:/Users/Soo/Desktop/Project_Backups/emotion_model.hdf5', compile=False)

# 새로운 모델에서 사용될 때 가중치 고정
model_original.trainable = False


#################
# 2. Fine Tuning
#################

input_shape=(48, 48, 1)

## 1) 모델 생성
model = Sequential()

## 2) Layer 추가

### (1) Original Model : emotion_model.hdf5
model.add(Sequential(model_original))

### (2) Flatten Layer
model.add(Flatten())

### (3) Fully Connected Layer_01
model.add(Dense(14, activation = 'relu'))
model.add(Dropout(0.2))

### (4) Fully Connected Layer_04 - output layer
model.add(Dense(7, activation = 'softmax'))

## 3) Model Layer 확인
model.summary()


##############
# 3. Training
##############

#### HyperParameters ####
lr = 0.1                #
batch_size = 20         #
epochs = 50             #
steps_per_epoch = 40    #
validation_steps = 20   #
#########################

## 1) Model Compile
model.compile(optimizer='adam',
              lr=lr,
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])


## 2) Train/Val directory

train_dir = "C:/Users/Soo/Desktop/Project_Backups/dataset_original/train"
val_dir = "C:/Users/Soo/Desktop/Project_Backups/dataset_original/test"

## 3) Preparing Dataset using ImageGenerator

### (1) Training data : 이미지 증식 -> overfitting solution
train_data = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 20, # image 회전 각도 범위(+, - 범위)
        width_shift_range = 0.3, # image 수평 이동 범위
        height_shift_range = 0.3, # image 수직 이동 범위  
        zoom_range=0.3, # image 확대 범위
        ) 

train_generator = train_data.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size, 
        class_mode='categorical',
        color_mode='grayscale')

### (2) Validation data : 이미지 증식 사용 안 함
val_data = ImageDataGenerator(rescale=1./255)

validation_generator = val_data.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale')

## 4) Model Training
callback = EarlyStopping(monitor='val_loss', patience=4)


# 원하는 성능 나올 때까지 반복 학습
while True :
    model_fit = model.fit_generator(
              train_generator, 
              steps_per_epoch=steps_per_epoch, 
              epochs=epochs, 
              validation_data=validation_generator,
              validation_steps=validation_steps,
              callbacks=[callback])
    val_loss_list = model_fit.history['val_accuracy']
    if max(val_loss_list) >= 0.62 :
        break
    
    else :
        train_data = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 20, # image 회전 각도 범위(+, - 범위)
        width_shift_range = 0.3, # image 수평 이동 범위
        height_shift_range = 0.3, # image 수직 이동 범위  
        zoom_range=0.3, # image 확대 범위
        ) 

        train_generator = train_data.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size, 
        class_mode='categorical',
        color_mode='grayscale')

        ### (2) Validation data : 이미지 증식 사용 안 함
        val_data = ImageDataGenerator(rescale=1./255)

        validation_generator = val_data.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale')
        
        # Model 초기화
        model = Sequential()
        
        model_original = load_model('C:/Users/Soo/Desktop/Project_Backups/emotion_model.hdf5', compile=False)
        model_original.trainable = False
        
        # emotion_model.hdf5
        model.add(Sequential(model_original))
        
        # Flatten Layer
        model.add(Flatten())
        
        # Fully Connected Layer_01
        model.add(Dense(14, activation = 'relu'))
        model.add(Dropout(0.2))
        
        # Fully Connected Layer_02 - output layer
        model.add(Dense(7, activation = 'softmax'))
        
        # Model Compile
        model.compile(optimizer='adam',
              lr=lr,
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
      

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
plt.plot(epochs, loss, 'b', label='Training Loss')
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

model.save('C:/Users/Soo/Desktop/Project_Backups/new_model.hdf5')
print('model saved...')

model_load = load_model('C:/Users/Soo/Desktop/Project_Backups/ai_interview_model.hdf5', compile=False)

model_load.compile(optimizer='adam',
                   lr=lr,
                   loss = 'categorical_crossentropy',
                   metrics=['accuracy'])

model_load.predict(test_img_gray_reshaped)








































































































































































