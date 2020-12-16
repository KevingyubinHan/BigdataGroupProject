# -*- coding: utf-8 -*-
"""
Modeling.py

1. 기존 모델 로딩
2. Fine Tuning
3. Model Training
4. Model Evaluation
5. Save
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

####################
# 1. 기존 모델 로딩
####################

# 기존 Model Loading : "emotion_model.hdf5" - 경로 주의!
model_original = load_model('C:/Users/Soo/Documents/GitHub/BigdataGroupProject/files/emotion_model.hdf5', compile=False)

# 기존 Model Layer 확인
model_original.summary() 

# 새로운 모델에서 사용될 때 가중치 고정
model_original.trainable = False


#################
# 2. Fine Tuning
#################
input_shape=(48, 48, 1)

## 1) 모델 생성
model = Sequential()

## 2) Layer 추가
'''
### (1) Convolution Layer_01 
model.add(Conv2D(52, kernel_size = (6, 6), input_shape=input_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Dropout(0.3))


### (2) Convolution Layer_02
model.add(Conv2D(50, kernel_size = (6, 6), activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3), strides=(1, 1)))
model.add(Dropout(0.3))

### (3) Batch Normalization
model.add(Flatten())
'''
### (4) emotion_model.hdf5
model.add(Sequential(model_original))

### (5) Flatten Layer
model.add(Flatten())

### (6) Fully Connected Layer_01
model.add(Dense(14, activation = 'relu'))

### (7) Fully Connected Layer_01
model.add(Dense(21, activation = 'relu'))

### (8) Fully Connected Layer_01
model.add(Dense(14, activation = 'relu'))

### (9) Fully Connected Layer_02 - output layer
model.add(Dense(7, activation = 'sigmoid'))

## 3) Model Layer 확인
model.summary()


##############
# 3. Training
##############


## 1) Model Compile
model.compile(optimizer=RMSprop(lr=0.0001),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])


## 2) Train/Val directory
train_dir = "C:/Users/Soo/Documents/GitHub/BigdataGroupProject/train_dir"
val_dir = "C:/Users/Soo/Documents/GitHub/BigdataGroupProject/val_dir"


## 3) Preparing Dataset using ImageGenerator

### (1) Training data : 이미지 증식 -> overfitting solution
train_data = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 40, # image 회전 각도 범위(+, - 범위)
        width_shift_range = 0.2, # image 수평 이동 범위
        height_shift_range = 0.2, # image 수직 이용 범위  
        shear_range = 0.2, # image 전단 각도 범위
        zoom_range=0.2, # image 확대 범위
        horizontal_flip=True,) # image 수평 뒤집기 범위 

train_generator = train_data.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=20, 
        class_mode='categorical')

### (2)) Validation data : 이미지 증식 사용 안 함
val_data = ImageDataGenerator(rescale=1./255)

validation_generator = val_data.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=20,
        class_mode='categorical')

## 4) Model Training
model_fit = model.fit_generator(
          train_generator, 
          steps_per_epoch=100, 
          epochs=30, # 30 epochs()
          validation_data=validation_generator,
          validation_steps=50) 


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
plt.plot(epochs, acc, 'bo', label='Training Acc.')
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


########################################
# 5. Save Model - HDF5 형식 
########################################

model.save('C:/Users/Soo/Documents/GitHub/BigdataGroupProject/files/new_model')
print('model saved...')












































































































































































