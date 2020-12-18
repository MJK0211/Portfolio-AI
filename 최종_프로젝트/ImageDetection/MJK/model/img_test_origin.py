from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, BatchNormalization
import PIL.Image as pilimg
from keras import regularizers
import cv2
from tensorflow.keras.utils import to_categorical

np.random.seed(33)

# 이미지 생성 옵션 정하기
train_datagen = ImageDataGenerator(     
    rescale=1 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.1, 1.0],
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'    
)

test_datagen = ImageDataGenerator(rescale=1./255)
pred_datagen = ImageDataGenerator(rescale=1./255.)

# 1. 데이터
xy_train = train_datagen.flow_from_directory(
    './MJK/data/train', #폴더 위치 
    target_size=(200,200), #이미지 크기 설정 - 기존 이미지보다 크게 설정하면 늘려준다 
    batch_size=1, 
    class_mode='categorical', #클래스모드는 찾아보기!  
    # save_to_dir='./MJK/data/gen/train' #전환된 이미지 데이터 파일을 이미지 파일로 저장
) 

# # x=(150,150,1), train 폴더안에는 ad/normal이 들어있다. y - ad:0, normal:1

xy_test = test_datagen.flow_from_directory(
   './MJK/data/test',
    target_size=(200,200),
    batch_size=64,  
    class_mode='categorical'   
    # save_to_dir='./data/img/data1_2/test'
)

predict = pred_datagen.flow_from_directory(
    './MJK/data/pred', 
    target_size=(200,200),
    batch_size=55,  
    class_mode=None,
    shuffle=False,
)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(256, (6, 6), padding="same", activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten()) 
model.add(Dense(512, activation='relu'))
model.add(Dense(4, activation='softmax')) 
model.summary()

#3. 컴파일, 훈련
modelpath = "D:/ImgDetection/MJK/data/weight/cp-final-{epoch:02d}-{val_loss:4f}.hdf5"  
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics =['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=5,
    factor=0.5,
    verbose=1
)

early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)

check_point = ModelCheckpoint(
    filepath = modelpath,
    # save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

hist = model.fit_generator(
    xy_train, #train 안에 x, y의 데이터를 모두 가지고 있다
    steps_per_epoch=len(xy_train),
    validation_data=(xy_test),
    validation_steps=len(xy_test),
    epochs=150,
    verbose=1,
    callbacks=[reduce_lr, early_stopping, check_point]
)

#4. 평가, 예측
loss, acc = model.evaluate(xy_test)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(predict)
# print("y_pred : ", y_pred)

import matplotlib.pyplot as plt

fig = plt.figure()
rows = 5
cols = 11

# print(y_pred[1][0])
a = ['BAE', 'NAM', "HA", "KANG"]

def printIndex(array, i):
    if np.argmax(y_pred[i]) == 0:
        return a[0]
    elif np.argmax(y_pred[i]) == 1:
        return a[1]
    elif np.argmax(y_pred[i]) == 2:
        return a[2]
    elif np.argmax(y_pred[i]) == 3:
        return a[3]

for i in range(len(predict[0])):
    ax = fig.add_subplot(rows, cols, i+1)
    ax.imshow(predict[0][i])
    label = printIndex(y_pred, i)
    print(label)
    ax.set_xlabel(label)
    ax.set_xticks([]), ax.set_yticks([])
plt.show()

# loss :  0.16894356906414032
# acc :  0.918749988079071

# adam
# loss :  0.6471794843673706
# acc :  0.7384615540504456

# rmsprop
# loss :  0.4521673619747162
# acc :  0.8523076868057251
