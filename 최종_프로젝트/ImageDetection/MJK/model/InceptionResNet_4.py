from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, BatchNormalization
import PIL.Image as pilimg
from keras import regularizers
import cv2

# find_namelist = np.load('./Common_data/npy/find_namelist.npy', all)
np.random.seed(33)

# 이미지 생성 옵션 정하기
train_datagen = ImageDataGenerator(     
    rescale=1 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
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
    batch_size=64, 
    class_mode='categorical' #클래스모드는 찾아보기!  
    # save_to_dir='./data/img/data1_2/train' #전환된 이미지 데이터 파일을 이미지 파일로 저장
) 

# x=(150,150,1), train 폴더안에는 ad/normal이 들어있다. y - ad:0, normal:1

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

inceptionResNetV2 = InceptionResNetV2(include_top=False, input_shape=(xy_train[0][0].shape[1], xy_train[0][0].shape[2], xy_train[0][0].shape[3]))
inceptionResNetV2.trainable = False

model = Sequential()
model.add(inceptionResNetV2)
model.add(Flatten())   
model.add(Dense(64, kernel_initializer='he_normal', activation='relu'))                            
model.add(Dense(128, kernel_initializer='he_normal', activation='relu'))                                    
model.add(Dense(4, activation = 'softmax'))

#3. 컴파일, 훈련
# model = load_model('./MJK/data/weight/cp_inc-33-0.115449.hdf5')
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

modelpath = "D:/ImgDetection/MJK/data/weight/cp_inc_rmsprop-{epoch:02d}-{val_loss:4f}.hdf5"  

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics =['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=5,
    factor=0.5,
    verbose=1
)
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1)

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
print("y_pred : ", y_pred)
# print(y_pred.shape)

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

# loss :  0.09911339730024338
# acc :  0.9624999761581421


# rmsprop
# loss :  0.6048264503479004
# acc :  0.75

# adam
# loss :  0.5558212995529175
# acc :  0.7538461685180664

#RMSprop(0.001)
# loss :  0.5510021448135376
# acc :  0.7692307829856873