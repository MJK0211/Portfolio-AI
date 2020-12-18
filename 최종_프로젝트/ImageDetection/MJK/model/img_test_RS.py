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
    class_mode='categorical' #클래스모드는 찾아보기!  
    # save_to_dir='./data/img/data1_2/train' #전환된 이미지 데이터 파일을 이미지 파일로 저장
) 
xy_test = test_datagen.flow_from_directory(
   './MJK/data/test',
    target_size=(200,200),
    batch_size=1,  
    class_mode='categorical'   
    # save_to_dir='./data/img/data1_2/test'
)

x_train = list()
y_train = list()
x_test = list()
y_test = list()

for i in range(len(xy_train)):
    x_train.append(xy_train[i][0])    
    y_train.append(xy_train[i][1])

for j in range(len(xy_test)):
    x_test.append(xy_test[j][0])
    y_test.append(xy_test[j][1])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# (1095, 1, 200, 200, 3)
# (1095, 1, 4)
# (260, 1, 200, 200, 3)
# (260, 1, 4)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[3], x_train.shape[4])
y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[3], x_test.shape[4])
y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])

predict = pred_datagen.flow_from_directory(
    './MJK/data/pred', 
    target_size=(200,200),
    batch_size=55,  
    class_mode=None,
    shuffle=False,
)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization, Input

#2. 모델 구성
def build_model(optimizer='adam', lr=0.001):
    inputs = Input(shape=(200, 200, 3), name='input')
    x = Conv2D(64, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (6, 6), padding="same", activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x) 
    outputs = Dense(4, activation='softmax', name='outputs')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer(lr=lr), metrics=['acc'], loss='categorical_crossentropy')

    return model

from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

def create_hyperparameter():
    # batches = [10, 20, 30, 40, 50]
    optimizers = [Adam, Adadelta, RMSprop, Adamax, Adagrad, SGD, Nadam]
    lr=[0.1, 0.01]
    return {"optimizer" : optimizers, "lr" : lr }

hyperparameters = create_hyperparameter()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier #keras를 sklearn으로 쌓겠다. 
model = KerasClassifier(build_fn=build_model, verbose=1) #케라스 모델을 맵핑

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
model = GridSearchCV(model, hyperparameters)

model.fit(x_train, y_train, batch_size=64, verbose=1)
print(model.best_params_)

acc = model.score(x_test, y_test)
print("acc : ", acc)

# {'lr': 0.01, 'optimizer': <class 'tensorflow.python.keras.optimizer_v2.adagrad.Adagrad'>}
# 9/9 [==============================] - 0s 21ms/step - loss: 1.3690 - acc: 0.3077


'''
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=5,
    factor=0.5,
    verbose=1
)

#3. 컴파일, 훈련
# modelpath = "D:/ImgDetection/MJK/data/weight/cp-rmsprop-{epoch:02d}-{val_loss:4f}.hdf5"  

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=5,
    factor=0.5,
    verbose=1
)

early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)
# check_point = ModelCheckpoint(
#     filepath = modelpath,
#     # save_weights_only=True,
#     save_best_only=True,
#     monitor='val_loss',
#     verbose=1
# )

hist = model.fit_generator(
    xy_train, #train 안에 x, y의 데이터를 모두 가지고 있다
    steps_per_epoch=len(xy_train),
    validation_data=(xy_test),
    validation_steps=len(xy_test),
    epochs=1,
    verbose=1,
    callbacks=[reduce_lr, early_stopping]
)
#4. 평가, 예측
loss, acc = model.evaluate(xy_test)
print("loss : ", loss)
print("acc : ", acc)

# y_pred = model.predict(predict)
# print("y_pred : ", y_pred)

'''


# import matplotlib.pyplot as plt

# fig = plt.figure()
# rows = 5
# cols = 11

# # print(y_pred[1][0])
# a = ['BAE', 'NAM', "HA", "KANG"]

# def printIndex(array, i):
#     if np.argmax(y_pred[i]) == 0:
#         return a[0]
#     elif np.argmax(y_pred[i]) == 1:
#         return a[1]
#     elif np.argmax(y_pred[i]) == 2:
#         return a[2]
#     elif np.argmax(y_pred[i]) == 3:
#         return a[3]

# for i in range(len(predict[0])):
#     ax = fig.add_subplot(rows, cols, i+1)
#     ax.imshow(predict[0][i])
#     label = printIndex(y_pred, i)
#     print(label)
#     ax.set_xlabel(label)
#     ax.set_xticks([]), ax.set_yticks([])
# plt.show()

# loss :  0.16894356906414032
# acc :  0.918749988079071


# adam
# loss :  0.6471794843673706
# acc :  0.7384615540504456


# rmsprop
# loss :  0.4521673619747162
# acc :  0.8423076868057251
