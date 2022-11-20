import PIL
import os
# pyright: reportMissingImports=false
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax

# from keras.applications.vgg16 import VGG16


folder =  ['angry', 'embarrassed', 'happy', 'neutral', 'sad']
number = [0,1,2,3,4]

# folder =  ['angry', 'embarrassed', 'happy', 'sad']
# number = [0,1,2,3]


path = "./integ_data"


imgs = []
label = []

size = 256

for (name, labeling) in zip(folder, number):
    data = "%s/%s"%(path,name)
    # path = "./resize_%s/%s"%(size,name)
    count = os.listdir("%s/"%(data))

    for i in count:
        if i.endswith(".png"):
            img = PIL.Image.open("%s/%s"%(data,i))
            Processed_img = img.resize((size,size))
            Processed_img_data = np.array(Processed_img)
            imgs.append(Processed_img_data)
            label.append(labeling)

imgs = np.array(imgs)
label = np.array(label)
idx = np.arange(imgs.shape[0])
np.random.shuffle(idx)

imgs = imgs[idx]
label = label[idx]
print(len(imgs))
print(len(label))
# x_train = imgs
# x_test = imgs
# y_train = label
# y_test = label

num = 400 # 이미지 총 개수
# num = 14056

x_train = imgs[:num]
y_train = label[:num]

x_test = imgs[num:]
y_test = label[num:]

# print(x_test[0])
# print(y_test[0])
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)
# print(x_test)
# print(y_test)

model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=(size, size, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data = (x_train, y_train) ,validation_split = 0.1 )
loss, acc = model.evaluate(x_test, y_test, verbose=1)

print(loss, acc)

# ##데이터 시각화
# fig = plt.figure()
# # plt.subplots_adjust(left=0.1, right=2, top=1.3, bottom=0.1)

# for i in range(9):
#     i += 1
#     # num = '25' + str(i)
#     # num = int(num)
#     # ax = fig.add_subplot(num)
#     plt.title("train_X[{}] / train_Y[{}] / label_number : {}".format(i, i, y_train[i]) )
#     plt.imshow(x_train[i])

#     plt.show()

###### 그래프 출력 ##########
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'],'y',label='train loss')
loss_ax.plot(hist.history['val_loss'],'r',label='val loss')
acc_ax.plot(hist.history['accuracy'],'b',label='train acc')
acc_ax.plot(hist.history['val_accuracy'],'g',label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

model.summary()

model.save('result.h5')

f = open('acc.txt', 'w')

f.write("%lf \t %lf "%(loss, acc))
# 파일 닫기
f.close()


# asdf = np.ravel(x_test, order='C')
# asdf = asdf.reshape(28,28)
# print(asdf)
# import pandas as pd
# df = pd.DataFrame(asdf)
# print(y_test)
# df.to_csv('D:\\gray_mnist_data\\sample.csv', index=False)

# xhat_idx = np.random.choice(x_test.shape[0], pred_count)
# xhat = x_test[xhat_idx]
# # yhat = model.predict_classes(xhat)
# print(xhat)
# print(xhat.shape)
# print(type(xhat))
# y_prob = model.predict(xhat, verbose=0) 
# pred = y_prob.argmax(axis=-1)

# for i in range(pred_count):
#     print('True : ' + str(argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(pred[i]))


import random

pred_count = 10
pred_list = [] ; true_list = []
correct = 0

for i in range(pred_count):
    th = len(y_test)
    th = random.randint(0, th-1)

    xhat = x_test[th]
    xhat = np.array([xhat])
    y_prob = model.predict(xhat, verbose=0) 
    pred = y_prob.argmax(axis=-1)
    true_list.append(str(y_test[th]))
    pred_list.append(str(pred[0]))
    print('True : ' + str(y_test[th]) + ', Predict : ' + str(pred[0]))
    if str(y_test[th]) == str(pred[0]):
        correct = correct + 1

print(true_list)
print(pred_list)
print(correct)

