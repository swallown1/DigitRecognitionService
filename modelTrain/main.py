# -*- coding: utf-8 -*-#
'''
# Name:         main
# Description:  
# Author:       neu
# Date:         2020/7/30
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense, BatchNormalization
from keras.datasets import mnist
from keras.utils import to_categorical

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def load_data():
    (x_train, y_train), (x_test, y_test)  = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float64')/255
    x_test = x_test.astype('float64')/255

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#画出训练过程中训练和验证的精度与损失
def draw_train_history(history):
    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.savefig('history.png')
    plt.show()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    print(model.summary())
    model.save('model.h5')
    history = model.fit(x_train, y_train,
                        epochs=5,
                        batch_size=64,
                        validation_split=0.3)
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))

    new_model = load_model('model.h5')
    loss, accuracy = new_model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))