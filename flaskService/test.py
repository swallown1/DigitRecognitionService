# -*- coding: utf-8 -*-#
'''
# Name:         test
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

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

    new_model = load_model('model.h5')
    loss, accuracy = new_model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))