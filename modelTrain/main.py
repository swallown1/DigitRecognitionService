# -*- coding: utf-8 -*-#
'''
# Name:         main
# Description:  
# Author:       neu
# Date:         2020/7/30
'''
from data.data_loader import *
from model.base_model import *


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    print(model.summary())
    history = model.fit(x_train, y_train,
                        epochs=5,
                        batch_size=64,
                        validation_split=0.3)
    model.save('model.h5')
    draw_train_history(history)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss: {}, test accuracy: {}".format(loss, accuracy))