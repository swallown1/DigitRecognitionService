# -*- coding: utf-8 -*-#
'''
# Name:         main
# Description:  
# Author:       neu
# Date:         2020/7/30
'''
import numpy as np

from keras.models import load_model
from test import *

from flask import Flask, request

model = load_model('model.h5')

def data_process(data):
    data = np.array(data)
    data = data.reshape((1, 28, 28, 1))
    # data = data.astype('float64')/255
    return data

app = Flask(__name__)
@app.route('/', methods=['POST', 'GET'])
def predict_data():
    # input = request.args.get('input')
    (x_train, y_train), (x_test, y_test) = load_data()
    data, label = generate_data(x_test, y_test)
    data = data_process(data)
    result = model.predict(data)
    print("res", result)
    print(result.argmax())
    print(label)
    return str(result)


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run()