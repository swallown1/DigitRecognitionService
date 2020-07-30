# -*- coding: utf-8 -*-#
'''
# Name:         main
# Description:  
# Author:       neu
# Date:         2020/7/30
'''
from PIL import Image

import numpy as np

from keras.models import load_model

from flask import Flask, request

model = load_model('model.h5')

def data_process(data):
    data = np.array(data)
    data = data.reshape((1, 28, 28, 1))
    data = data.astype('float64')/255
    return data

app = Flask(__name__)
@app.route('/', methods=['POST', 'GET'])
def predict_data():
    image = request.files.get('file')
    # file_path = image.filename
    # image.save(file_path)

    # read image and convert to gray
    # data = Image.open(file_path).convert('L')
    data = Image.open(image).convert('L')
    data = data.resize((28,28))
    data = np.array(data, 'f')

    data = data_process(data)
    result = model.predict(data)
    print("res", result)
    return str(result.argmax())


if __name__ == '__main__':
    app.run()