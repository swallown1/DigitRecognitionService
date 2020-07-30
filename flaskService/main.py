# -*- coding: utf-8 -*-#
'''
# Name:         main
# Description:  
# Author:       neu
# Date:         2020/7/30
'''
from PIL import Image
import base64

import numpy as np

from keras.models import load_model

from flask import Flask, request, render_template

model = load_model('model.h5')

def data_process(data):
    data = np.array(data)
    data = data.reshape((1, 28, 28, 1))
    data = data.astype('float64')/255
    return data

app = Flask(__name__)
@app.route('/predict', methods=['POST', 'GET'])
def predict_data():
    t = request.form['file']
    print(t)
    t = t[23:]
    print(t)
    print(t)
    image = base64.b64decode(t)
    path = "test.jpeg"
    file = open(path, 'wb')
    file.write(image)
    file.close()
    # image = request.files.get('file')
    # print(image)
    # file_path = image.filename
    # image.save(file_path)

    # read image and convert to gray
    # data = Image.open(file_path).convert('L')
    data = Image.open(path).convert('L')
    data = data.resize((28,28))
    data = np.array(data, 'f')

    data = data_process(data)
    result = model.predict(data)
    # print("res", result)
    return str(result.argmax())

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run("0.0.0.0", port=5001)