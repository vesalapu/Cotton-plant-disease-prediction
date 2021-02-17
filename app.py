#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:46:00 2021

@author: rohitvesalapu
"""

from flask import Flask, render_template, request

import numpy as np
import os

from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model


model = load_model("Model/cpdp.h5")

print('@@ Model loaded')

def predict_plant(plant):
    
    test_image = load_img(plant,target_size =(150,150))
    
    test_image = img_to_array(test_image)/255
    
    test_image = np.expand_dims(test_image, axis=0)
    
    result = model.predict(test_image).round(3)
    
    pred = np.argmax(result)
    
    if(pred == 0):
        return "Healthy cotton plant"
    elif(pred == 1):
        return "diseased cotton plant"
    elif (pred == 2):
        return "Healthy cotton plant"
    else:
        return "Healthy cotton plant"
    


app = Flask(__name__)

@app.route("/predict", methods = ['GET','POST'])

def predict():
    
    if(request.method == 'POST'):
        file = request.files['image']
        filename = file.filename
        print("@@ Imput posted = ", filename)
        
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)
        
        print("@@@ Predicting class.....")
        
        pred, output_page = predict_plant(plant = file_path)
        
        return render_template(output_page,pred_output = pred, user_image = file_path)
    
    
if(__name__ == "__main__"):
    app.run(threaded = False)
