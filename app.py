from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES

import sys
import os
import itertools
import random
from PIL import Image  # PIL
from svmutil import *  # libSVM
import cv2
import numpy as np

# Image data constants
DIMENSION = 32
ROOT_DIR = "images_2/"

CLASSES = ["Chinese_Tallow", "Euphorbia_Mili", "excoecaria", "Garden_Croton", "Hevea_Brasilinsis"]


app = Flask(__name__)

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

@app.route("/", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        dir_saved = 'static/img/' + filename
        # PraProses
        remove_background(dir_saved)

        img = Image.open(dir_saved).resize((DIMENSION, DIMENSION))
        img = list(itertools.chain.from_iterable(img.getdata()))
        predClazz, prob = predict(models, img)
        print prob
        return str(predClazz)

    return render_template('index.html')

@app.route("/species", methods=['GET', 'POST'])
def species():
    return render_template('Species.html')

@app.route("/up", methods=['GET', 'POST'])
def up():
    return render_template('upload.html')


def classify(models, dataSet):
    results = {}
    for trueClazz in CLASSES:
        count = 0
        correct = 0
        for item in dataSet[trueClazz]:
            predClazz, prob = predict(models, item)
            print "%s,%s,%f" % (trueClazz, predClazz, prob)
            count += 1
            if trueClazz == predClazz: correct += 1
        results[trueClazz] = (count, correct)
    return results

def predict(models, item):
    maxProb = 0.0
    bestClass = ""
    for clazz, model in models.iteritems():
        output = svm_predict([0], [item], model, "-q -b 1")
        prob = output[2][0][0]
        if prob > maxProb:
            maxProb = prob
            bestClass = clazz
    return (bestClass, maxProb)

def getModels(trainingData):
    models = {}
    param = getParam()
    for c in CLASSES:
        labels, data = getTrainingData(trainingData, c)
        prob = svm_problem(labels, data)
        m = svm_train(prob, param)
        models[c] = m
    return models

def getTrainingData(trainingData, clazz):
    labeledData = getLabeledDataVector(trainingData, clazz, 1)
    negClasses = [c for c in CLASSES if not c == clazz]
    for c in negClasses:
        ld = getLabeledDataVector(trainingData, c, -1)
        labeledData += ld
    random.shuffle(labeledData)
    unzipped = [list(t) for t in zip(*labeledData)]
    labels, data = unzipped[0], unzipped[1]
    return (labels, data)

def getParam():
    param = svm_parameter("-q")
    param.probability = 1
    param.kernel_type = 2 # RBF
    param.C = .01
    param.gamma = .00000001
    return param

def getLabeledDataVector(dataset, clazz, label):
    data = dataset[clazz]
    labels = [label] * len(data)
    output = zip(labels, data)
    return output


def buildImageList(dirName):
    for fileName in os.listdir(dirName):
        remove_background(dirName+fileName)
    imgs = [Image.open(dirName + fileName).resize((DIMENSION, DIMENSION)) for fileName in os.listdir(dirName)]
    imgs = [list(itertools.chain.from_iterable(img.getdata())) for img in imgs]
    return imgs


# remove backgound  = PRAPROSES PCD
def remove_background(file_location):
        ### CROP
        img = cv2.imread(file_location)

        ## (1) Convert to gray, and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, threshed = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

        ## (2) Morph-op to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

        ## (3) Find the max-area contour
        _, cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(cnts, key=cv2.contourArea)[-1]

        ## (4) Crop and save it
        x,y,w,h = cv2.boundingRect(cnt)
        dst = img[y:y+h, x:x+w]
        

        ### REMOVE WHITE

        ## (1) Convert to gray, and threshold
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        # th, threshed = cv2.threshold(gray, 163, 255, cv2.THRESH_BINARY_INV)
        threshed2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY_INV,251,6)

        kernel = np.ones((8,8), np.uint8)

        dilation = cv2.dilate(threshed2,kernel,iterations = 10)
        inverted = cv2.bitwise_not(dilation)
        backtorgb = cv2.cvtColor(inverted,cv2.COLOR_GRAY2RGB)

        hasil = cv2.subtract(dst,backtorgb)
        cv2.imwrite(file_location,hasil)



if __name__ == "__main__":
    
    ### CEK APAKAH ADA MODEL ADA ATAU TIDAK
    flag = True
    models = {}
    for i in CLASSES:
        models[i] = svm_load_model("model_"+i)
        if models[i] == None:
            flag = False    

    if flag==False : 
        trainingData = {}
        for clazz in CLASSES:
            train = buildImageList(ROOT_DIR + clazz + "/")
            trainingData[clazz] = train

        train = trainingData
        models = getModels(trainingData)
        
        for clazz, model in models.iteritems():
            svm_save_model("model_"+clazz, model) 

    app.run()

