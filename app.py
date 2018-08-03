from flask import Flask, render_template, request, json
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import logging
from logging.handlers import RotatingFileHandler

from ocr.normalization import imageNorm, letterNorm
from ocr import page, words, charSeg
from ocr.helpers import implt, resize
from ocr.tfhelpers import Graph
from ocr.datahelpers import idx2char

import Cycler

app = Flask(__name__)

if not app.debug:
    formatter = logging.Formatter(
        "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s")
    handler = RotatingFileHandler('logs/errors.log', maxBytes=10000000, backupCount=5)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)


MODEL_LOC = 'models/char-clas/en/CharClassifier'

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/recognize", methods=['POST'])
def recognize():
    if request.method == 'POST':
        if 'image' not in request.files:
            print('No file part')
            return render_template('index.html')
        file = request.files['image']
        if file.filename == '':
            print('No selected file')
            return render_template('index.html')
        if file:
            file.save(os.path.join('images',file.filename))
            filename = file.filename
            # load the trained model
            charClass = Graph(MODEL_LOC)
            # load the saved image
            image = cv2.cvtColor(cv2.imread("images/"+filename), cv2.COLOR_BGR2RGB)
            crop = page.detection(image)
            bBoxes = words.detection(crop)
            cycler = Cycler.Cycler(crop,bBoxes,charClass)
            allWords = []
            for i in range(len(bBoxes)-1):
                allWords.append(cycler.idxImage(i))
                print(allWords[i])


    return render_template('words.html', allWords=allWords)

if __name__ == "__main__":
    app.run()
