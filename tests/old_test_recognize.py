import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os

from ocr.normalization import imageNorm, letterNorm
from ocr import page, words, charSeg
from ocr.helpers import implt, resize
from ocr.tfhelpers import Graph
from ocr.datahelpers import idx2char

import Cycler

FILE_LOC = 'data/pagedet/1.jpg'
MODEL_LOC = 'models/char-clas/en/CharClassifier'

def test_recognize():
    charClass = Graph(MODEL_LOC)
    # load the saved image
    image = cv2.cvtColor(cv2.imread(FILE_LOC), cv2.COLOR_BGR2RGB)
    crop = page.detection(image)
    bBoxes = words.detection(crop)
    cycler = Cycler.Cycler(crop,bBoxes,charClass)
    allWords = []
    for i in range(len(bBoxes)-1):
        allWords.append(cycler.idxImage(i))

    isStructEmpty = structEmpty(allWords)
    assert isStructEmpty == False

def structEmpty(struct):
    if struct:
        return False
    else:
        return True
