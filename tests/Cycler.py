import numpy as np

from ocr.normalization import imageNorm, letterNorm
import cv2
from ocr import page, words, charSeg
from ocr.helpers import implt, resize
from ocr.tfhelpers import Graph
from ocr.datahelpers import idx2char

class Cycler:
    """ Cycle through the words and recognise them """
    height = 60

    def __init__(self, image, boxes, charClass):
        self.boxes = boxes       # Array of bounding boxes
        self.image = image       # Whole image
        self.charClass = charClass

    def recognise(self, img):
        """ Recognising word and printing it """
        # Pre-processing the word
        img = imageNorm(
            img,
            60,
            border=False,
            tilt=True,
            hystNorm=True)

        # Separate letters
        img = cv2.copyMakeBorder(
            img,
            0, 0, 30, 30,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0])
        gaps = charSeg.segmentation(img, RNN=True, debug=True)

        chars = []
        for i in range(len(gaps)-1):
            char = img[:, gaps[i]:gaps[i+1]]
            # TODO None type error after treshold
            char, dim = letterNorm(char, is_thresh=True, dim=True)
            # TODO Test different values
            if dim[0] > 4 and dim[1] > 4:
                chars.append(char.flatten())

        chars = np.array(chars)
        word = ''
        if len(chars) != 0:
            pred = self.charClass.run(chars)
            for c in pred:
                word += idx2char(c)

        # print("Word: " + word)
        return word

    def idxImage(self, index):
        """ Getting next image from the array """
        if index < len(self.boxes):
            b = self.boxes[index]
            x1, y1, x2, y2 = b

            # Cuting out the word image
            img = self.image[y1:y2, x1:x2]
            # implt(img, t='Index: ' + str(index))

            word = self.recognise(img)
            return word
