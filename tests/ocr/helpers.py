# -*- coding: utf-8 -*-
"""
Helper functions for ocr project
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

SMALL_HEIGHT = 800

def implt(img, cmp=None, t=''):
    """ Show image using plt """
    plt.imshow(img, cmap=cmp)
    plt.title(t)
    plt.show()


def resize(img, height=SMALL_HEIGHT, allways=False):
    """ Resize image to given height """
    if (img.shape[0] > height or allways):
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))

    return img


def ratio(img, height=SMALL_HEIGHT):
    """ Getting scale ratio """
    return img.shape[0] / height


def extendImg(img, shape):
    """ Extend 2D image (numpy array) in vertical and horizontal direction
    Shape of result image will match 'shape'
    Args:
        img: image to be extended
        shape: shape (touple) of result image
    Returns:
        Extended image
    """
    x = np.zeros(shape, np.uint8)
    x[:img.shape[0], :img.shape[1]] = img
    return x

def thresholdToZero(img,threshold=50,maxVal=255):
    imageWidth = img.shape[1]
    imageHeight = img.shape[0]

    for pos in product(range(imageHeight), range(imageWidth)):
        # print(img[pos])
        pixel = img.item(pos)
        if pixel < threshold:
            img[pos] = 0

    return img

def rgbToGray(img):
    W = [0.11,0.59,0.3] # weights
    W_mean = np.tensordot(img,W, axes=((-1,-1)))[...,None]
    img[:] = W_mean.astype(img.dtype)
    without_alpha = img[:,:,0]

    return without_alpha
