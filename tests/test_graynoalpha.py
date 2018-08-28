import numpy as np
import cv2

def test_graynoalpha():
    img = cv2.cvtColor(cv2.imread("data/pagedet/1.jpg"), cv2.COLOR_BGR2RGB)
    W = [0.11,0.59,0.3] # weights
    W_mean = np.tensordot(img,W, axes=((-1,-1)))[...,None]
    img[:] = W_mean.astype(img.dtype)
    without_alpha = img[:,:,0]

    assert without_alpha.ndim == 2
