import numpy as np
import cv2
from skimage import color, exposure, transform

def preprocess_gtsrb_img(img, img_size=48):
    # Histogram normalization
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img[:,:,1] = cv2.equalizeHist(img[:,:,1])
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])

    # rescale to img_size
    img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_CUBIC)

    return img
