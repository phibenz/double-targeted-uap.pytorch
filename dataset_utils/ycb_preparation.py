import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import shutil
import csv
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

from config.config import YCB_PATH, YCB_PYTORCH_PATH


SCALE_PERCENT=11
assert os.path.exists(YCB_PATH)
if not os.path.exists(YCB_PYTORCH_PATH):
    os.makedirs(YCB_PYTORCH_PATH)

ycb_pytorch_train = os.path.join(YCB_PYTORCH_PATH, "train")
if not os.path.exists(ycb_pytorch_train):
    os.makedirs(ycb_pytorch_train)

ycb_pytorch_test = os.path.join(YCB_PYTORCH_PATH, "test")
if not os.path.exists(ycb_pytorch_test):
    os.makedirs(ycb_pytorch_test)

ycb_folders = sorted(os.listdir(YCB_PATH))
for folder in ycb_folders:
    print("Splitting folder: {}".format(folder))
    file_list = os.listdir(os.path.join(YCB_PATH, folder))
    # Create folder in YCB_PYTORCH
    ycb_pytorch_train_folder = os.path.join(ycb_pytorch_train, folder)
    os.makedirs(ycb_pytorch_train_folder)

    ycb_pytorch_test_folder = os.path.join(ycb_pytorch_test, folder)
    os.makedirs(ycb_pytorch_test_folder)

    # get only the .jpg files
    jpg_list = [f for f in file_list if f.endswith(".jpg")]
    train_files, test_files = train_test_split(jpg_list, test_size=0.2, random_state=42)
    for train_f in train_files:
        src = os.path.join(YCB_PATH, folder, train_f)
        # Load the image
        input_img = cv2.imread(src, flags=cv2.IMREAD_COLOR)

        width = int(input_img.shape[1] * SCALE_PERCENT / 100)
        height = int(input_img.shape[0] * SCALE_PERCENT / 100)
        dim = (width, height)
        resized_img = cv2.resize(input_img, dim, interpolation=cv2.INTER_AREA)

        dst = os.path.join(ycb_pytorch_train_folder, train_f)
        cv2.imwrite(dst, resized_img)
    for test_f in test_files:
        src = os.path.join(YCB_PATH, folder, test_f)
        # Load the image
        input_img = cv2.imread(src, flags=cv2.IMREAD_COLOR)

        width = int(input_img.shape[1] * SCALE_PERCENT / 100)
        height = int(input_img.shape[0] * SCALE_PERCENT / 100)
        dim = (width, height)
        resized_img = cv2.resize(input_img, dim, interpolation=cv2.INTER_AREA)

        dst = os.path.join(ycb_pytorch_test_folder, test_f)
        cv2.imwrite(dst, resized_img)
