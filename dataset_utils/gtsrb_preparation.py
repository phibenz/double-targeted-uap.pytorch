import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import shutil
import csv
import numpy as np
from config.config import GTSRB_PATH

gtsrb_train_folder = os.path.join(GTSRB_PATH, "Training")
assert os.path.exists(gtsrb_train_folder)

gtsrb_test_folder_source = os.path.join(GTSRB_PATH, "Final_Test", "Images")
assert os.path.exists(gtsrb_test_folder_source)

gtsrb_test_folder_target = os.path.join(GTSRB_PATH, "Testing")
if not os.path.exists(gtsrb_test_folder_target):
    os.makedirs(gtsrb_test_folder_target)

gtsrb_test_gt_file = os.path.join(GTSRB_PATH, "GT-final_test.csv")
assert os.path.isfile(gtsrb_test_gt_file)

# # Delete README in Training folder
# train_readme_file = os.path.join(gtsrb_train_folder, "Readme.txt")
# if os.path.isfile(train_readme_file):
#     print("Removing: {}".format(train_readme_file))
#     os.remove(train_readme_file)
#
# # Delete the GT file in every Training folder
# for id in range(43):
#     subfolder_name = str(id).zfill(5)
#     gt_file_name = "GT-" + str(id).zfill(5) + ".csv"
#     gt_file = os.path.join(gtsrb_train_folder, subfolder_name, gt_file_name)
#     if os.path.isfile(gt_file):
#         print("Removing: {}".format(gt_file))
#         os.remove(gt_file)

# Creating all folders for test images
for id in range(43):
    test_folder = os.path.join(gtsrb_test_folder_target, str(id).zfill(5))
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

with open(gtsrb_test_gt_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    next(csv_reader)
    line_count = 0

    for row in csv_reader:
        filename = row[0]
        class_id = row[7]
        print('Filename: {}, Class {}'.format(filename, class_id))
        source_path = os.path.join(gtsrb_test_folder_source, filename)
        target_path = os.path.join(gtsrb_test_folder_target, class_id.zfill(5), filename)
        print("Source: {}\nTarget: {}\n".format(source_path, target_path))
        shutil.copyfile(source_path, target_path)
        line_count += 1
    print('Processed {} lines.'.format(line_count))
