import os
import json
import numpy as np
import shutil
from config.config import PROJECT_PATH, IMAGENET_PATH, CUSTOM_DATASETS_PATH, DATA_PATH, YCB_PATH, YCB_PYTORCH_PATH


def get_dataset_dicts(dataset):
    if dataset == "imagenet":
        # Imagenet class names
        idx2label = []
        cls2label = {}
        with open(os.path.join(PROJECT_PATH, "dataset_utils/imagenet_class_index.json"), "r") as read_file:
            class_idx = json.load(read_file)
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    elif dataset == "ycb":
        foldernames = sorted(os.listdir(YCB_PATH))
        idx2label = ["_".join(folder.split("_")[1:]) for folder in foldernames]
        cls2label = {folder: "_".join(folder.split("_")[1:])  for folder in foldernames}
    elif dataset == "gtsrb":
        idx2label = []
        cls2label = {}
        with open(os.path.join(PROJECT_PATH, "dataset_utils/gtsrb_class_index.json"), "r") as read_file:
            class_idx = json.load(read_file)
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
    elif dataset == "ycb":
        idx2label = sorted(os.listdir(os.path.join(YCB_PATH, "train")))
        cls2label = {}
    else:
        raise ValueError
    return idx2label, cls2label


def get_class_specific_dataset_folder_name(dataset, source_classes, samples_per_class=-1):
    assert dataset in ['imagenet', 'ycb']
    if len(source_classes)>1:
        class_string = "{}_classes_{}_{}".format(len(source_classes), source_classes[0], source_classes[-1])
    elif len(source_classes)==1:
        class_string = "{}_classes_{}".format(len(source_classes), source_classes[0])
    else:
        class_string = "no_classes"

    if samples_per_class <= 0:
        out_string = "separated_{}_{}".format(dataset, class_string)
    else:
        out_string = "separated_{}_{}_{}_imgs".format(dataset, class_string, samples_per_class)
    path = os.path.join(CUSTOM_DATASETS_PATH, out_string)
    return path


def generate_separated_dataset(dataset, source_classes, samples_per_class=-1):
    separated_dataset_path = get_class_specific_dataset_folder_name(dataset, source_classes, samples_per_class=samples_per_class)

    if dataset == "imagenet":
        src_traindir = os.path.join(IMAGENET_PATH, 'train')
        src_valdir = os.path.join(IMAGENET_PATH, 'val')
    elif dataset == "ycb":
        src_traindir = os.path.join(YCB_PYTORCH_PATH, 'train')
        src_valdir = os.path.join(YCB_PYTORCH_PATH, 'test')
    src_dirs = [src_traindir, src_valdir]

    if not os.path.exists(separated_dataset_path):
        os.makedirs(separated_dataset_path)
    else:
        shutil.rmtree(separated_dataset_path)
        os.makedirs(separated_dataset_path)

    source_traindir = os.path.join(separated_dataset_path, 'source_classes_train')
    if not os.path.exists(source_traindir):
        os.makedirs(source_traindir)
    source_valdir = os.path.join(separated_dataset_path, 'source_classes_val')
    if not os.path.exists(source_valdir):
        os.makedirs(source_valdir)
    source_classes_dirs = [source_traindir, source_valdir]

    others_traindir = os.path.join(separated_dataset_path, 'other_classes_train')
    if not os.path.exists(others_traindir):
        os.makedirs(others_traindir)
    others_valdir = os.path.join(separated_dataset_path, 'others_classes_val')
    if not os.path.exists(others_valdir):
        os.makedirs(others_valdir)
    other_classes_dirs = [others_traindir, others_valdir]

    if dataset == "imagenet":
        num_classes = 1000
    elif dataset == "ycb":
        num_classes = 98

    other_classes = [cl for cl in range(num_classes) if cl not in source_classes]

    idx2label, cls2label = get_dataset_dicts(dataset)
    idx2foldername = list(cls2label.keys())

    src_source_dir_pairs = zip(src_dirs, source_classes_dirs)
    src_others_dir_pairs = zip(src_dirs, other_classes_dirs)

    print("Creating symlinks for the source class")
    for src_dir, dest_dir in src_source_dir_pairs:
        for idx in range(num_classes):
            foldername = idx2foldername[idx]

            src = os.path.abspath(os.path.join(src_dir, foldername))
            dest = os.path.abspath(os.path.join(dest_dir, foldername))

            if idx in source_classes:
                if ("train" in dest_dir) & (samples_per_class>0):
                    os.makedirs(dest)
                    for idx, filename in enumerate(sorted(os.listdir(src))):
                        if idx >= samples_per_class:
                            continue
                        src_file = os.path.join(src, filename)
                        dest_file = os.path.join(dest, filename)
                        # print("SRC FILE: {}".format(src_file))
                        # print("DEST FILE: {}".format(dest_file))
                        os.symlink(src_file, dest_file)
                else:
                    # print("SRC: {}".format(src))
                    # print("DEST: {}".format(dest))
                    os.symlink(src, dest)
            if idx in other_classes:
                # print("Folder: {}".format(dest))
                os.makedirs(dest)

    print("Creating symlinks for the other classes")
    for src_dir, dest_dir in src_others_dir_pairs:
        for idx in range(num_classes):
            foldername = idx2foldername[idx]

            src = os.path.abspath(os.path.join(src_dir, foldername))
            dest = os.path.abspath(os.path.join(dest_dir, foldername))

            if idx in other_classes:
                if ("train" in dest_dir) & (samples_per_class>0):
                    os.makedirs(dest)
                    for idx, filename in enumerate(sorted(os.listdir(src))):
                        if idx >= samples_per_class:
                            continue
                        src_file = os.path.join(src, filename)
                        dest_file = os.path.join(dest, filename)
                        # print("SRC FILE: {}".format(src_file))
                        # print("DEST FILE: {}".format(dest_file))
                        os.symlink(src_file, dest_file)
                else:
                    # print("SRC: {}".format(src))
                    # print("DEST: {}".format(dest))
                    os.symlink(src, dest)
            if idx in source_classes:
                # print("Folder: {}".format(dest))
                os.makedirs(dest)
