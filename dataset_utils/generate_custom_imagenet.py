import os
import sys
import shutil
import numpy as np
from imagenet_utils.imagenet_utils import get_dicts
from config.config import IMAGENET_PATH, IMAGENET10_PATH


new_imagenet_path = IMAGENET10_PATH

if not os.path.exists(new_imagenet_path):
    os.makedirs(new_imagenet_path)

src_traindir = os.path.join(IMAGENET_PATH, 'train')
src_valdir = os.path.join(IMAGENET_PATH, 'val')
src_dirs = [src_traindir, src_valdir]

dest_traindir = os.path.join(new_imagenet_path, 'train')
if not os.path.exists(dest_traindir):
    os.makedirs(dest_traindir)
dest_valdir = os.path.join(new_imagenet_path, 'val')
if not os.path.exists(dest_valdir):
    os.makedirs(dest_valdir)
dest_dirs = [dest_traindir, dest_valdir]

dirs = zip(src_dirs, dest_dirs)

idx2label, cls2label = get_dicts()
idx2foldername = list(cls2label.keys())

# Select 10 random classes
np.random.seed(0)
cl_idxs = np.random.randint(0, 1000, 10)
# Take only the first 100 classes
# cl_idxs = np.arange(100)

for src_dir, dest_dir in dirs:
    for idx in cl_idxs:
        foldername = idx2foldername[idx]

        src = os.path.abspath(os.path.join(src_dir, foldername))
        dest = os.path.abspath(os.path.join(dest_dir, foldername))
        print("SRC: {}".format(src))
        print("DEST: {}".format(dest))

        # The method link() creates a hard link pointing to src named dst.
        # This method is very useful to create a copy of existing file.
        # src − This is the source file path for which hard link would be created.
        # dest − This is the target file path where hard link would be created.
        # Linking did not work due to permission error
        # os.link(src=src, dst=dest)
        # Copy instead
        shutil.copytree(src=src, dst=dest, symlinks=False, ignore=None)

print("Done!")
