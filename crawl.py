from pathlib import Path
import cv2
import os
import shutil

path = Path('block')
if not path.exists():
    Exception('Path does not exist')
    exit(-1)

storage = Path('block16')
if not storage.exists():
    Exception('Storage path does not exist')
    os.mkdir(storage)

block_files = path.glob('*.png')

for block_file in block_files:
    block = cv2.imread(str(block_file))
    width, height, _ = block.shape
    if width == 16 and height == 16:
        shutil.copy(str(block_file), str(storage))
