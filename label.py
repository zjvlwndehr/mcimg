import os
import shutil
import cv2
from pathlib import Path
import json

if __name__ == '__main__':
    path = Path('block16')
    if not path.exists():
        Exception('Path does not exist')
        exit(-1)
    storage = Path('block16_hist')
    if not storage.exists():
        Exception('Storage path does not exist')
        os.mkdir(storage)

    block_files = path.glob('*.png')

    for block_file in block_files:
        block = cv2.imread(str(block_file))
        hist = cv2.calcHist([block], [0], None, [256], [0, 256])
        
        with open(str(storage / (block_file.stem + '.json')), 'w') as f:
            f.write(json.dumps(hist.tolist()))