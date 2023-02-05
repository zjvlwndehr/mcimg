import os
import shutil
import cv2
from pathlib import Path
from threading import Thread
import json
import numpy as np
from datetime import datetime

class Engine():
    def __init__(self, img : str, method : int = 0) -> None:
        self.img = cv2.imread(img)
        self.width, self.height = self.img.shape[:2]
        self.width = self.width // 16 * 16
        self.height = self.height // 16 * 16

        self.path = Path('block16_hist')
        self.path_blocks = Path('block16')
        self.methods = [ cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA ]
        self.method = self.methods[method]
        self.histograms = []
        self.rtn = []
        self.times = 4
        self.save_dir = Path(f'{self.method}_{self.width}x{self.height}_{datetime.utcnow()}')

        self.images = []
    
    def startup(self) -> None:
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir()

    def resize(self, width : int, height : int) -> None:
        self.img = cv2.resize(self.img, (width, height))
        self.width, self.height = self.img.shape[:2]
        self.img = cv2.resize(self.img, (width * self.times, height * self.times))
        self.height, self.width = self.img.shape[:2]

    def start(self) -> None:
        print('Start engine')
        self.startup()
        print(f'Image size: {self.width}x{self.height}')
        self.resize(self.width, self.height)
        print(f'Resized image: {self.width}x{self.height}')
        print(f'Using method: {self.method}')
        print('Read histograms')
        self.read_hist()
        print('Parsing image')
        self.parsing_img()
        print('Rendering image')
        self.render_img()

    def read_hist(self) -> None:
        block_files = self.path_blocks.glob('*.png')
        for block_file in block_files:
            filename = block_file.stem
            block = cv2.imread(str(block_file))
            hist = cv2.calcHist([block], [0], None, [256], [0, 256])
            self.histograms.append([filename, hist])

    def calcHist(self, hist, coord : tuple) -> int:
        max_value : int = 0
        max_filename : str = ''
        for filename, _hist in self.histograms:
            value = cv2.compareHist(_hist, hist, self.method)
            if value > max_value:
                max_value = value
                max_filename = filename
        self.rtn.append([coord[0], coord[1], max_filename])
        # print(f'[DEBUG] {self.rtn[-1][2]}')
        return 0

    def parsing_img(self) -> None:
        # devide image to 16 x 16 pixel matrixes
        threads = []
        for i in range(0, self.height, 16):
            if i + 16 > self.height:
                break
            for j in range(0, self.width, 16):
                if j + 16 > self.width:
                    break
                hist = cv2.calcHist([self.img[i:i+16, j:j+16]], [0], None, [256], [0, 256])
                thread = Thread(target=self.calcHist, args=(hist, (i, j)))
                thread.start()
                threads.append(thread)

        for thread in threads:
            thread.join()
    
    def render_img(self) -> None:
        image = np.zeros(((self.width // 16) * 16, (self.height // 16) * 16, 3), int)
        image.fill(255)

        for i, j, filename in self.rtn:
            try:
                block = cv2.imread(str(self.path_blocks / (filename + '.png')))
                image[i:i+16, j:j+16] = block[15::-1, 15::-1]
            except:
                print(f'[ERROR] {filename}')
                print(f'[ERROR] {i}, {j}')
                print(f'[ERROR] image shape: {image.shape}')
                print(f'[ERROR] block shape: {block.shape}')
                continue
        print(f'Saving image as {self.method}_{self.width}x{self.height}.png')
        cv2.imwrite(f'{self.method}_{self.width}x{self.height}_{datetime.utcnow()}.png', image)
        print('Done')

    def devide_image(self, cut : int) -> None:
        for i in range(0, cut):
            for j in range(0, cut):
                self.images[i].append(self.img[i*self.width//cut:(i+1)*self.width//cut, j*self.height//cut:(j+1)*self.height//cut])
                
    def concat_image(self) -> None:
        full_image = np.zeros((self.width, self.height, 3), int)
        for i in range(0, len(self.images)):
            for j in range(0, len(self.images[i])):
                full_image[i*self.width//len(self.images):(i+1)*self.width//len(self.images), j*self.height//len(self.images[i]):(j+1)*self.height//len(self.images[i])] = self.images[i][j]
            
if __name__ == '__main__':
    # engine = Engine('img_512x512.png', 0)
    # engine = Engine('img_1024x1024.png', 2)
    # engine = Engine('img.png', 0)
    Engine('img_1024x1024.png', 0).start()
    # Engine('img_1024x1024.png', 3).start()
    
    