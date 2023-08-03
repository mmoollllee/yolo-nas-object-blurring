import glob
import os
from pathlib import Path

import cv2

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

class LoadImages:  # for inference
    def __init__(self, path, limit=0, rotate=0):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        ni = len(images)

        if (limit):
            images = images[:limit]

        self.rotate = rotate
        self.files = images
        self.nf = ni # number of files
        self.mode = 'image'
        
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are: {img_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        img = cv2.imread(path)  # BGR

        # If not a image, delete file
        if img is None:
            os.remove(path)

        # rotation
        if (self.rotate == 90):
            img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
        elif (self.rotate == 180):
            img0 = cv2.rotate(img0, cv2.ROTATE_180)
        elif (self.rotate == 270):
            img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)

        assert img is not None, 'Image Not Found ' + path

        return path, img

    def __len__(self):
        return self.nf  # number of files
