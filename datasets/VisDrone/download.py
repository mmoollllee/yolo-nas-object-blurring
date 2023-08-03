import os
from pathlib import Path
from ultralytics.utils.downloads import download
from utils.visdrone2yolo import visdrone2yolo

HOME = os.getcwd()

# Download
dir = Path(HOME)  # dataset root dir
urls = ['https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip',
         'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip',
         'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip',
         'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-challenge.zip']
download(urls, dir=dir, curl=True, threads=4)

# Convert
for d in 'VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev':
   visdrone2yolo(dir / d)  # convert VisDrone annotations to YOLO labels
