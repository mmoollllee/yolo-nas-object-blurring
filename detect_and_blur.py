import argparse
import configparser
from pathlib import Path
import os
import cv2
import torch
from numpy import random

from utils.loadimages import LoadImages
from utils.lock import lock_script
from utils.general import xyxy2xywh
from utils.plots import plot_one_box

from super_gradients.common.object_names import Models
from super_gradients.training import models

# Note that currently only YoloX, PPYoloE and YOLO-NAS are supported.

def predict(images, iou, conf):

    # model = models.get('yolo_nas_m', pretrained_weights="coco").to("cpu")
    model = models.get(model_name=Models.YOLO_NAS_M, num_classes=12, checkpoint_path="ckpt_best.pth").to("cpu")
    predictions = model.predict(images, iou, conf, fuse_model=False)
    return predictions

def process(predictions, img, path, dest, save_org, save_txt, blurratio, compression, hidedetarea, delete_org):
    for pred in predictions:
        class_names = pred.class_names
        labels = pred.prediction.labels
        confidence = pred.prediction.confidence
        bboxes = pred.prediction.bboxes_xyxy

        p = Path(path)  # to Path
        save_path = str(dest + "/" + p.name)  # img.jpg
        txt_path = str(dest + '/labels/' + p.stem)  # img.txt
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        # Get names and colors
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(12)]

        file_name, file_type = os.path.splitext(save_path)
        if (file_type == ".webp"):
            options = [int(cv2.IMWRITE_WEBP_QUALITY), compression]
        elif  (file_type == ".jpg" or file_type == ".jpeg"):
            options = [cv2.IMWRITE_JPEG_QUALITY, compression, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        else:
            options = []

        if save_org and save_txt:
            print(f"Saving original image to {txt_path}{file_type}");
            cv2.imwrite(txt_path + file_type, img, options)

        for i, (label, conf, bbox) in enumerate(zip(labels, confidence, bboxes)):
            print("--" * 10)
            print("prediction: ", i)
            print("label_id: ", int(label))
            print("label_name: ", class_names[int(label)])
            print("confidence: ", conf)
            print("bbox: ", bbox)
            print("--" * 10)

            xyxy = bbox
                
            if save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (label, *xywh, conf) # if opt.save_conf else (label, *xywh)  # label format
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            #Add Object Blurring Code
            if blurratio:
                crop_obj = img[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
                blur = cv2.blur(crop_obj,(blurratio,blurratio))
                img[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])] = blur
            
            # Add bbox to image
            print(int(label))
            color = colors[int(label)]
            labeltxt = f'{class_names[int(label)]} {conf}' # {conf:.2f}
            if not hidedetarea:
                plot_one_box(xyxy, img, label=labeltxt, color=color, line_thickness=2)

        # Save results (image with detections)
        cv2.imwrite(save_path, img, options)
        print(f" The image with the result is saved in: {save_path}")
        
        if delete_org:
            os.remove(p)

        return img


if __name__ == '__main__':

    if os.path.isfile('./config.txt'):
        config = configparser.ConfigParser()
        config.read('config.txt')
        
    defaults = {
        'source': config.get('Main', 'source', fallback='input'),
        'dest': config.get('Main', 'dest', fallback='output'),
        'conf_thres': config.getfloat('Main', 'conf_thres', fallback=0.25),
        'blurratio': config.getint('Main', 'blurratio', fallback=20),
        'rotate': config.getint('Main', 'rotate', fallback=0),
        'limit': config.getint('Main', 'limit', fallback=10),
        'compression': config.getint('Main', 'compression', fallback=60),
        'classes': [0, 1, 2, 3, 5],
        'delete_org': config.getboolean('Main', 'delete_org', fallback=False),
        'hidedetarea': config.getboolean('Main', 'hidedetarea', fallback=False),
        'save_org': config.getboolean('Main', 'save_org', fallback=False),
        'save_txt': config.getboolean('Main', 'save_txt', fallback=False),
    }
    
    parser = argparse.ArgumentParser()
    parser.set_defaults(**defaults)
    parser.add_argument('--source', type=str, default=defaults['source'], help='source') 
    parser.add_argument('--dest', default=defaults['dest'], help='results folder name')
    parser.add_argument('--conf-thres', type=float, default=defaults['conf_thres'], help='object confidence threshold')
    parser.add_argument('--classes', nargs='+', type=int, default=defaults['classes'], help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--blurratio',type=int,default=defaults['blurratio'], help='blur opacity')
    parser.add_argument('--rotate',type=int, default=defaults['rotate'], help='Rotate clockwise 90, 180, 270')
    parser.add_argument('--limit',type=int, default=defaults['limit'], help='Limit images to process')
    parser.add_argument('--compression',type=int, default=defaults['compression'], help='Compression Value for Output Images')

    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')

    parser.add_argument('--hidedetarea',action='store_true', help='Hide Detected Area')
    parser.add_argument('--delete-org',action='store_true', help='Delete Input Files')
    opt = parser.parse_args()

    if lock_script():
        print(opt)

        data = LoadImages(opt.source, limit=opt.limit, rotate=opt.rotate)

        limiter = 0
        for path, img in data:
            limiter += 1
            if (limiter == opt.limit):
                print("Limit reached!")
                break
            
            predictions = predict(img, opt.iou_thres ,opt.conf_thres)

            result = process(predictions, img, path, dest = opt.dest, save_org = opt.save_org, save_txt = opt.save_txt, blurratio = opt.blurratio, compression = opt.compression, hidedetarea = opt.hidedetarea, delete_org = opt.delete_org)

