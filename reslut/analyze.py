#用于分析数据样本

import json
import cv2
import os
from tqdm import tqdm

data_root = "/home/w/Desktop/TC/data/"
imageRoot = data_root + "images/"
imageOutRoot = data_root + "oImageBBox/"
annFilePath = data_root + 'train/annotationsw.json'

def getBBoxById(annFile, imageId):
    temp = []
    for imageBBoxInfo in annFile['annotations']:
        if int(imageBBoxInfo['image_id']) == int(imageId):
            temp.append(imageBBoxInfo)

    return temp


with open(annFilePath, 'r') as F:
    annFile = json.load(F)
    datalen = len(annFile['images'])
    count = 0
    for imageFileInfo in tqdm(annFile['images']):
        count = count + 1
        imageOutPath = os.path.join(imageOutRoot, imageFileInfo['file_name'])
        image = cv2.imread(os.path.join(imageRoot, imageFileInfo['file_name']))

        bbox = getBBoxById(annFile, imageFileInfo['id'])
        if len(bbox) == 0:
            continue

        for bboxInfo in bbox:
            coordinate = bboxInfo['bbox']
            categoryId = bboxInfo['category_id']
            image = cv2.rectangle(image, (int(coordinate[0]), int(coordinate[1])),
                                  (int(coordinate[2] + coordinate[0]), int(coordinate[3] + coordinate[1])), (0, 0, 255),
                                  2)
            image = cv2.putText(image, str(int(categoryId)), (int(coordinate[0]), int(coordinate[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imwrite(imageOutPath, image)
