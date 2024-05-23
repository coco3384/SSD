import os
import glob
import cv2
import numpy as np
import shutil
from tqdm import tqdm


def create_dataset(name: str, data: dict):
    os.makedirs(os.path.join(name, 'images'), exist_ok=True)
    os.makedirs(os.path.join(name, 'annotations'), exist_ok=True)
    for img, annotation in tqdm(zip(data['images'], data['annotations'])):
        img_src = img
        img_dst = os.path.join(name, 'images', os.path.basename(img))
        shutil.copy(src=img_src, dst=img_dst)

        ann_src = annotation
        ann_dst = os.path.join(name, 'annotations', os.path.basename(annotation))
        shutil.copy(src=ann_src, dst=ann_dst)


def main():
    img_size_threshold = [960, 1400]
    small_data = {'images': [], 'annotations': []}
    medium_data = {'images': [], 'annotations': []}
    large_data = {'images': [], 'annotations': []}

    img_list = glob.glob(os.path.join('VisDrone2019-DET-train', 'images', '*.jpg'))
    annotation_list = glob.glob(os.path.join('VisDrone2019-DET-train', 'annotations', '*.txt'))
    img_list.sort()
    annotation_list.sort()

    for img, annotation in zip(img_list, annotation_list):
        img_shape = cv2.imread(img).shape[:2]

        if img_shape[1] <= img_size_threshold[0]:
            small_data['images'].append(img)
            small_data['annotations'].append(annotation)
        if (img_shape[1] > img_size_threshold[0]) and (img_shape[1] <= img_size_threshold[1]):
            medium_data['images'].append(img)
            medium_data['annotations'].append(annotation)
        if img_shape[1] > img_size_threshold[1]:
            large_data['images'].append(img)
            large_data['annotations'].append(annotation)
    
    create_dataset(name='VisDrone2019-DET-train-small', data=small_data)
    create_dataset(name='VisDrone2019-DET-train-medium', data=medium_data)
    create_dataset(name='VisDrone2019-DET-train-large', data=large_data)




if __name__ == '__main__':
    main()