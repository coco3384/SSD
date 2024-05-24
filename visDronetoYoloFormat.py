import glob
import os
from tqdm import tqdm
import cv2
import yaml

if __name__ == '__main__':
    des_dir = 'annotations_yolo'
    os.makedirs(des_dir, exist_ok=True)
    annotations_list = glob.glob(os.path.join('annotations_2000_1500', '*.txt'))
    image_list = glob.glob(os.path.join('images_2000_1500', '*.jpg'))
    annotations_list.sort()
    image_list.sort()
    """
    for image, annotations_txt in tqdm(zip(image_list, annotations_list)):
        img = cv2.imread(image)
        h, w = img.shape[:2]
        file_name = os.path.basename(annotations_txt)
        with open(annotations_txt, 'r') as f:
            content = f.readlines()
        
        with open(os.path.join(des_dir, file_name), 'a') as f2:
            for anno in content:
                bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion = anno.rstrip().split(',')
                bbox_left, bbox_top, bbox_width, bbox_height = [int(item) for item in [bbox_left, bbox_top, bbox_width, bbox_height]]
                bbox_center = ((bbox_left + bbox_width / 2) / w, (bbox_top + bbox_height / 2) / h)
                f2.write(f'{category} {bbox_center[0]:.6f} {bbox_center[1]:.6f} {bbox_width / w:.6f} {bbox_height / h:.6f}\n')
    """
    #    0: pedestrian
    #    1: people
    #    2: bicycle
    #    3: car
    #    4: van
    #    5: truck
    #    6: tricycle
    #    7: awning-tricycle
    #    8: bus
    #    9: motor      
    #   10: others
    data = {
        'names': ['pedestrain', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'],
        'nc': 11,
        'test': '',
        'train': '',
        'val': '',

    }      
    with open('data.ymal', 'w') as f:
        yaml.dump(data, f)