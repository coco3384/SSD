import os
import glob
from tqdm import tqdm
from VisDrone import VisDrone
from YoloClass import YoloClass
import numpy as np
import cv2

def xywh2xyxy(x, y, w, h):
    bbox_top = int(y - h/2)
    bbox_left = int(x - w/2)
    bbox_bottom = int(y + h/2)
    bbox_right = int(x + w/2)
    return [bbox_left, bbox_top, bbox_right, bbox_bottom]


def nms(boxes, scores, threshold=0.5):
    boxes = np.array(boxes)
    # sorted by scores
    idxs = np.argsort(scores)[::-1]

    picked = []

    while len(idxs) > 0:
        current = idxs[0]
        picked.append(current)

        other_boxes = boxes[idxs[1:]]
        picked_box = boxes[current]
        
        x1 = np.maximum(picked_box[0], other_boxes[:, 0])
        y1 = np.maximum(picked_box[1], other_boxes[:, 1])
        x2 = np.minimum(picked_box[2], other_boxes[:, 2])
        y2 = np.minimum(picked_box[3], other_boxes[:, 3])

        intersection_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)

        picked_area = (picked_box[2] - picked_box[0] + 1) * (picked_box[3] - picked_box[1] + 1)
        other_area = (other_boxes[:, 2] - other_boxes[:, 0] + 1) * (other_boxes[:, 3] - other_boxes[:, 1] + 1)

        union_area = picked_area + other_area - intersection_area
        iou = intersection_area / union_area

        idxs = idxs[np.where(iou <= threshold)[0] + 1]

    return picked

def apply_nms(annotation_path, origin_shape):
    with open(annotation_path, 'r') as f:
        content = f.readlines()
    boxes = []
    scores = []
    for line in content:
        cls, x, y, w, h, conf = [float(el) for el in line.split()]
        box = xywh2xyxy(x, y, w, h)
        boxes.append(box)
        scores.append(conf)
    
    picked_indices = nms(boxes, scores)
    
    # print('applying NMS & mergeing result')
    with open(annotation_path, 'w') as f:
        for idx, line in enumerate(content):
            cls, x, y, w, h, conf = [float(el) for el in line.split()]
            bbox = []
            for i, p in enumerate([x, y, w, h]):
                bbox.append(p / origin_shape[i % 2])
            # bbox = [p / origin_shape[i % 2] for i, p in enumerate([x, y, w, h])]
            if idx in picked_indices:
                f.write(f'{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {conf}\n')
    

def get_image_name_and_position(sub_region_name):
    base_name = os.path.basename(sub_region_name)
    position = base_name.split('_')[-2:]

    position[-1], file_extension = position[-1].split('.')

    len_of_underline = 2
    len_of_position_info = len(position[0]) + len(position[1]) + len(file_extension) + 1 + len_of_underline
    name = base_name[:-len_of_position_info]
    return name, position



def main():
    rebine_result_path = os.path.join('dataset', 'VisDrone2019-DET-test-large', 'labels_tile')
    os.makedirs(rebine_result_path, exist_ok=True)

    origin_image_dir = os.path.join('dataset', 'VisDrone2019-DET-test-large', 'images')
    sub_regions_image_dir = os.path.join('dataset', 'VisDrone2019-DET-test-large-tile-sub-regions', 'images')
    sub_regions_annotation_dir = os.path.join('dataset', 'VisDrone2019-DET-test-large-tile-sub-regions', 'predict_labels')

    sub_regions_annotation_list = glob.glob(os.path.join(sub_regions_annotation_dir, '*.txt'))
    sub_regions_image_list = glob.glob(os.path.join(sub_regions_image_dir, '*.jpg'))

    sub_regions_annotation_list.sort()
    sub_regions_image_list.sort()

    previous_name = None

    for sub_annotation in tqdm(sub_regions_annotation_list[:], total=len(sub_regions_annotation_list)):
        # split name and positoin
        name, position = get_image_name_and_position(sub_annotation)
        sub_image = os.path.join(sub_regions_image_dir, os.path.basename(sub_annotation)[:-4] + '.jpg')
        # check meta name
        if previous_name:
            if previous_name != name:
                origin_shape = cv2.imread(os.path.join(origin_image_dir, previous_name + '.jpg')).shape[:2][::-1]
                apply_nms(os.path.join(rebine_result_path, previous_name + '.txt'), origin_shape=origin_shape)

        previous_name = name

        # write bbox plus position
        sub_data = YoloClass(image=sub_image, annotations=sub_annotation)
        with open(os.path.join(rebine_result_path, name + '.txt'), 'a') as f:
            h, w = sub_data.get_img().shape[:2]
            for annotation in sub_data.annotations:
                cls = annotation.get_category()
                conf = annotation.get_conf()
                bbox = annotation.get_bbox_xywh(w, h)
                bbox_proj = {'x_center': bbox['x_center'] + int(position[0]),
                             'y_center': bbox['y_center'] + int(position[1]),
                             'bbox_width': bbox['bbox_width'],
                             'bbox_height': bbox['bbox_height']}
                f.write(f'{cls} {bbox_proj["x_center"]} {bbox_proj["y_center"]} {bbox_proj["bbox_width"]} {bbox_proj["bbox_height"]} {conf}\n')

    print(os.path.join(rebine_result_path, name + '.txt'))
    apply_nms(os.path.join(rebine_result_path, name + '.txt'), origin_shape=(1920, 1080))
        
        
        

if __name__ == '__main__':
    main()
