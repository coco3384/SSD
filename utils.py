from VisDroneAnnotation import VisDroneAnnotation, VisDroneAnnotations
from VisDrone import VisDrone
import numpy as np


def candidates_resize(candidates, scale, bound):
    resize_candidates = set()
    for v in candidates:
        x, y, w, h = v
        x = min(x * scale, bound[1])
        y = min(y * scale, bound[0])
        w = w * scale
        h = h * scale
        resize_candidates.add((int(x), int(y), int(w), int(h)))
    return resize_candidates

def _out_of_bound(p, length, bound):
    return bound - p if p + length > bound else length

def _cal_new_annotation_bbox(candidate_region, gt_bbox, bounds):
    x = max(gt_bbox['min_x'] - candidate_region['min_x'], 0)
    y = max(gt_bbox['min_y'] - candidate_region['min_y'], 0)
    w = _out_of_bound(x, gt_bbox['max_x'] - gt_bbox['min_x'], bounds[1])
    h = _out_of_bound(y, gt_bbox['max_y'] - gt_bbox['min_y'], bounds[0])
    return x, y, w, h

def _get_sub_img(candidate_region, img):
    sub_img = img[candidate_region['min_y']:candidate_region['max_y'], candidate_region['min_x']:candidate_region['max_x'], :]
    return sub_img

def _crop_from_candidate(candidate, annotations, img):
    candidate_region = {'min_x': candidate[0], 'min_y': candidate[1], 'max_x': candidate[0] + candidate[2], 'max_y': candidate[1] + candidate[3]}
    sub_img = _get_sub_img(candidate_region, img)
    sub_annotations = []
    for annotation in annotations:
        gt_bbox = {'min_x': annotation.bbox_left, 'min_y': annotation.bbox_top, 'max_x': annotation.bbox_right, 'max_y': annotation.bbox_bottom}
        if intersect_perc(candidate_region, gt_bbox) > 0.5:
            bbox_left, bbox_top, bbox_width, bbox_height = _cal_new_annotation_bbox(candidate_region, gt_bbox, sub_img.shape[:2])
            sub_annotation = VisDroneAnnotation(annotation=[bbox_left, bbox_top, bbox_width, bbox_height, annotation.score, annotation.category, annotation.truncation,annotation.occlusion])
            sub_annotations.append(sub_annotation)

    sub_annotations = VisDroneAnnotations(sub_annotations)
    sub_data = VisDrone(sub_img, sub_annotations)
    return sub_data

def crop_from_candidates(candidates, annotations, img):
    sub_regions = {}
    for candidate in candidates:
        sub_data = _crop_from_candidate(candidate, annotations, img)
        # sub_data.show_annotations()
        sub_regions[(candidate[0], candidate[1])] = sub_data
            
    return sub_regions

def is_distorted(candidate, prefered_size):
    x, y, w, h = candidate
    long_side = max(w, h)
    if long_side > prefered_size * 1.2:
        return True
    else:
        return False

def crop_along_x_axis(x, y, w, h, prefered_size):
    prefered_size = int(prefered_size)
    overlay = 0.3
    start_point = x
    end_point = x + w
    cropped_regions = set()
    while start_point < (end_point - prefered_size):
        cropped_region = (start_point, y, prefered_size, h)
        cropped_regions.add(cropped_region)

        start_point = int(start_point + (1 - overlay) * prefered_size)

    return cropped_regions


def crop_along_y_axis(x, y, w, h, prefered_size):
    overlay = 0.3
    prefered_size = int(prefered_size)
    start_point = y
    end_point = y + h
    cropped_regions = set()
    while start_point < (end_point - prefered_size):
        cropped_region = (x, start_point, w, prefered_size)
        cropped_regions.add(cropped_region)

        start_point = int(start_point + (1 - overlay) * prefered_size)

    return cropped_regions

def crop_distorted_region(candidates, prefered_size):
    new_candidates = set()
    for candidate in candidates:
        if is_distorted(candidate, prefered_size):
            x, y, w, h = candidate
            if w > h:
                cropped_regions = crop_along_x_axis(x, y, w, h, prefered_size)
                for crop_region in cropped_regions:
                    new_candidates.add(crop_region)
            else:
                cropped_regions = crop_along_y_axis(x, y, w, h, prefered_size)
                for crop_region in cropped_regions:
                    new_candidates.add(crop_region)

        else:
            new_candidates.add(candidate)

    return new_candidates

def contain(a, b):
    # return weather b is inside a
    if isinstance(a, dict) and isinstance(b, dict):
        return contain_xyxy_dict(a, b)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return contain_xywh_tuple(a, b)

def contain_xywh_tuple(a, b):
    # (x, y, w, h)
    xa, ya, wa, ha = a
    xb, yb, wb, hb = b
    a = {'min_x': xa, 'min_y': ya, 'max_x': xa+wa, 'max_y': ya+ha}
    b = {'min_x': xb, 'min_y': yb, 'max_x': xb+wb, 'max_y': yb+hb}
    return contain_xyxy_dict(a, b)

def contain_xyxy_dict(a, b):
    # return wheather b is inside a
    if (a['min_x'] < b['min_x'] and a['min_y'] < b['min_y']) and (a['max_x'] > b['max_x'] and a['max_y'] > b['max_y']):
        return True
    else:
        return False


def iou(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return iou_xyxy_dict(a, b)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return iou_xywh_tuple(a, b)


def iou_xywh_tuple(a, b):
    x_a, y_a, w_a, h_a = a
    x_b, y_b, w_b, h_b = b
    
    a = {'min_x': x_a, 'min_y': y_a, 'max_x': x_a + w_a, 'max_y': y_a + h_a}
    b = {'min_x': x_b, 'min_y': y_b, 'max_x': x_b + w_b, 'max_y': y_b + h_b}
    return iou_xyxy_dict(a, b)

def iou_xyxy_dict(a, b):
    areaA = (a['max_x'] - a['min_x']) * (a['max_y'] - a['min_y'])
    areaB = (b['max_x'] - b['min_x']) * (b['max_y'] - b['min_y'])

    if contain(a, b):
        return areaB / areaA
    
    elif contain(b, a):
        return areaA / areaB
    
    else:
        xA = max(a['min_x'], b['min_x'])
        yA = max(a['min_y'], b['min_y'])
        xB = min(a['max_x'], b['max_x'])
        yB = min(a['max_y'], b['max_y'])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        return inter_area / (areaA + areaB - inter_area)


def intersect_perc(a, b):
    # intersection perc of b in a
    if isinstance(a, dict) and isinstance(b, dict):
        return intersect_perc_xyxy_dict(a, b)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return intersect_perc_xywh_tuple(a, b)

def intersect_perc_xywh_tuple(a, b):
    xa, ya, wa, ha = a
    xb, yb, wb, hb = b
    a = {'min_x': xa, 'min_y': ya, 'max_x': xa+wa, 'max_y': ya+ha}
    b = {'min_x': xb, 'min_y': yb, 'max_x': xb+wb, 'max_y': yb+hb}
    return intersect_perc_xyxy_dict(a, b)

def intersect_perc_xyxy_dict(a, b):
    # perc of b inside a
    areaB = (b['max_x'] - b['min_x']) * (b['max_y'] - b['min_y'])
    xA = max(a['min_x'], b['min_x'])
    yA = max(a['min_y'], b['min_y'])
    xB = min(a['max_x'], b['max_x'])
    yB = min(a['max_y'], b['max_y'])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    return inter_area / areaB


def xywh2xyxy(regions: list):
    xyxy = []
    for region in regions:
        xyxy.append({'min_x': region[0], 'min_y': region[1], 'max_x': region[0] + region[2], 'max_y': region[1] + region[3]})
    return xyxy



def ground_truth_coverage(annotations, sub_regions):
    # annotations: VisDroneAnnotations class
    # sub_regions: list [(x, y, w, h), (x, y, w, h)...]
    sub_regions = xywh2xyxy(sub_regions)
    coverages = []
    for annotation in annotations:
        gt_bbox = {'min_x': annotation.bbox_left, 'min_y': annotation.bbox_top, 'max_x': annotation.bbox_right, 'max_y': annotation.bbox_bottom}
        coverage = 0
        for sub_region in sub_regions:
            if contain(sub_region, gt_bbox):
                coverage = 1
            else:
                temp = intersect_perc(gt_bbox, sub_region)
                if coverage < temp:
                    coverage = temp
        coverages.append(coverage)

    return sum(coverages) / len(coverages)



if __name__ == '__main__':
    predicted_boxes = [
        (50, 50, 150, 150),
        (30, 30, 120, 120),
        (90, 80, 200, 200),
        (100, 100, 150, 150),
        (50, 50, 100, 100),
        (60, 60, 170, 170),
        (25, 25, 125, 125),
        (10, 10, 80, 80),
        (75, 75, 175, 175),
        (40, 40, 160, 160)
    ]

    predicted_boxes = [(boxes[0], boxes[1], boxes[2]-boxes[0], boxes[3]-boxes[1])  for boxes in predicted_boxes]

    true_boxes = [
        (70, 70, 160, 160),
        (20, 20, 110, 110),
        (80, 70, 180, 180),
        (110, 110, 160, 160),
        (40, 40, 90, 90),
        (65, 65, 180, 180),
        (30, 30, 130, 130),
        (15, 15, 85, 85),
        (80, 80, 170, 170),
        (35, 35, 150, 150)
    ]
    true_boxes = [(boxes[0], boxes[1], boxes[2]-boxes[0], boxes[3]-boxes[1])  for boxes in true_boxes]

    for pred_box, true_box in zip(predicted_boxes, true_boxes):
        iou_score = iou(pred_box, true_box)
        print(f'{iou_score:.3f}')