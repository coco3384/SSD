import os
import numpy as np
import glob
import cv2

def overlay_with_sub_region(image: np.array, sub_region: str):
    # image: cv2 image (np.array)
    # sub_region: path/to/sub_region
    _, position = get_image_name_and_position(sub_region)
    sub_region = cv2.imread(sub_region)
    height, width = sub_region.shape[:2]
    output = image.copy()
    overlay = image.copy()

    rect_x, rect_y, rect_width, rect_height = position[0], position[1], width, height

    color = (0, 255, 0)
    alpha = 0.3
    cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), color, -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


def get_image_name_and_position(sub_region_name):
    base_name = os.path.basename(sub_region_name)
    position = base_name.split('_')[-2:]

    position[-1], file_extension = position[-1].split('.')

    len_of_underline = 2
    len_of_position_info = len(position[0]) + len(position[1]) + len(file_extension) + 1 + len_of_underline
    name = base_name[:-len_of_position_info]
    return name, [int(p) for p in position]


def time_analysis(file_path):
    with open(file_path, 'r') as f:
        content = f.readlines()

    return sum([float(c) for c in content])

def gt_covergage_analysis(file_path):
    with open(file_path, 'r') as f:
        content = f.readlines()
    
    content = np.array([float(c) for c in content])

    mean = sum(content) / len(content)
    min = np.min(content)
    argmin = np.argmin(content)
    max = np.max(content)


    result = {
        'mean': mean,
        'min': min,
        'argmin': argmin,
        'max': max,
    }
    return result
    


def main():
    # asosr = analysis(file_path=os.path.join('dataset', 'VisDrone2019-DET-test-large-sub-regions', 'total_asosr_score.txt'))
    gt_coverage = gt_covergage_analysis(file_path=os.path.join('dataset', 'VisDrone2019-DET-test-large-sub-regions', 'total_gt_coverage.txt'))
    time = time_analysis(file_path=os.path.join('dataset', 'VisDrone2019-DET-test-large-sub-regions', 'total_time_cost.txt'))
    
    image_list = glob.glob(os.path.join('dataset', 'VisDrone2019-DET-test-large', 'images', '*.jpg'))
    # ground truth
    annotation_list = glob.glob(os.path.join('dataset', 'VisDrone2019-DET-test-large', 'annotations', '*.txt'))

    
    image_list.sort()
    min_gt_coverage_img = cv2.imread(image_list[gt_coverage['argmin']])
    name = os.path.basename(image_list[gt_coverage['argmin']]).split('.')[0]
    sub_regions_list = glob.glob(os.path.join('dataset', 'VisDrone2019-DEt-test-large-sub-regions', 'images', f'{name}*.jpg'))

    for sub_region in sub_regions_list:
        min_gt_coverage_img = overlay_with_sub_region(min_gt_coverage_img, sub_region)
    
    cv2.imshow('the lowest ground truth coverage image', min_gt_coverage_img)
    cv2.waitKey(0)
    print('done!')

if __name__ == '__main__':
    main()