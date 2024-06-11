import os
import glob
import cv2

def get_image_name_and_position(sub_region_name):
    base_name = os.path.basename(sub_region_name)
    position = base_name.split('_')[-2:]

    position[-1], file_extension = position[-1].split('.')

    len_of_underline = 2
    len_of_position_info = len(position[0]) + len(position[1]) + len(file_extension) + 1 + len_of_underline
    name = base_name[:-len_of_position_info]
    return name, [int(p) for p in position]


if __name__ == '__main__':
    base_image = cv2.imread(os.path.join('dataset', 'VisDrone2019-DET-test-large', 'images', '0000073_00377_d_0000001.jpg'))
    annotation_list = glob.glob(os.path.join('dataset', 'VisDrone2019-DET-test-large-sub-regions', 'labels', '*.txt'))
    result_list = glob.glob(os.path.join('test', '*.jpg'))
    annotation_list.sort()
    result_list.sort()

    for annotation, result in zip(annotation_list[:17], result_list[:17]):
        name, position = get_image_name_and_position(annotation)
        
        print(name, position)
        sub_region_image = cv2.imread(result)
        h, w = sub_region_image.shape[:2]
        base_image[position[1]:position[1]+h, position[0]:position[0]+w, :] = sub_region_image
        cv2.imshow('test', base_image)
        cv2.waitKey(0)    


