import os
import glob
from YoloClass import YoloClass
import cv2
import numpy as np

def get_basename(path):
    return os.path.basename(path).split('.')[0]


if __name__ == '__main__':
    annotation_list = glob.glob(os.path.join('dataset', 'VisDrone2019-DET-test-large', 'labels_subRegion_yolo', '*.txt'))
    predict_annotation_list = glob.glob(os.path.join('dataset', 'VisDrone2019-DET-test-large', 'labels_noSubRegion_yolo', '*.txt'))
    image_list = glob.glob(os.path.join('dataset', 'VisDrone2019-DET-test-large', 'images', '*.jpg'))
    annotation_list.sort()
    predict_annotation_list.sort()
    image_list.sort()

    for i, (annotation, annotation2) in enumerate(zip(annotation_list[:], predict_annotation_list[:])):
        name = get_basename(annotation)
        # img = os.path.join('dataset', 'VisDrone2019-DET-test-large-sub-regions', 'images', f'{name}.jpg')
        img = image_list[i] 
        data = YoloClass(image=img, annotations=annotation)
        data.set_line_width(2)
        data = data.show_annotations(show=False)
        # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        data2 = YoloClass(image=img, annotations=annotation2)
        data2.set_line_width(2)
        data2 = data2.show_annotations(show=False)
        # data2 = cv2.cvtColor(data2, cv2.COLOR_BGR2RGB)

        c = np.hstack((data, data2))
        cv2.imshow('test', c)
        cv2.waitKey(0)