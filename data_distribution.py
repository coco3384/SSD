from VisDrone import VisDrone
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm


def distribute_bbox(img_list, annotation_list):
    map = np.zeros((100, 100))
    for img, annotation in tqdm.tqdm(zip(img_list, annotation_list)):
        data = VisDrone(img, annotation)
        img = data.get_img()
        annotations = data.get_annotations()
        
        for annotation in annotations:
            img_xy = (img.shape[1], img.shape[0])
            nbbox_left = int(annotation.bbox_left / img_xy[0] * 100)
            nbbox_right = int(annotation.bbox_right / img_xy[0] * 100)
            nbbox_top = int(annotation.bbox_top / img_xy[1] * 100)
            nbbox_bottom = int(annotation.bbox_bottom / img_xy[1] * 100)
            map[nbbox_top:nbbox_bottom, nbbox_left:nbbox_right] += 1
    
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    plt.pcolormesh(X, Y, map, shading='auto', cmap='inferno')
    plt.colorbar()
    plt.show()


def distribute_bbox_center(img_list, annotation_list):
    map = np.zeros((100, 100))
    for img, annotation in tqdm.tdqm(zip(img_list, annotation_list)):
        data = VisDrone(img, annotation)
        img = data.get_img()
        annotations = data.get_annotations()
        
        for annotation in annotations:
            bbox_center = np.array((annotation.bbox_left + annotation.bbox_width / 2, annotation.bbox_top + annotation.bbox_height / 2))
            img_xy = (img.shape[1], img.shape[0])
            nbbox_center = (bbox_center / img_xy * 100).astype(int)
            map[nbbox_center[1], nbbox_center[0]] += 1
    
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    plt.pcolormesh(X, Y, map, shading='auto', cmap='inferno')
    plt.colorbar()
    plt.show()


def main():
    img_list = glob.glob(os.path.join('VisDrone2019-DET-train', 'images', '*.jpg'))
    annotation_list = glob.glob(os.path.join('VisDrone2019-DET-train', 'annotations', '*.txt'))
    img_list.sort()
    annotation_list.sort()
    distribute_bbox(img_list, annotation_list)




if __name__ == '__main__':
    main()