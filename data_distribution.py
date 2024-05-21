from VisDrone import VisDrone
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def main(img, annotation):
    data = VisDrone(img, annotation)
    img = data.get_img()
    annotations = data.get_annotations()
    map = np.zeros((100, 100))
    for annotation in annotations:
        bbox_center = np.array((annotation.bbox_left + annotation.bbox_width / 2, annotation.bbox_top + annotation.bbox_height / 2))
        nbbox_center = (bbox_center / img.shape[:2] * 100).astype(int)
        map[nbbox_center[0], nbbox_center[1]] += 1
    
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    plt.pcolormesh(X, Y, map, shading='auto', cmap='inferno')
    plt.colorbar()
    plt.show()




if __name__ == '__main__':
    img = '9999998_00317_d_0000270.jpg'
    annotation = '9999998_00317_d_0000270.txt'
    main(img, annotation)