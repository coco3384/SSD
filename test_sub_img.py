import os
import glob
from VisDrone import VisDrone

img_list = glob.glob(os.path.join('dataset', 'VisDrone2019-DET-val-large', 'images', '*.jpg'))
annotation_list = glob.glob(os.path.join('dataset', 'VisDrone2019-DET-val-large', 'annotations', '*.txt'))
# img_list = glob.glob(os.path.join('sub_images', '*.jpg'))
# annotation_list = glob.glob(os.path.join('sub_annotations', '*.txt'))
img_list.sort()  
annotation_list.sort()
for img, annotation in zip(img_list[-2:], annotation_list[-2:]):
    print(img)    
    print(annotation)  
    data = VisDrone(img, annotation)
    data.show_annotations()   