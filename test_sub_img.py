import os
import glob
from VisDrone import VisDrone

img_list = glob.glob(os.path.join('dataset', 'large', 'sub_images_train(large)', '*.jpg'))
annotation_list = glob.glob(os.path.join('dataset', 'large', 'sub_annotations_train(large)', '*.txt'))
# img_list = glob.glob(os.path.join('sub_images', '*.jpg'))
# annotation_list = glob.glob(os.path.join('sub_annotations', '*.txt'))
img_list.sort()  
annotation_list.sort()
for img, annotation in zip(img_list[100:120], annotation_list[100:120]):
    print(img)  
    print(annotation) 
    data = VisDrone(img, annotation)
    data.show_annotations()   