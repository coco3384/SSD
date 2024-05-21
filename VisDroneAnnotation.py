import numpy as np
from typing import Tuple, List

class VisDroneAnnotation():
    # names:
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

    def __init__(self, annotation):
        if isinstance(annotation, str):
            bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion = annotation.rstrip().split(',')
        else:
            bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion = annotation
            
        self.bbox_left = int(bbox_left)
        self.bbox_top = int(bbox_top)
        self.bbox_width = int(bbox_width)
        self.bbox_height = int(bbox_height)
        self.bbox_right = self.bbox_left + self.bbox_width
        self.bbox_bottom = self.bbox_top + self.bbox_height
        self.bbox_size = self.bbox_width * self.bbox_height
        self.score = float(score)
        self.category = int(category)
        self.truncation = truncation
        self.occlusion = occlusion

    def resize(self, wh:Tuple):
        bbox_left = self.bbox_left * wh[0]
        bbox_top = self.bbox_top * wh[1]
        bbox_width = self.bbox_width * wh[0]
        bbox_height = self.bbox_height * wh[1]
        score = self.score
        category = self.category
        truncation = self.truncation
        occlusion = self.occlusion
        return VisDroneAnnotation(annotation=[bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion])

    def category_name(self):
        if self.category == 0:
            return 'predestrain'
        if self.category == 1:
            return 'people'
        if self.category == 2:
            return 'bicycle'
        if self.category == 3:
            return 'car'
        if self.category == 4:
            return 'van'
        if self.category == 5:
            return 'truck'
        if self.category == 6:
            return 'tricycle'
        if self.category == 7:
            return 'awning-tricycle'
        if self.category == 8:
            return 'bus'
        else:
            return 'motor'

    def __str__(self):
        info = {
            'box': {
                'xyxy': np.array([self.bbox_left, self.bbox_top, self.bbox_right, self.bbox_bottom]),
                'xywh': np.array([self.bbox_left, self.bbox_top, self.bbox_width, self.bbox_height])
            },
            'category': self.category,
            'truncation': self.truncation,
            'occlusion': self.occlusion,
        }
        return f'{info}'
    
    def save_txt(self, annotation_path):
        with open(annotation_path, 'a') as f:
            f.write(f'{self.bbox_left},{self.bbox_top},{self.bbox_width},{self.bbox_height},{self.score},{self.category},{self.truncation},{self.occlusion}\n')

class VisDroneAnnotations():
    def __init__(self, annotations):
        self.current = 0
        # annotations from .txt readlines
        if len(annotations) != 0:
            if isinstance(annotations[0], str):
                self.annotations = self.__txt2class(annotations)

            # annotations from visDroneAnnotations
            if isinstance(annotations[0], VisDroneAnnotation):
                self.annotations = annotations
        else:
            self.annotations = []
    
    def mean_size(self):
        total = 0
        for annotation in self.annotations:
            total = total + annotation.bbox_size
        return total / len(self)

    def __txt2class(self, annotations):
        visDroneAnnotations = []
        for annotation in annotations:
            a = VisDroneAnnotation(annotation)
            visDroneAnnotations.append(a)
        return visDroneAnnotations
    
    def resize(self, wh:tuple):
        visDroneAnnotations = []
        print(wh)
        for annotation in self.annotations:
            visDroneAnnotations.append(annotation.resize(wh))
        return VisDroneAnnotations(visDroneAnnotations)
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, key):
        return self.annotations[key]
    
    def __str__(self):
        info = {}
        for annotation in self.annotations:
            cls = annotation.category
            if cls not in info.keys():
                info[cls] = 1
            else:
                info[cls] = info[cls] + 1

        return f'{info}'
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self):
            self.current = 0
            raise StopIteration
        else:
            num = self.current
            self.current += 1
            return self.annotations[num]
    
if __name__ == '__main__':
    with open('9999998_00317_d_0000270.txt', 'r') as f:
        texts = f.readlines()
    
    visDroneAnnotation = VisDroneAnnotations(texts)
    visDroneAnnotation = visDroneAnnotation.resize((1/5, 1/5))
    print(visDroneAnnotation)
    for a in visDroneAnnotation:
        print(a)

        