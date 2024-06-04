import cv2
import numpy as np


class YoloAnnotation:
    def __init__(self, content):
        if len(content.split()) == 6:
            self.category, self.x_center, self.y_center, self.bbox_width, self.bbox_height, self.conf = [float(el) for el in content.split()]
        else:
            self.category, self.x_center, self.y_center, self.bbox_width, self.bbox_height = [float(el) for el in content.split()]
        self.bbox_top = self.y_center - (self.bbox_height / 2)
        self.bbox_left = self.x_center - (self.bbox_width / 2)
        self.bbox_bottom = self.y_center + (self.bbox_height / 2)
        self.bbox_right = self.x_center + (self.bbox_width / 2)


    def get_category(self):
        return self.category
    
    def get_bbox_xyxy(self, w, h):
        return {
            'bbox_top': int(self.bbox_top * h),
            'bbox_left': int(self.bbox_left * w),
            'bbox_bottom': int(self.bbox_bottom * h),
            'bbox_right': int(self.bbox_right * w)
            }
    
    def get_bbox_xywh(self, w, h):
        return {
            'x_center': int(self.x_center * w),
            'y_center': int(self.y_center * h),
            'bbox_width': int(self.bbox_width * w),
            'bbox_height': int(self.bbox_height * h)
        }

    def get_conf(self):
        return self.conf
    

class YoloAnnotations:
    def __init__(self, annotations):
        self.current = 0
        self.annotations = [YoloAnnotation(content) for content in annotations] if len(annotations) != 0 else []
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, key):
        return self.annotations[key]
    
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
        

class YoloClass:
    def __init__(self, image, annotations, line_width=1):
        with open(annotations, 'r') as f:
            annotations = f.readlines()

        self.annotations = YoloAnnotations(annotations)
        self.img = cv2.imread(image)
        self.line_width = line_width


    def set_line_width(self, line_width):
        self.line_width = line_width

    def get_img(self):
        return self.img

    def __draw_annot(self):
        num_colors = 20
        np.random.seed(seed=0)
        colors = np.random.randint(0, 256, (num_colors, 3))
        baseImage = self.img.copy()
        for annotation in self.annotations:
            bbox = annotation.get_bbox_xyxy(w=self.img.shape[1], h=self.img.shape[0])
            # bbox = annotation.get_bbox_xyxy(w=1, h=1)
            cv2.rectangle(baseImage, 
                        (bbox['bbox_left'], bbox['bbox_top']),
                        (bbox['bbox_right'], bbox['bbox_bottom']),
                        colors[int(annotation.category)].tolist(), self.line_width)
        return baseImage


    def __draw_annotations_onebyone(self):
        pass

    def show_annotations(self, show=True):
        img_with_annot = self.__draw_annot()
        if show:
            cv2.imshow('image with annotations', img_with_annot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img_with_annot
    
    def show_annotaions_onebyone(self):
        for i in range(len(self.annotations)):
            img_with_annot = self.__draw_annot_onebyone(i)
            cv2.imshow('image with annotations', img_with_annot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


