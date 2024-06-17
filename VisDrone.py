import cv2
import numpy as np
import VisDroneAnnotation
from typing import Tuple


class VisDrone:
    def __init__(self, image, annotations, name=None, line_width=2):
        # annotations: plent read from VisDrone Annotations
        if isinstance(annotations, str):
            with open(annotations, 'r') as f:
                content = f.readlines()
        # annotations: from resize
        else:
            content = annotations
        
        if isinstance(image, str):
            self.im = cv2.imread(image)
        # annotations: from resize
        else:
            self.im = image
            

        if isinstance(image, str):
            self.name = image.split('.')[0]
        # annotations: from resize
        else:
            self.name = name
        
        self.annotations = VisDroneAnnotation.VisDroneAnnotations(content)
        self.line_width = line_width
        self.shape = self.im.shape
        self.randomseed = 0
        np.random.seed(self.randomseed)

    def resize(self, wh: Tuple[int, int]):
        resize_im = cv2.resize(self.im, wh)
        wh_scale = (wh[0] / self.im.shape[1], wh[1] / self.im.shape[0])
        resize_annotations = self.annotations.resize(wh_scale)
        return VisDrone(resize_im, resize_annotations, name=self.name, line_width=1)


    def get_img(self):
        return self.im
    
    def get_annotations(self):
        return self.annotations

    def __draw_annot(self):
        num_colors = 20
        colors = np.random.randint(0, 256, (num_colors, 3))
        baseImage = self.im.copy()
        for annotation in self.annotations:
            cv2.rectangle(baseImage, 
                        (annotation.bbox_left, annotation.bbox_top),
                        (annotation.bbox_right, annotation.bbox_bottom),
                        colors[int(annotation.category)].tolist(), self.line_width)
            text = str(annotation.category)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (255, 255, 255)
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = annotation.bbox_left + annotation.bbox_width // 2 - text_size[0] // 2
            text_y = annotation.bbox_top - 10  # 10 pixels above the top of the bbox

            # 如果文本位置超出图像上边界，可以调整位置
            if text_y < 0:
                text_y = annotation.bbox_top + annotation.bbox_height + 20  # Place the text below the bbox if not enough space above

            cv2.putText(baseImage, text, (text_x, text_y), font, font_scale, color, thickness)

        return baseImage
    
    def __draw_annot_onebyone(self, i):
        num_colors = 20
        colors = np.random.randint(0, 256, (num_colors, 3))
        annotation = self.annotations[i]
        baseImage = self.im.copy()
        
        cv2.rectangle(baseImage, 
                    (annotation.bbox_left, annotation.bbox_top),
                    (annotation.bbox_right, annotation.bbox_bottom),
                    colors[int(annotation.category)].tolist(), self.line_width)
        
        print(annotation.category_name())
        """
        cv2.rectangle(baseImage,
                    (annotation.bbox_left, annotation.bbox_top - 20),
                    (annotation.bbox_left + int(annotation.bbox_width), annotation.bbox_top),
                    colors[int(annotation.category)].tolist(), -1
                    )
        cv2.putText(baseImage, annotation.category_name(), 
                    (annotation.bbox_left, annotation.bbox_top - 10),
                    4, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    (0, 0, 0), 
                    2
                    )
        """
        return baseImage

    def __len__(self):
        return 1
    
    def __number_of_anno__(self):
        return len(self.annotations)

    def show(self):
        cv2.imshow('origin image', self.im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
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

    def _save_txt(self, annotation_path):
        # create file
        with open(annotation_path, 'w') as f:
            f.write('')
        
        # write annotations
        if len(self.annotations) != 0:
            for annotation in self.annotations:
                annotation.save_txt(annotation_path)

    def _save_img(self, img_path):
        cv2.imwrite(img_path, self.im)

    def save(self, img_path, annotation_path):
        self._save_img(img_path)
        self._save_txt(annotation_path)



    
if __name__ == '__main__':
    img = '9999998_00317_d_0000270.jpg' 
    annotations = '9999998_00317_d_0000270.txt'

    visDrone = VisDrone(img, annotations)
    print(visDrone.im.shape)
    visDrone.show_annotations()

    resize = visDrone.resize((400, 300))
    resize.show_annotations()
    