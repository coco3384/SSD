import os
import glob
from tqdm import tqdm
from VisDrone import VisDrone


def get_image_name_and_position(sub_region_name):
    base_name = os.path.basename(sub_region_name)
    position = base_name.split('_')[-2:]
    len_of_txt = 4
    len_of_underline = 2
    len_of_position_info = len(position[0]) + len(position[1]) + len_of_txt + len_of_underline
    name = base_name[:-len_of_position_info]
    return name, position



def main():
    annotation_list = glob.glob(os.path.join('dataset', 'large', 'visDrone2019-DET-test-large-sub-regions-predict', '*.txt'))
    origin_image_list = glob.glob(os.path.join('dataset', 'large', 'visDrone2019-DET-test-large', 'images', '*.jpg'))
    for annotation in annotation_list[:1]:
        name, position = get_image_name_and_position(annotation)
        

if __name__ == '__main__':
    main()
