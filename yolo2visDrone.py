import os
import glob
from YoloClass import YoloClass
from tqdm import tqdm

def get_basename(path):
    return os.path.basename(path).split('.')[0]

def main():
    dir = os.path.join('dataset', 'VisDrone2019-DET-test-large')
    save_dir = os.path.join(dir, 'baseline_visDrone_labels')
    os.makedirs(save_dir, exist_ok=True)
    image_dir = os.path.join(dir, 'images')
    annotation_dir = os.path.join(dir, 'labels')

    image_list = glob.glob(os.path.join(image_dir, '*.jpg'))
    annotation_list = glob.glob(os.path.join(annotation_dir, '*.txt'))

    image_list.sort()
    annotation_list.sort()

    for image, annotation in tqdm(zip(image_list, annotation_list), total=len(image_list)):
        name = get_basename(annotation)
        save_path = os.path.join(save_dir, f'{name}.txt')
        data = YoloClass(image, annotation)
        data.save2visDrone(save_path=save_path)



if __name__ == '__main__':
    main()