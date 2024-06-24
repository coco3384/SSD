import os
import glob
import shutil
import random
import yaml
from tqdm import tqdm


def random_split(list, partition=[0.7, 0.2, 0.1]):
    
    random.shuffle(list)
    total_length = len(list)

    part1_length = int(total_length * partition[0])
    part2_length = int(total_length * partition[1])
    
    part1 = list[:part1_length]
    part2 = list[part1_length:part2_length + part1_length]
    part3 = list[part1_length+part2_length:]

    return part1, part2, part3

def get_corresbounding_annotation(partial_img_list, annotations_list):
    if partial_img_list is None:
        return None
    
    baseDir = os.path.dirname(annotations_list[0])
    partial_img_name_list = get_basename(partial_img_list)
    annotations_name_list = get_basename(annotations_list)
    partial_annotation_list = []
    for img_name in partial_img_name_list:
        if img_name in annotations_name_list:
            partial_annotation_list.append(os.path.join(baseDir, img_name + '.txt'))
    return partial_annotation_list

def get_basename(path_list):
    basename_list = []
    for path_name in path_list:
        basename_list.append(os.path.basename(path_name).split('.')[0])
    return basename_list


def copy(name, phase, img_list, annotation_list):
    if img_list is None or annotation_list is None:
        pass

    for img, annotation in tqdm(zip(img_list, annotation_list), total=len(img_list)):
        shutil.copy(img, os.path.join(name, phase, 'images', os.path.basename(img)))
        shutil.copy(annotation, os.path.join(name, phase, 'labels', os.path.basename(annotation)))



def construct_yolov8_database(train, val, test, name):
    os.makedirs(name, exist_ok=True)
    os.makedirs(os.path.join(name, 'train'), exist_ok=True)
    os.makedirs(os.path.join(name, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(name, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(name, 'val'), exist_ok=True)
    os.makedirs(os.path.join(name, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(name, 'val', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(name, 'test'), exist_ok=True)
    os.makedirs(os.path.join(name, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(name, 'test', 'labels'), exist_ok=True)

    data = {
        'names': ['pedestrain', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others', 'others2'],
        'nc': 12,
        'path': f'{name}',
        'test': '../test/images',
        'train': '../train/images',
        'val': '../val/images',
    }      

    with open(os.path.join(name, 'data.yaml'), 'w') as f:
        yaml.dump(data, f)

    for phase, phase_dataset in zip(['train', 'val', 'test'], [train, val, test]):
        print(f'creating {phase} dataset!')
        copy(name, phase, phase_dataset[0], phase_dataset[1])
    



if __name__ == '__main__':
    img_list = glob.glob(os.path.join('dataset', 'large', 'sub_images(large)', '*.jpg'))
    annotations_list = glob.glob(os.path.join('dataset', 'large', 'sub_annotations(large)_yolo', '*.txt'))

    train_img, val_img, test_img = random_split(img_list, partition=[0.8, 0.2, 0])
    train_annotation = get_corresbounding_annotation(train_img, annotations_list)
    val_annotation = get_corresbounding_annotation(val_img, annotations_list)
    test_annotation = get_corresbounding_annotation(test_img, annotations_list)


    construct_yolov8_database(
        [train_img, train_annotation],
        [val_img, val_annotation],
        [test_img, test_annotation],
        'visDrone_L'    
    )


    
    

