import os
import yaml

if __name__ == '__main__':
    des_dir = os.path.join('dataset', 'VisDrone2019-DET-medium-sub-regions')

    data = {
    'names': ['ignore regions', 'pedestrain', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'],
    'nc': 12,
    'path': '/content/',
    'test': './test/images',
    'train': './train/images',
    'val': './val/images',
    }  
        
    with open(os.path.join(des_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data, f)