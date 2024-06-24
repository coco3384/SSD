from math import ceil
from VisDrone import VisDrone
from utils import crop_from_candidates, ground_truth_coverage
from eval import ASoSR
import os
import glob
from tqdm import tqdm
from focus import write_result


if __name__ == '__main__':

    phase = 'VisDrone2019-DET-test-large'
    
    sub_regions_img_dir = os.path.join('dataset', f'{phase}-tile-sub-regions', 'images')
    sub_regions_annotation_dir = os.path.join('dataset', f'{phase}-tile-sub-regions', 'annotations')
    
    os.makedirs(sub_regions_img_dir, exist_ok=True)
    os.makedirs(sub_regions_annotation_dir, exist_ok=True)
    
    total_gt_coverage = []
    total_asosr_score = []
    
    img_list = glob.glob(os.path.join('dataset', phase, 'images', '*.jpg'))
    annotation_list = glob.glob(os.path.join('dataset', phase, 'annotations', '*.txt'))
    img_list.sort()
    annotation_list.sort()

    for img, annotations in tqdm(zip(img_list, annotation_list), total=len(img_list)):

        data = VisDrone(img, annotations)  
        name = os.path.basename(img).split('.')[0]

        img = data.get_img()
        annotations = data.get_annotations()
        
        tile_size = 640

        candidates = set()
        for h_tile in range(ceil(img.shape[0] / tile_size)):
            for w_tile in range(ceil(img.shape[1] / tile_size)):
                x = tile_size * w_tile
                y = tile_size * h_tile
                w = tile_size if (w_tile + 1) * tile_size <= img.shape[1] else img.shape[1] - w_tile * tile_size
                h = tile_size if (h_tile + 1) * tile_size <= img.shape[0] else img.shape[0] - h_tile * tile_size
                candidates.add((x, y, w, h))

        sub_regions = crop_from_candidates(candidates, annotations, img)
        asosr, score_of_sub_regions = ASoSR(candidates, annotations)
        gt_coverage = ground_truth_coverage(annotations, candidates)

        total_asosr_score.append(asosr)
        total_gt_coverage.append(gt_coverage)
    
        for xy, sub_region in list(sub_regions.items()):
            img_path = os.path.join(sub_regions_img_dir, name + f'_{xy[0]}_{xy[1]}.jpg')
            annotation_path = os.path.join(sub_regions_annotation_dir, name + f'_{xy[0]}_{xy[1]}.txt')
            sub_region.save(img_path=img_path, annotation_path=annotation_path)

        write_result(os.path.join('dataset', f'{phase}-tile-sub-regions', 'total_gt_coverage.txt'), total_gt_coverage)
        write_result(os.path.join('dataset', f'{phase}-tile-sub-regions', 'total_asosr_score.txt'), total_asosr_score)
        

        