import cv2
from VisDrone import VisDrone
from eval import ASoSR
from testSS import main


def effectivness_of_sub_regions(score_of_sub_regions):
    total_score = 0
    for score in score_of_sub_regions.values():
        if score > 0:
            total_score += 1
    return total_score, total_score / len(score_of_sub_regions.values())



if __name__ == '__main__':
    img = '9999998_00317_d_0000270.jpg'
    annotations = '9999998_00317_d_0000270.txt'

    data = VisDrone(img, annotations)
    
    img = data.get_img()
    annotations = data.get_annotations()
    tile_size = 640

    sub_regions = set()
    for h_tile in range(int(img.shape[0] / tile_size) + 1):
        for w_tile in range(int(img.shape[1] / tile_size) + 1):
            x = tile_size * w_tile
            y = tile_size * h_tile
            w = tile_size if (w_tile + 1) * tile_size < img.shape[1] else img.shape[1] - w_tile * tile_size
            h = tile_size if (h_tile + 1) * tile_size < img.shape[0] else img.shape[0] - h_tile * tile_size
            sub_regions.add((x, y, w, h))
    
    asosr, score_of_sub_regions = ASoSR(sub_regions, annotations)
    tile = {'score': asosr, 'ind': score_of_sub_regions}

    segment_resize_scale = 0.4
    ss_scale = 500
    prefered_size = 640
    border = 10
    intersect_score_threshold = 0.7

    gt_coverage, sub_regions, asosr_score, time_cost, score_of_sub_regions = main(img, annotations,
        segment_resize_scale=segment_resize_scale,
        ss_scale=ss_scale,
        default_sub_region_size=prefered_size,
        border=border, 
        intersect_score_threshold=intersect_score_threshold,
        vis=False, bbox_size_visbility=False)

    ssd = {'score': asosr_score, 'ind': score_of_sub_regions}
    
    ssd_effective_sub_regoin, ssd_perc = effectivness_of_sub_regions(ssd['ind'])
    tile_effective_sub_regoin, tile_perc = effectivness_of_sub_regions(tile['ind'])
    print('done!')
    # sub_regions: set((x, y, w, h), (x, y, w, h), ...)
    # annotations: visDroneAnnotations
    