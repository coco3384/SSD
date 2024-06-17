import os
import glob
from modified_selective_search import focus
from tqdm import tqdm


def write_result(path, content):
    with open(path, 'w') as f:
        for el in content:
            f.write(str(el) + '\n')


def main():
    segment_resize_scale = 0.4
    # for ss_scale 500 is the best
    # ss_scales = [100, 200, 300, 400, 500]
    ss_scale = 300
    # for prefered_size 1024 is the best
    # prefered_sizes = [224, 412, 512, 640, 1024]
    prefered_size = 640
    # for border 50 is the best
    # borders = [10]
    border = 10
    intersect_score_threshold = 0.7
    phase = 'VisDrone2019-DET-train-large'

    
    sub_regions_img_dir = os.path.join('dataset', f'{phase}4-sub-regions', 'images')
    sub_regions_annotation_dir = os.path.join('dataset', f'{phase}5-sub-regions', 'annotations')
    os.makedirs(sub_regions_img_dir, exist_ok=True)
    os.makedirs(sub_regions_annotation_dir, exist_ok=True)
    
    total_gt_coverage = []
    total_asosr_score = []
    total_time_cost = []
    
    img_list = glob.glob(os.path.join('dataset', phase, 'images', '*.jpg'))
    annotation_list = glob.glob(os.path.join('dataset', phase, 'annotations', '*.txt'))
    img_list.sort()
    annotation_list.sort()
    # img_list = ['0000073_00377_d_0000001.jpg']
    # annotation_list = ['0000073_00377_d_0000001.txt']

    for img, annotation in tqdm(zip(img_list, annotation_list), total=len(img_list)):
        try:
            gt_coverage, sub_regions, asosr_score, time_cost, score_of_sub_regions = focus(img, annotation,
                segment_resize_scale=segment_resize_scale,
                ss_scale=ss_scale,
                default_sub_region_size=prefered_size,
                border=border, 
                intersect_score_threshold=intersect_score_threshold,
                vis=False, bbox_size_visbility=False)
            
            total_gt_coverage.append(gt_coverage)
            total_asosr_score.append(asosr_score)
            total_time_cost.append(time_cost)
            name = os.path.basename(img).split('.')[0]

            for xy, sub_region in list(sub_regions.items()):
                img_path = os.path.join(sub_regions_img_dir, name + f'_{xy[0]}_{xy[1]}.jpg')
                annotation_path = os.path.join(sub_regions_annotation_dir, name + f'_{xy[0]}_{xy[1]}.txt')
                sub_region.save(img_path=img_path, annotation_path=annotation_path)
        except:
            write_result(os.path.join('dataset', f'{phase}5-sub-regions', 'total_gt_coverage.txt'), total_gt_coverage)
            write_result(os.path.join('dataset', f'{phase}5-sub-regions', 'total_asosr_score.txt'), total_asosr_score)
            write_result(os.path.join('dataset', f'{phase}5-sub-regions', 'total_time_cost.txt'), total_time_cost)
            print('accidently quit!')

    
    write_result(os.path.join('dataset', f'{phase}5-sub-regions', 'total_gt_coverage.txt'), total_gt_coverage)
    write_result(os.path.join('dataset', f'{phase}5-sub-regions', 'total_asosr_score.txt'), total_asosr_score)
    write_result(os.path.join('dataset', f'{phase}5-sub-regions', 'total_time_cost.txt'), total_time_cost)



if __name__ == '__main__':
    main()