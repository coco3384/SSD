from selectivesearch_Detect.selectivesearch.selectivesearch import selective_search
from VisDrone import VisDrone
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import time
import numpy as np
from utils import iou, contain, intersect_perc, crop_distorted_region, crop_from_candidates, candidates_resize
from eval import ASoSR
from utils import ground_truth_coverage
import os
import glob


def define_max_region_size(annotations, scale=10):
    x = []
    y = []
    
    for annotation in annotations:
        x.append(annotation.bbox_width)
        y.append(annotation.bbox_height)

    p50_width = np.percentile(x, 50)
    p50_height = np.percentile(y, 50)
    max_region_size = (scale * p50_width) *  (scale * p50_height)

    return max_region_size


def focus(img, annotations, segment_resize_scale=0.4, ss_scale=500, default_sub_region_size=640, intersect_score_threshold=0.7, border=10, vis=False, bbox_size_visbility=False):
    # img           : path/to/img
    # annotation    : path/to/annotation

    visDrone = VisDrone(img, annotations)

    # region_size_define
    deafault_sub_region_rescale_size = (default_sub_region_size * segment_resize_scale)**2
    # max_region_size = define_max_region_size(annotations, scale=20)

    resize_shape = (int(visDrone.shape[1] * segment_resize_scale), int(visDrone.shape[0] * segment_resize_scale))
    
    threshold = deafault_sub_region_rescale_size / 2

    visDrone_resize = visDrone.resize(resize_shape)

    img = visDrone_resize.get_img()
    annotations = visDrone_resize.get_annotations()

    h, w = img.shape[:2]
    

    print('starting selective search...')
    start_time = time.time()

    # perform selective search
    img_lbl, regions, og_regions = selective_search(
        img, scale=ss_scale, sigma=0.9, min_size=10, region_pop=True, max_region_size=deafault_sub_region_rescale_size, border=border)
    
    end_time = time.time()



    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        
        # excluding regions smaller than threshold pixels
        if r['bbox_size'] < threshold:
            continue
        
        # check if region is a sub-region of one of the candidate
        sub_region = False
        
        for candidate in candidates:
            intersect_score = intersect_perc(candidate, r['rect'])
            if intersect_score > intersect_score_threshold:
                sub_region = True
        
        if sub_region:
            continue
        
        candidates.add(r['rect'])
    
    candidates = crop_distorted_region(candidates, default_sub_region_size * segment_resize_scale)
    gt_coverage = ground_truth_coverage(annotations, candidates)

    """
    fig, ax = plt.subplots(ncols=1, nrows=2, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(img_lbl[:, :, -1])
    ax[1].axis('off')

    for annotation in annotations:
        rect = mpatches.Rectangle(
            (annotation.bbox_left, annotation.bbox_top), annotation.bbox_width, annotation.bbox_height, fill=False, edgecolor='blue', linewidth=1)
        ax[0].add_patch(rect)
    """

    segmentation_img = img_lbl[:, :, -1].copy()
    segmentation_img = segmentation_img / max(segmentation_img.flatten())
    annotation_img = img.copy()

    ori_img = visDrone.get_img()

    ori_annotations = visDrone.get_annotations()
    
    resize_candidates = candidates_resize(candidates, 1 / segment_resize_scale, ori_img.shape[:2])
    sub_regions = crop_from_candidates(resize_candidates, ori_annotations, ori_img)

    # draw annotation
    for annotation in annotations:
        annotation_img = cv2.rectangle(annotation_img, (annotation.bbox_left, annotation.bbox_top), (annotation.bbox_right, annotation.bbox_bottom), color=(255, 0, 0), thickness=1)
        segmentation_img = cv2.rectangle(segmentation_img, (annotation.bbox_left, annotation.bbox_top), (annotation.bbox_right, annotation.bbox_bottom), color=(255, 0, 0), thickness=1)

    origin_img = annotation_img.copy()

    """
    for i in range(len(candidates)):
        if (i+1) % 20 == 0:
            # show
            cv2.imshow('test', rect_img)
            cv2.imshow('test2', rect_seg_img)
            cv2.waitKey(0)

            # reset image
            origin_img = annotation_img.copy()
            og_segmentation_img = segmentation_img.copy()
       
        # draw sub_region
        x, y, w, h = list(candidates)[i]
        rect_img = cv2.rectangle(origin_img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=1)
        rect_seg_img = cv2.rectangle(og_segmentation_img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=1)
    
    cv2.imshow('test', rect_img)
    cv2.imshow('test2', rect_seg_img)
    cv2.waitKey(0)
    """
    """
    for x, y, w, h in candidates:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax[0].add_patch(rect)
        
    for x, y, w, h in candidates:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='white', linewidth=1)
        ax[1].add_patch(rect)
    plt.show()
    """
    
    asosr_score, asosr = ASoSR(candidates, annotations)
    time_cost = end_time - start_time
    print('Groud Truth Coverage:', gt_coverage)
    print('numbers of candidates:', len(candidates))
    print(f'Average Score of Sub Region: {asosr_score:.2f}')
    print('Time Cost:', time_cost)

    if vis:
        sorted_asosr = dict(sorted(asosr.items(), key=lambda item: item[1]))
        for bbox, i in sorted_asosr.items():       
            # draw sub_region
            x, y, w, h = bbox
            origin_img = annotation_img.copy()
            rect_img = cv2.rectangle(origin_img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=1)
            cv2.imshow('test', rect_img)
            cv2.waitKey(0)
    
    widths = []
    heights = []

    for candidate in candidates:
        widths.append(candidate[2])
        heights.append(candidate[3])

    if bbox_size_visbility:
        plt.clf()
        plt.scatter(widths, heights)
        x_line = np.linspace(0.5 * min(widths), max(widths) * 2, num=100)
        y_line = deafault_sub_region_rescale_size / x_line
        y_line_two_times = deafault_sub_region_rescale_size * 2 / x_line
        
        plt.axvline(x = prefered_size * segment_resize_scale, color='gray', linestyle='--', label='prefered sub region width')
        plt.axhline(y = prefered_size * segment_resize_scale, color='gray', linestyle='--', label='prefered sub region height')
        plt.plot(x_line, y_line, color='green', linestyle='--', label='prefered sub region size')
        plt.plot(x_line, y_line_two_times, color='red', linestyle='--', label='2 * prefered sub region size')

        plt.xlim(0.5 * min(widths), max(widths) * 2)
        plt.ylim(0.5 * min(heights), max(widths) * 2)
        plt.legend()
        plt.plot()
        plt.savefig(f'bbox_scatter.png')
    

    return gt_coverage, sub_regions, asosr_score, time_cost, asosr
    print('done!')



if __name__ == '__main__':
    img = '9999998_00317_d_0000270.jpg'
    annotations = '9999998_00317_d_0000270.txt'
    

    # (2000, 1500)
    # for segment resize scale 0.4 is the best
    # segment_resize_scales = [0.2, 0.4, 0.6, 0.8, 1]
    segment_resize_scale = 0.4
    # for ss_scale 500 is the best
    # ss_scales = [100, 200, 300, 400, 500]
    ss_scale = 500
    # for prefered_size 1024 is the best
    # prefered_sizes = [224, 416, 512, 640, 1024]
    prefered_size = 640
    # for border 50 is the best
    # borders = [10]
    border = 10
    intersect_score_threshold = 0.7

    gt_coverage, sub_regions, asosr_score, time_cost, score_of_sub_regions = focus(img, annotations,
        segment_resize_scale=segment_resize_scale,
        ss_scale=ss_scale,
        default_sub_region_size=prefered_size,
        border=border, 
        intersect_score_threshold=intersect_score_threshold,
        vis=False, bbox_size_visbility=False)
    
    print('done!')

    # save sub regions
    """
    name = os.path.basename(img).split('.')[0]
    sub_regions_img_dir = 'sub_images'
    sub_regions_annotation_dir = 'sub_annotations'
    os.makedirs(sub_regions_img_dir, exist_ok=True)
    os.makedirs(sub_regions_annotation_dir, exist_ok=True)

    for xy, sub_region in list(sub_regions.items()):
        img_path = os.path.join(sub_regions_img_dir, name + f'_{xy[0]}_{xy[1]}.jpg')
        annotation_path = os.path.join(sub_regions_annotation_dir, name + f'_{xy[0]}_{xy[1]}.txt')
        sub_region.save(img_path=img_path, annotation_path=annotation_path)
    """