import VisDroneAnnotation
from utils import intersect_perc, contain

# Average Score of Sub-Region
def ASoSR(sub_regions, annotations):
    # sub_regions: set((x, y, w, h), (x, y, w, h), ...)
    # annotations: visDroneAnnotations
    # return :
    #       asosr: avergage score of asosr
    #       score_of_sub_regions: score of each sub-region
    score_of_sub_regions = {}
    for annotation in annotations:
        annotation_bbox = (annotation.bbox_left, annotation.bbox_top, annotation.bbox_width, annotation.bbox_height)
        for sub_region in sub_regions:
            if sub_region not in score_of_sub_regions.keys():
                score_of_sub_regions[sub_region] = 0

            if contain(sub_region, annotation_bbox):
                score_of_sub_regions[sub_region] = score_of_sub_regions[sub_region] + 1

    if len(score_of_sub_regions) != 0:
        asosr = sum(score_of_sub_regions.values()) / len(score_of_sub_regions)
    else:
        asosr = 0
    
    return asosr, score_of_sub_regions



