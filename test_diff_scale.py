from selectivesearch.selectivesearch.selectivesearch import selective_search
from VisDrone import VisDrone
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import time


def main():
    img = cv2.imread('9999998_00317_d_0000270.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    print(img.shape)
    img = cv2.resize(img, (int(w/5), int(h/5)))


    img_lbl_list = []
    regions_list = []
    scales = [250, 500]
    for scale in scales:
        print('starting selective search...')
        start_time = time.time()
        # perform selective search
        img_lbl, regions = selective_search(
            img, scale=scale, sigma=0.9, min_size=10, region_pop=True)

        
        end_time = time.time()
        img_lbl_list.append(img_lbl)
        regions_list.append(regions)
        print(f'Time cost for selective search with scale {scale}: ', end_time - start_time)
    
    print()
    candidates_list = []
    for i in range(len(scales)):
        candidates = set()
        for r in regions_list[i]:
            # excluding same rectangle (with different segments)
            if r['rect'] in candidates:
                continue
            # excluding regions smaller than 2000 pixels
            if r['size'] < 1000:
                continue
            # distorted rects
            x, y, w, h = r['rect']
            if w / h > 1.2 or h / w > 1.2:
                continue
            candidates.add(r['rect'])
        candidates_list.append(candidates)

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=len(scales), nrows=2, gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    if len(scales) != 1:
        for i in range(len(scales)):
            ax[0, i].imshow(img)
            ax[0, i].axis('off')
            ax[1, i].imshow(img_lbl_list[i][:, :, -1])
            ax[1, i].axis('off')
            
            for x, y, w, h in candidates_list[i]:
                # print(x, y, w, h)
                rect = mpatches.Rectangle(
                    (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
                ax[0, i].add_patch(rect)
        else:
            ax[0].imshow(img)
            ax[0].axis('off')
            ax[1].imshow(img_lbl_list[i][:, :, -1])
            ax[1].axis('off')

            for x, y, w, h in candidates_list[0]:
                # print(x, y, w, h)
                rect = mpatches.Rectangle(
                    (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
                ax[0, i].add_patch(rect)

        plt.show()


if __name__ == '__main__':
    main()