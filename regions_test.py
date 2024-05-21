import matplotlib.pyplot as plt

class RegionTest():
    def __init__(self, regions, img, annotation, threshold):
        self.regions = regions
        self.img = img
        self.annotation = annotation
        self.threshold = threshold
    
    def __len__(self):
        return len(self.regions)

    def show_size_distribution_of_region(self):
        widths = []
        heights = []
        for region in self.regions:
            for x, y, w, h in region['rect']:
                widths.append(w)
                heights.append(h)
        size = widths * heights
        plt.scatter(widths, heights, c=size, vmin=self.threshold)
        plt.xlabel('width')
        plt.ylabel('height')
        plt.show()
        
            
