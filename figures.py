import matplotlib.pyplot as plt
from dataset import dataloader as dl
import pandas as pd
from collections import Counter

class grapher:
    
    def __init__(self, dataset: dl) -> None:
        self.annotation_data =  [annotation for image in dataset.get_all() for annotation in image['annotations']]
        self.image_data = dataset.get_all()
        self.label_id_to_name = dataset.id_to_label
        self.df = pd.DataFrame(data = self.annotation_data)

    def plot_class_spread(self) -> None:        
        counts = Counter(self.df['labels'].apply(lambda x :self.label_id_to_name[int(x)]))
        plt.bar(counts.keys(), counts.values())
        plt.show()
        plt.close()

    def plot_bounding_box(self) -> None:
        areas = []
        for row in self.df['bounding_box']:
            areas.append(self.area(*row))
        plt.hist(areas)
        plt.show()
        plt.close()        


    def area(self, x, y, w, h):
        return float(w) * float(h)
    
    def plot_occlusion(self) -> None:
        num_labels_per_image = []
        for image in self.image_data:
           num_labels_per_image.append(len(image['annotations']))
        plt.hist(num_labels_per_image)
        plt.show()
