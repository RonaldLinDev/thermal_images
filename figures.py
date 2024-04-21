import matplotlib.pyplot as plt
from dataset import dataloader as dl
import pandas as pd
from collections import Counter

class grapher:
    
    def __init__(self, dataset: dl) -> None:
        self.annotation_data =  [annotation for image in dataset.get_all() for annotation in image['annotations']]
        self.label_id_to_name = dataset.id_to_label
        self.df = pd.DataFrame(data = self.annotation_data)
        print(self.df)

    def plot_class_spread(self) -> None:        
        counts = Counter(self.df['labels'].apply(lambda x :self.label_id_to_name[int(x)]))
        plt.bar(counts.keys(), counts.values())
        plt.show()

