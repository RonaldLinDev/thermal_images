from dataset import dataloader
from figures import grapher
a = dataloader(r'example/')
b = grapher(a)
b.plot_class_spread()