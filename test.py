from dataset import dataloader

a = dataloader(r'example/yolov8/')
print(a.read_pair('0-8_1663598421-8378496_jpg.rf.3c639b32119f9f2d263d4bd359838f3d.jpg', 'train'))