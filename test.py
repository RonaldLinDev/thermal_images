from dataset import dataloader
from figures import grapher
# from autodistill_grounded_sam import GroundedSAM
# from autodistill.detection import CaptionOntology
# from autodistill_yolov8 import YOLOv8

a = dataloader(r'test/')
b = dataloader(r'test2/')
b.move_split('train', a)
# a.distill_image(r'0-8_1663569550-11585_jpg.rf.18f12e026fb5b66ef6545aa878c5c612.jpg', split='train', missing=['person'], confidence_threshold= 0, prompt='thermal images of ')

