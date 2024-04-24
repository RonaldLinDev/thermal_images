import yaml
import os
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
import cv2

class dataloader:

    def __init__(self, path: str) -> None:
        prev = os.getcwd()
        self.path = path
        try:
            os.chdir(self.path)
            with open('data.yaml', 'r') as stream:
                self.info = yaml.safe_load(stream)
                self.id_to_label = {i : label for i, label in enumerate(self.info['names'])}
            os.chdir(prev)
        except yaml.YAMLError as e:
            print('error loading data, make sure your data.yaml folder is unchanged', e)
        except (OSError, IOError) as e:
            print('cant find path specified', e)
    


    # note file_name has no filetype
    def read_pair(self, image_name: str, split: str) -> dict:
        prev = os.getcwd()
        ret = dict()
        ret['annotations'] = []
        try:
            os.chdir(self.path)
            ret['image_path'] = self.info[split][3:] + image_name
            # gets rid of the ../ that os.chdir doesnt like
            
            
            with open(self.get_label_path(split, image_name)) as f:
                for line in f:
                    tokens = line.split()
                    ret['annotations'].append({'labels': tokens[0], 'bounding_box': tokens[1:]})    
            os.chdir(prev)
        except (OSError, IOError) as e:
            print('couldnt find file')
        except KeyError as e:
            print('couldnt find that split')

        return ret
    
    # returns a dictionary with the keys image_path and annotations -> list of dicts
    # 'annotations', 'image_path'
        # 'labels', 'bounding_box'
            
    def get_split(self, split: str) -> list[dict]:
        ret = list()
        try:
            for file in os.listdir(os.path.join(self.path, self.info[split][3:])):
                ret.append(self.read_pair(file, split))
        except KeyError as e:
            print("couldn't find split")
        return ret

    def get_all(self) -> list[dict]:
        return self.get_split('train') + self.get_split('test') + self.get_split('val')
    

    def combine(self, other: 'dataloader'): # should manually prune dataset for agreed vocab, cant really do anything rn 
        missing_in_self = [label for label in other.id_to_label.values() if label not in self.id_to_label.values()]
        missing_in_other = [label for label in self.id_to_label.values() if label not in other.id_to_label.values()]
        
        ## DISTILLING SELF
        with open(os.path.join(self.path, 'data.yaml'), 'w') as stream:
            self.info['nc'] += len(missing_in_self)
            self.info['names'] += missing_in_self
            self.id_to_label = {i : label for i, label in enumerate(self.info['names'])}
            stream.write(yaml.dump(self.info))

        # for split in ['train', 'test', 'val']:
            # for image_name in os.listdir(os.path.join(self.path, self.info[split][3:])):
            #     self.distill_image(image_name, split, missing_in_self)
        
        ## DISTILLING OTHER
        
        with open(os.path.join(other.path, 'data.yaml'), 'w') as stream:
            other.info['nc'] += len(missing_in_self)
            other.info['names'] += missing_in_self
            other.id_to_label = {i : label for i, label in enumerate(self.info['names'])}
            stream.write(yaml.dump(self.info))

        for split in ['train', 'test', 'val']:
            # for image_name in os.listdir(os.path.join(other.path, other.info[split][3:])):
            #     other.distill_image(image_name, split, missing_in_self)
            annotation_path = os.path.join(other.path, other.info[split][3:-6], 'labels/')
            for annotation in os.listdir(annotation_path):
                with open(annotation_path + annotation, 'r+') as f:
                    newlines = []
                    for line in f:
                            newline = str(self.info['names'].index(other.info['names'][int(line[0])])) + line[1:]
                            print(newline)
                            newlines.append(newline)
                    f.truncate(0)
                    f.seek(0)
                    for line in newlines:
                        f.write(line)

        
        # add missing labels to yaml files


    
            

    def distill_image(self, image_name: str, split: str, missing: list, confidence_threshold: int, prompt: str = ''):
        prev = os.getcwd()
        os.chdir(self.path)
        caption_ontology = CaptionOntology({prompt + label : label for label in missing})
        base_model = GroundedSAM(caption_ontology)
        new_annotations = []
        for box, _, confidence, class_id, _ in base_model.predict(self.get_image_path(split, image_name)):
            if confidence > confidence_threshold:
                h, w, _ = cv2.imread(self.get_image_path(split, image_name)).shape
                x_center = ((box[0] + box[2]) / 2.0) / w
                y_center = ((box[1] + box[3]) / 2.0) / h
                width = (box[2] - box[0]) / w
                height = (box[3] - box[1]) / h
                new_annotations.append(f'{self.info['names'].index(base_model.ontology.classes()[class_id])} {x_center} {y_center} {width} {height}')
        with open(self.get_label_path(split, image_name), 'a') as f:
            for annotation in new_annotations:
                f.write('\n' + annotation)
            
        print(new_annotations)
        os.chdir(prev)



        

    def get_label_path(self, split: str, image_name: str) -> str:
        return os.path.join(self.info[split][3:-6], 'labels', os.path.splitext(image_name)[0] + '.txt')

    def get_image_path(self, split: str, image_name: str) -> str:
        return os.path.join(self.info[split][3:-6], 'images', image_name)
        
    #moves split to other    
    def move_split(self, split: str, other: 'dataloader') -> None:
        for image in os.listdir(self.path + self.info[split][3:]):
            os.rename(os.path.join(self.path, self.get_image_path(split, image)), os.path.join(other.path, other.get_image_path(split, image)))
        for annotation in os.listdir(annotation_path := os.path.join(self.path, self.info[split][3:-6], 'labels/')):
            os.rename(os.path.join(self.path, self.get_label_path(split, image)), os.path.join(other.path, other.get_label_path(split, image)))

        
    def move_all(self, other: 'dataloader'):
        for split in ['train', 'test', 'val']:
            self.move_split(split, other)
        os.remove(os.path.join(self.path, 'data.yaml'))
        os.removedirs(self.path)

        
        
 


