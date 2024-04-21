import yaml
import os

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
        print(self.info)


    # note file_name has no filetype
    def read_pair(self, image_name: str, split: str) -> dict:
        prev = os.getcwd()
        ret = dict()
        ret['annotations'] = []
        try:
            os.chdir(self.path)
            ret['image_path'] = self.info[split][3:] + image_name
            # gets rid of the ../ that os.chdir doesnt like
            label_path = os.path.join(self.info[split][3:-6], 'labels', os.path.splitext(image_name)[0] + '.txt')
            
            with open(label_path) as f:
                for line in f:
                    tokens = line.split()
                    ret['annotations'].append({'labels': tokens[0], 'bounding_box': tokens[1:]})    
            os.chdir(prev)
        except (OSError, IOError) as e:
            print('couldnt find file')
        except KeyError as e:
            print('couldnt find that split')

        return ret
    
    ## returns a dictionary with the keys image_path and annotations -> list of dicts
            
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


