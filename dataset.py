import yaml
import os

class dataloader:

    def __init__(self, path: str) -> None:
        try:
            self.path = path
            with open(path + 'data.yaml', 'r') as stream:
                self.info = yaml.safe_load(stream)
                self.id_to_label = {i : label for i, label in enumerate(self.info['names'])}
        except yaml.Exception as e:
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
            label_path = os.path.join(self.info[split][3:-6], 'labels', os.path.splitext(image_name)[0] + '.txt')
            
            with open(label_path) as f:
                for line in f:
                    tokens = line.split()
                ret['annotations'].append((tokens[0], tokens[1:]))    
            os.chdir(prev)
        except (OSError, IOError) as e:
            print('couldnt find file')
        except KeyError as e:
            print('couldnt find that split')

        return ret
            
        



