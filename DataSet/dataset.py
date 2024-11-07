import os
import torch
from PIL import Image
from Helper.json_reader import JsonReader
from torch.utils.data import Dataset, DataLoader

class Easy4ProDataset(Dataset):
    def __init__(self, working_dir, json_reader: JsonReader, transform):
        self.json_images = json_reader.get_images()
        self.json_targets = json_reader.get_target()
        self.working_dir = working_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.json_images)
        
    def __getitem__(self, idx):
        json_image = self.json_images[idx]
                
        #Read image, convert to RGB and normalize
        image_path = os.path.join(self.working_dir, json_image['file_name'])
        image = Image.open(image_path).convert("RGB") 
        if self.transform:
            image = self.transform(image)

        
        #Read annotations for the image
        targets = []
        for json_target in self.json_targets:
            if json_target['image_id'] == json_image['id']:
                targets.append(json_target)
         
        boxes = []
        labels = []
        for target in targets:
            labels.append(target["category_id"])
            x, y, w, h = target["bbox"]
            boxes.append([x, y, x + w, y + h])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = iscrowd
        image_id = torch.tensor([idx])
        target['image_id'] = image_id
 
        return image, target
    
    def print(self):
        print(self.targets)

from Helper.config import BATCH_SIZE
def create_loader(dataset, shuffle=True):
    return DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=__collate_fn
    )

def __collate_fn(batch):
    return tuple(zip(*batch))