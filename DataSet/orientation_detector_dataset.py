import os
import torch
import numpy as np

from PIL import Image
from helper.json_reader import JsonReader
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

class OrientationDetectorDataset(Dataset):
    """
    Dataset per la rilevazione di orientamento con bounding box.

    Fornisce logica per il caricamento di immagini e annotazioni, supportando
    trasformazioni e preparazione di target per il training.

    Args:
        working_dir (str): Percorso della directory contenente le immagini.
        json_reader (JsonReader): Oggetto JsonReader per leggere immagini e annotazioni.
        transform (albumentations.pytorch.ToTensorV2): Trasformazioni da applicare alle immagini.

    Attributes:
        json_images (list): Lista di immagini dal JSON.
        json_targets (list): Lista di annotazioni dal JSON.
        working_dir (str): Directory di lavoro per il caricamento delle immagini.
        transform (albumentations.pytorch.ToTensorV2): Trasformazioni da applicare.
        classes (list): Categorie di classi estratte dal JSON.
    """
    def __init__(self, working_dir, json_reader: JsonReader, transform: ToTensorV2):
        self.json_images = json_reader.get_images()
        self.json_targets = json_reader.get_target()
        self.working_dir = working_dir
        self.transform = transform
        self.classes = json_reader.get_categories()
        
    def __len__(self):
        """
        Restituisce il numero di immagini nel dataset.

        Returns:
            int: Numero di immagini.
        """
        return len(self.json_images)
        
    def __getitem__(self, idx):
        """
        Restituisce un'immagine e il corrispondente target dato un indice.

        Args:
            idx (int): Indice dell'immagine e del relativo target.

        Returns:
            tuple: (image, target), dove `image` è un tensore normalizzato 
                   e `target` è un dizionario contenente bounding box, etichette, area, ecc.
        """
        json_image = self.json_images[idx]
                
        image_path = os.path.join(self.working_dir, json_image['file_name'])
        image = Image.open(image_path).convert("RGB") 
        image = np.array(image, dtype=np.float32) / 255.0
        
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
        
        if self.transform:
            transform = self.transform(image=image)
            image = transform['image']
 
        return image, target
    
    def print(self):
        """
        Stampa i target associati al dataset (debugging).
        """
        print(self.targets)

def create_loader(dataset, batch_size=4, shuffle=True):
    """
    Crea un DataLoader per il dataset.

    Args:
        dataset (Dataset): Il dataset da caricare.
        batch_size (int, opzionale): Dimensione del batch. Default: 4.
        shuffle (bool, opzionale): Se mescolare i dati. Default: True.

    Returns:
        DataLoader: Oggetto DataLoader configurato.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=__collate_fn
    )

def __collate_fn(batch):
    """
    Collate function per gestire la raccolta di immagini e target.

    Args:
        batch (list): Batch di dati da processare.

    Returns:
        tuple: Unione di immagini e target.
    """
    return tuple(zip(*batch))
