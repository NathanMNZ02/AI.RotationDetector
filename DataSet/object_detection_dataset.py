import os
import torch
import numpy as np
import albumentations as A

from PIL import Image
from helper.coco_annotations_reader import CocoAnnotationsReader
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

class ObjectDetectionDataset(Dataset):
    """
    Dataset per la rilevazione di oggetti tramite bounding box.

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
    def __init__(self, working_dir, json_reader: CocoAnnotationsReader, transforms = [A.Compose([ToTensorV2(p=1.0)])]):
        self.working_dir = working_dir
        
        self.json_images = json_reader.get_images()
        self.json_targets = json_reader.get_targets()
        self.classes = json_reader.get_categories()
        
        if transforms != None:
            self.transforms = transforms
        else:
            self.transforms = [A.Compose([ToTensorV2(p=1.0)])]
                    
    def __len__(self):
        """
        Restituisce il numero di immagini nel dataset.

        Returns:
            int: Numero di immagini.
        """
        return len(self.json_images) * len(self.transforms)
        
    def __getitem__(self, idx):
        """
        Restituisce un'immagine e il corrispondente target dato un indice.

        Args:
            idx (int): Indice dell'immagine e del relativo target.

        Returns:
            tuple: (image, target), dove `image` è un tensore normalizzato 
                   e `target` è un dizionario contenente bounding box, etichette, area, ecc.
        """
        original_idx = idx // len(self.transforms)
        transform_idx = idx % len(self.transforms)
        
        json_image = self.json_images[original_idx]
                
        image_path = os.path.join(self.working_dir, json_image['file_name'])
        image = Image.open(image_path).convert("RGB") 
        image = np.array(image, dtype=np.float32) 
        
        targets = []
        for json_target in self.json_targets:
            if json_target['image_id'] == json_image['id']:
                targets.append(json_target)
         
        bboxes = []
        labels = []
        for target in targets:
            labels.append(target["category_id"])
            x, y, w, h = target["bbox"]
            bboxes.append([x, y, x + w, y + h])
        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': bboxes, 
            'labels': labels,
        }
        
        if self.transforms:
            transform = self.transforms[transform_idx]
            image = transform(image = image)['image']
 
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
