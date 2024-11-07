import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class Easy4ProDataset(Dataset):
    def __init__(self, working_dir, csv_path, transform=None):
        df = pd.read_csv(csv_path)

        self.working_dir = working_dir
        
        #Estrae i percorsi delle immagini
        self.images = df['filename'].tolist()
        
        #Estrae la matrice degli orientamenti
        orientation_columns = [col for col in df.columns if col.startswith('Orientation')]      
        self.orientations = torch.tensor(df[orientation_columns].values, dtype=torch.float32)
        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.working_dir, self.images[idx])
        image = Image.open(image_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
        
        orientation = self.orientations[idx]
 
        return image, orientation