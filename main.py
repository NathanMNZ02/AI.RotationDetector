import albumentations as A
from albumentations.pytorch import ToTensorV2
from network.mask_rcnn.mask_rcnn_trainer import Trainer
from dataset.instance_segmentation_dataset import InstanceSegmentationDataset
from helper.coco_annotations_reader import CocoAnnotationsReader

dataset_dir = '/Users/nathanmonzani/Downloads/Easy4Pro.Intelligence2.v3i.coco'
dataset = InstanceSegmentationDataset(f'{dataset_dir}/valid', 
                                json_reader=CocoAnnotationsReader(f'{dataset_dir}/valid/_annotations.coco.json'),
                                transforms=None)
dataset.visualize()

transforms = [
            # Trasformazione 0: Immagine base
            A.Compose([ToTensorV2(p=1.0)]),
            
            # Trasformazione 1: Variazioni illuminazione lightbox
            A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.8
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, 
                    sat_shift_limit=20, 
                    val_shift_limit=20, 
                    p=0.6
                ),
                A.Equalize(p=0.4),
                ToTensorV2(p=1.0)
            ], p=1.0),
            
            # Trasformazione 2: Disturbi acquisizione telecamera
            A.Compose([
                A.GaussNoise(var_limit=(10, 30), p=0.6),
                A.ISONoise(
                    color_shift=(0.01, 0.03), 
                    intensity=(0.1, 0.3), 
                    p=0.5
                ),
                A.RandomFog(
                    fog_coef_lower=0.1, 
                    fog_coef_upper=0.3, 
                    p=0.4
                ),
                A.Defocus(radius=(3, 7), p=0.5),
                A.GaussianBlur(
                    blur_limit=(3, 5), 
                    p=0.4
                ),
                ToTensorV2(p=1.0)
            ], p=1.0),
            
            # Trasformazione 3: Variazioni superficie/oggetto
            A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, 
                    contrast_limit=0.15, 
                    p=0.7
                ),
                A.ColorJitter(
                    brightness=0.1, 
                    contrast=0.1, 
                    saturation=0.1, 
                    hue=0.1, 
                    p=0.5
                ),
                A.MultiplicativeNoise(
                    multiplier=(0.95, 1.05), 
                    per_channel=True, 
                    p=0.4
                ),
                ToTensorV2(p=1.0)
            ], p=1.0),
            
            # Trasformazione 4: Disallineamenti e rumore
            A.Compose([
                A.CLAHE(clip_limit=2, p=0.3),
                A.PixelDropout(dropout_prob=0.02, per_channel=True, p=0.4),
                A.RandomShadow(p=0.3),
                A.RandomRain(p=0.2),
                ToTensorV2(p=1.0)
            ], p=1.0)
        ]
        
trainer = Trainer('/Users/nathanmonzani/Downloads/Easy4Pro.Intelligence2.v3i.coco', transforms=transforms)
trainer.start(50)