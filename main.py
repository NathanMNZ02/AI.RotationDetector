import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from network.mask_rcnn.mask_rcnn_trainer import Trainer
from network.mask_rcnn.mask_rcnn_predictor import Predictor
from PIL import Image, ImageOps

def mask_rcnn_training():
    train_transforms = [
                # Trasformazione 0: Immagine base
                A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(p=1.0)]),

                # Trasformazione 1: Variazioni illuminazione lightbox
                A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    A.CLAHE(clip_limit=2, p=0.3),
                    A.PixelDropout(dropout_prob=0.02, per_channel=True, p=0.4),
                    A.RandomShadow(p=0.3),
                    A.RandomRain(p=0.2),
                    ToTensorV2(p=1.0)
                ], p=1.0)
            ]

    validation_transforms = [
                    A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2(p=1.0)]),
    ]
            
    trainer = Trainer('/Users/nathanmonzani/Downloads/Easy4Pro.Intelligence2.v4i.coco', 
                      train_transforms=train_transforms, 
                      validation_transforms=validation_transforms)
    
    trainer.start(100)
    
def mask_rcnn_predict():
    classes = [
        "orientation-tIlo",
        "x-0",
        "x-105",
        "x-120",
        "x-135",
        "x-15",
        "x-150",
        "x-165",
        "x-180",
        "x-195",
        "x-210",
        "x-225",
        "x-240",
        "x-255",
        "x-270",
        "x-285",
        "x-30",
        "x-300",
        "x-315",
        "x-330",
        "x-345",
        "x-45",
        "x-60",
        "x-75",
        "x-90",
    ]
    
    predict_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(p=1.0)]
    )
     
    
    predictor = Predictor('/Users/nathanmonzani/Downloads/best_model.pth', classes, predict_transform=predict_transform)
    
    images = [Image.open('/Users/nathanmonzani/Desktop/RotationDetecor.Images/20241021_103626.jpg'),
              Image.open('/Users/nathanmonzani/Desktop/RotationDetecor.Images/20241021_102833.jpg'),
              Image.open('/Users/nathanmonzani/Desktop/RotationDetecor.Images/20241021_103006.jpg')]
    
    for image in images:
        if image.size != (600, 600):
            image = ImageOps.exif_transpose(image).convert('RGB')        
            image.thumbnail((600, 600), Image.LANCZOS)
            
            background = Image.new('RGB', (600, 600), (255, 255, 255))
            offset = (((600, 600)[0] - image.width) // 2, ((600, 600)[1] - image.height) // 2)
            background.paste(image, offset)
            
            image = background
            
            image = np.array(image, dtype=np.float32)
            
        print(predictor.start(image))
    
mask_rcnn_predict()