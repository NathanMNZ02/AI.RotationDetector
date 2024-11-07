import os
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageOps
from Helper.config import TRAINING_SIZE, TEST_DIR, MEAN, STD, CLASSES, NUM_CLASSES, DEVICE

class Predictor:    
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        
        checkpoint = torch.load('Output/best_model.pth', map_location=DEVICE, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def __preprocess__(self, image):
        image = ImageOps.exif_transpose(image).convert('RGB')
        image.thumbnail(TRAINING_SIZE, Image.LANCZOS)
        
        result = Image.new('RGB', TRAINING_SIZE, (255, 255, 255))
        offset = ((TRAINING_SIZE[0] - image.width) // 2, (TRAINING_SIZE[1] - image.height) // 2)      
        result.paste(image, offset)
        
        return result
    
    def start(self):
        transform = transforms.Compose([
            transforms.Resize(TRAINING_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[MEAN], std=[STD])
        ])
        
        for file in os.listdir(TEST_DIR):
            if file.endswith("jpg") or file.endswith("png") or file.endswith("jpeg") or file.endswith("avif"):
                image_path = os.path.join(TEST_DIR, file)
                image = self.__preprocess__(Image.open(image_path))
                
                tensor = transform(image)
                with torch.no_grad():
                    outputs = self.model([tensor])
                    
                if len(outputs[0]['boxes']) != 0:
                    boxes = outputs[0]['boxes'].data.numpy()
                    scores = outputs[0]['scores'].data.numpy()
                    print(f'Boxes: {boxes}'
                        f'Scores: {scores}')
                    
                    pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
                    print(pred_classes)
                    
                    boxes = boxes[scores >= 0.8].astype(np.int32)      
                    orig_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    for j, box in enumerate(boxes):
                        class_name = pred_classes[j]
                        
                        cv2.rectangle(orig_image,
                                    (int(box[0]), int(box[1])),
                                    (int(box[2]), int(box[3])),
                                    (0, 0, 255), 2)
                        
                        cv2.putText(orig_image, class_name, 
                                    (int(box[0]), int(box[1]-5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 
                                    2, lineType=cv2.LINE_AA)

                    cv2.imshow('Prediction', orig_image)
                    cv2.waitKey(0)