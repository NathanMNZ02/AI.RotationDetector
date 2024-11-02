### 1) Create Model
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from Helper.config import NUM_CLASSES

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

### 2) Load weights
import torch
from Helper.config import DEVICE

checkpoint = torch.load('Output/best_model.pth', map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

### 3) Test
import os
import cv2
import numpy as np
from PIL import Image
from Helper.config import TEST_DIR
import torchvision.transforms as transforms
from Helper.config import TRAINING_SIZE, MEAN, STD, CLASSES

transform = transforms.Compose([
    transforms.Resize(TRAINING_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN], std=[STD])
])

for file in os.listdir(TEST_DIR):
    if file.endswith("jpg"):
        image_path = os.path.join(TEST_DIR, file)
        image = Image.open(image_path).convert("RGB") 
        image = transform(image)
        
        with torch.no_grad():
            outputs = model([image])
            
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            print(f'Boxes: {boxes}'
                  f'Scores: {scores}')
            
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            print(pred_classes)
            
            boxes = boxes[scores >= 0.8].astype(np.int32)
            orig_image = cv2.imread(image_path)
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