import torch
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2

from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class Predictor:    
    def __init__(self, model_path: str, classes: list[str], predict_transform = [A.Compose([ToTensorV2(p=1.0)])]):
        self.classes = classes        
        self.transform = predict_transform
        
        self.model = self.__get_model__(len(classes))
        self.model.load_state_dict(
            state_dict=torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)['model_state_dict'])
        self.model.eval()
        
    def __get_model__(self, num_classes):
        model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
        
        return model
    
    def __get_class_by_id(self, idx):
        return self.classes[idx]
    
    def start(self, image: np.ndarray):
        image = self.transform(image = image)['image']
                
        with torch.no_grad():
            outputs = self.model([image])
            
        if len(outputs) == 1:
            boxes = outputs[0]['boxes'].numpy()
            scores = outputs[0]['scores'].numpy()
            masks = outputs[0]['masks'].numpy()

            pred_classes = [self.__get_class_by_id(i) for i in outputs[0]['labels'].numpy()]            
            boxes = boxes[scores >= 0.6].astype(np.int32) 
            masks = masks[scores >= 0.6].astype(np.bool)
                 
        return masks, boxes, pred_classes