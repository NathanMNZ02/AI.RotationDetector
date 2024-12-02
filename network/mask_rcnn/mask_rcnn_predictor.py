import torch
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2

from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class Predictor:    
    def __init__(self, model_path: str, classes: list[str]):
        self.device = self.__get_dev__()
        
        self.classes = classes

        self.transform = A.Compose([ToTensorV2(p=1.0)])
        
        self.model = maskrcnn_resnet50_fpn_v2(MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        
        dim_reduced = self.model.roi_heads.mask_predictor.conv5_mask.out_channels
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, len(self.classes))
        self.model.roi_heads.box_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced, len(self.classes))
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def __get_dev__(self):
        if torch.cuda.is_available():
            device = torch.device('cuda') 
        elif torch.backends.mps.is_available():
            device = torch.device('mps')  
        elif torch.backends.opencl.is_available():
            device = torch.device('opencl')
        else:
            device = torch.device('cpu')
            
        return device
    
    def start(self, image):
        image_array = np.array(image, dtype=np.float32) / 255.0

        transform = self.transform(image=image_array)
        image_array = transform['image']
        with torch.no_grad():
            outputs = self.model([image_array])
            
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()

            pred_classes = [self.classes[i] for i in outputs[0]['labels'].cpu().numpy()]            
            boxes = boxes[scores >= 0.6].astype(np.int32) 
                 
            return pred_classes, boxes
        
        return None, None