import torch
import albumentations as A

from albumentations.pytorch import ToTensorV2

from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset.instance_segmentation_dataset import InstanceSegmentationDataset, create_loader
from helper.model_saver import ModelSaver
from helper.coco_annotations_reader import CocoAnnotationsReader

class Trainer:
    def __init__(self, dataset_dir: str, transforms = [A.Compose([ToTensorV2(p=1.0)])]):
        self.device = self.__get_dev__()
                
        train_dataset = InstanceSegmentationDataset(f'{dataset_dir}/train', 
                                json_reader=CocoAnnotationsReader(f'{dataset_dir}/train/_annotations.coco.json'), 
                                transforms=transforms)
        
        val_dataset = InstanceSegmentationDataset(f'{dataset_dir}/valid', 
                                      json_reader=CocoAnnotationsReader(f'{dataset_dir}/valid/_annotations.coco.json'),
                                      transforms=transforms)
        
        self.train_loader = create_loader(train_dataset)
        self.val_loader = create_loader(val_dataset, shuffle=False)
        
        self.model = maskrcnn_resnet50_fpn_v2(MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        
        dim_reduced = self.model.roi_heads.mask_predictor.conv5_mask.out_channels
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, len(train_dataset.classes))
        self.model.roi_heads.box_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced, len(train_dataset.classes))
        self.model.to(device = self.device)
        
        self.model.device = self.device
        self.model.name = 'maskrcnn_resnet50_fpn_v2'
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        
    def __get_dev__(self):
        if torch.cuda.is_available():
            device = torch.device('cuda') 
       # elif torch.backends.mps.is_available():
            device = torch.device('mps')  
        # elif torch.backends.opencl.is_available():
            device = torch.device('opencl')
        else:
            device = torch.device('cpu')
            
        return device
        
    def start(self, num_epochs: int):
        model_saver = ModelSaver()
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            
            for _, data in enumerate(self.train_loader):
                self.optimizer.zero_grad() # Set gradient value to zero for new batch

                images, targets = data
                
                #Put data on the GPU
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets]
                
                loss_dict = self.model(images, targets) 

                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                losses.backward()
                self.optimizer.step() 
                
                train_loss += loss_value

            avg_train_loss = train_loss / len(self.train_loader)
            
            val_loss = 0.0
            for _, data in enumerate(self.val_loader):
                images, targets = data
                
                #Put data on the GPU
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets]
                
                with torch.no_grad():
                    loss_dict = self.model(images, targets)
                        
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                val_loss += loss_value

            avg_val_loss = val_loss / len(self.val_loader)

            model_saver.add_model(avg_train_loss, self.model, self.optimizer)
            
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                    f'Train Loss: {avg_train_loss:.4f}, '
                        f'Val Loss: {avg_val_loss:.4f}')

        model_saver.save_best_model()        
