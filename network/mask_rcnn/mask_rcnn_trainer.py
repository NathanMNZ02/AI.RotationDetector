import torch
import albumentations as A

from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset.instance_segmentation_dataset import InstanceSegmentationDataset, create_loader
from helper.model_saver import ModelSaver
from helper.coco_annotations_reader import CocoAnnotationsReader

class Trainer:
    def __init__(self, dataset_dir: str, train_transforms = [A.Compose([ToTensorV2(p=1.0)])], validation_transforms = [A.Compose([ToTensorV2(p=1.0)])]):
        self.device = self.__get_dev__()
                
        train_dataset = InstanceSegmentationDataset(f'{dataset_dir}/train', 
                                json_reader=CocoAnnotationsReader(f'{dataset_dir}/train/_annotations.coco.json'), 
                                transforms=train_transforms)
        
        val_dataset = InstanceSegmentationDataset(f'{dataset_dir}/valid', 
                                      json_reader=CocoAnnotationsReader(f'{dataset_dir}/valid/_annotations.coco.json'),
                                      transforms=validation_transforms)
        
        self.train_loader = create_loader(train_dataset)
        self.val_loader = create_loader(val_dataset, shuffle=False)
        

        self.model = self.__get_model__(len(train_dataset.classes))
        self.model.to(device=self.device)
        self.model.device = self.device
        self.model.name = 'maskrcnn_resnet50_fpn_v2'
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        
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
            
    def __get_dev__(self):
        if torch.cuda.is_available():
            device = torch.device('cuda') 
        else:
            device = torch.device('cpu')
            
        return device
        
    def start(self, num_epochs: int):
        print(f"Start training on device: {self.device}")
        
        model_saver = ModelSaver()
        for epoch in range(num_epochs):
            self.model.train()
            
            train_loss = 0.0     
            train_loader = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False)
            for _, (images, targets) in enumerate(train_loader):
                self.optimizer.zero_grad() # Set gradient value to zero for new batch
                
                #Put data on the GPU
                images = torch.stack(images).to(self.device)
                targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets]
                
                loss_dict = self.model(images.to(self.device), targets)

                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                losses.backward()
                self.optimizer.step() 
                
                train_loss += loss_value
                
                train_loader.set_postfix(loss=loss_value)

            avg_train_loss = train_loss / len(self.train_loader)
            
            val_loss = 0.0
            val_loader = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False)
            for _, data in enumerate(val_loader):
                images, targets = data
                
                #Put data on the GPU
                images = torch.stack(images).to(self.device)
                targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets]
                
                with torch.no_grad():
                    loss_dict = self.model(images.to(self.device), targets)
                        
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                val_loss += loss_value

                val_loader.set_postfix(loss=loss_value)

            avg_val_loss = val_loss / len(self.val_loader)

            model_saver.add_model(
                train_loss=avg_train_loss, 
                val_loss=avg_val_loss, 
                model=self.model, 
                optimizer=self.optimizer)
            
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                    f'Train Loss: {avg_train_loss:.4f}, '
                        f'Val Loss: {avg_val_loss:.4f}')

        model_saver.save_best_model()   
