import torch
import albumentations as A

from albumentations.pytorch import ToTensorV2

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset.orientation_detector_dataset import OrientationDetectorDataset, create_loader
from helper.model_saver import ModelSaver
from helper.json_reader import JsonReader

class Trainer:
    def __init__(self, dataset_dir: str, num_classes: int):
        self.device = self.__get_dev__()
        
        self.transform = A.Compose([ToTensorV2(p=1.0)])
            
        train_dataset = OrientationDetectorDataset(f'{dataset_dir}/train', 
                                        json_reader=JsonReader(f'{dataset_dir}/train/_annotations.coco.json'), 
                                        transform=self.transform)
        
        val_dataset = OrientationDetectorDataset(f'{dataset_dir}/valid', 
                                      json_reader=JsonReader(f'{dataset_dir}/valid/_annotations.coco.json'), 
                                      transform=self.transform)

        self.train_loader = create_loader(train_dataset)
        self.val_loader = create_loader(val_dataset, False)
        
        self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.to(self.device)
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
        
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
                
                # Get model losses dictionary, contains something like:
                # {'loss_classifier': 0.123, 'loss_box_reg': 0.045, 'loss_objectness': 0.032, 'loss_rpn_box_reg': 0.012}
                loss_dict = self.model(images, targets) 
                
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                losses.backward()
                self.optimizer.step() # Update model weights
                
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

