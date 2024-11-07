import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset.orientation_detector_dataset import OrientationDetectorDataset, create_loader
from helper.model_saver import ModelSaver
from helper.json_reader import JsonReader
from helper.config import NUM_CLASSES, DEVICE, TRAINING_SIZE, MEAN, STD, NUM_EPOCHS

class Trainer:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(TRAINING_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[MEAN], std=[STD])
        ])
            
        train_dataset = OrientationDetectorDataset('DataSet/train', 
                                        json_reader=JsonReader('DataSet/train/_annotations.coco.json'), 
                                        transform=self.transform)
        
        val_dataset = OrientationDetectorDataset('DataSet/valid', 
                                      json_reader=JsonReader('DataSet/valid/_annotations.coco.json'), 
                                      transform=self.transform)

        self.train_loader = create_loader(train_dataset)
        self.val_loader = create_loader(val_dataset, False)
        
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        self.model.to(DEVICE)
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        
    def start(self):
        model_saver = ModelSaver()
        for epoch in range(NUM_EPOCHS):
            self.model.train()
            train_loss = 0.0
            
            for i, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                images, targets = data
                
                #Put data on the GPU
                images = [image.to(DEVICE) for image in images]
                targets = [{k: v.to(DEVICE) for k, v in target.items()} for target in targets]
                
                loss_dict = self.model(images, targets)     
                
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                losses.backward()
                self.optimizer.step()
                
                train_loss += loss_value

            avg_train_loss = train_loss / len(self.train_loader)
            
            val_loss = 0.0
            for i, data in enumerate(self.val_loader):
                images, targets = data
                
                #Put data on the GPU
                images = [image.to(DEVICE) for image in images]
                targets = [{k: v.to(DEVICE) for k, v in target.items()} for target in targets]
                
                with torch.no_grad():
                    loss_dict = self.model(images, targets)
                        
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                val_loss += loss_value

            avg_val_loss = val_loss / len(self.val_loader)

            model_saver.add_model(avg_train_loss, self.model, self.optimizer)
            
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], '
                    f'Train Loss: {avg_train_loss:.4f}, '
                        f'Val Loss: {avg_val_loss:.4f}')

        model_saver.save_best_model()

