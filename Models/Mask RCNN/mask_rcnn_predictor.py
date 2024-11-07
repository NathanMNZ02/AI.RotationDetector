### 1) Load dataset
import torchvision.transforms as transforms
from DataSet.dataset import Easy4ProDataset, create_loader
from Helper.json_reader import JsonReader
from Helper.config import TRAINING_SIZE, MEAN, STD

train_json_reader = JsonReader('DataSet/test/_annotations.coco.json')
val_json_reader = JsonReader('DataSet/valid/_annotations.coco.json')

transform = transforms.Compose([
    transforms.Resize(TRAINING_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN], std=[STD])
])
    
train_dataset = Easy4ProDataset('DataSet/test', train_json_reader, transform)
val_dataset = Easy4ProDataset('DataSet/valid', val_json_reader, transform)

train_loader = create_loader(train_dataset)
val_loader = create_loader(val_dataset, False)

### 2) Create Model
import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from Helper.config import NUM_CLASSES, DEVICE

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

model.to(DEVICE)

### 3) Define optimizer and loss function
import torch.nn as nn

loss_function = nn.CrossEntropyLoss()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

### 3) Start training and validate
from Helper.config import NUM_EPOCHS
from Helper.model_saver import ModelSaver

model_saver = ModelSaver()
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        images, targets = data
        
        #Put data on the GPU
        images = [image.to(DEVICE) for image in images]
        targets = [{k: v.to(DEVICE) for k, v in target.items()} for target in targets]
        
        loss_dict = model(images, targets)     
        
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        losses.backward()
        optimizer.step()
        
        train_loss += loss_value

    avg_train_loss = train_loss / len(train_loader)
    
    val_loss = 0.0
    for i, data in enumerate(val_loader):
        images, targets = data
        
        #Put data on the GPU
        images = [image.to(DEVICE) for image in images]
        targets = [{k: v.to(DEVICE) for k, v in target.items()} for target in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
                
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss += loss_value

    avg_val_loss = val_loss / len(val_loader)

    model_saver.add_model(avg_train_loss, model, optimizer)
    
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], '
            f'Train Loss: {avg_train_loss:.4f}, '
                f'Val Loss: {avg_val_loss:.4f}')

model_saver.save_best_model()