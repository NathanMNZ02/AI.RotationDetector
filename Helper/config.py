import torch

DEVICE_GPU = False

BATCH_SIZE = 4
NUM_EPOCHS = 10
NUM_WORKERS = 4

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = torch.device('cpu') if torch.backends.mps.is_available() else torch.device('cpu')

TRAINING_SIZE = (600, 600)
TRAIN_DIR = 'DataSet/train/'
VALID_DIR = 'DataSet/valid/'
TEST_DIR = 'DataSet/test/'

CLASSES = [
    '__background__', 'None', 'est', 'nord', 'nord-est', 'nord-ovest', 'ovest', 'sud', 'sud-est', 'sud-ovest'
]
NUM_CLASSES = len(CLASSES)