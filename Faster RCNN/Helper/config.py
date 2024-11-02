import torch

DEVICE_GPU = False

BATCH_SIZE = 4
RESIZE = 224
NUM_EPOCHS = 10
NUM_WORKERS = 4

MEAN, STD = 0.5, 0.5

DEVICE = torch.device('cpu') if torch.backends.mps.is_available() else torch.device('cpu')

TRAINING_SIZE = (128, 128)
TRAIN_DIR = 'DataSet/train/'
VALID_DIR = 'DataSet/test/'
TEST_DIR = 'DataSet/valid/'

CLASSES = [
    '__background__', 'glasses', 'ring'
]
NUM_CLASSES = len(CLASSES)