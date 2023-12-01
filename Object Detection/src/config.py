import torch

BATCH_SIZE = 4
RESIZE_TO = 512
NUM_EPOCHS = 100

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = '../dataset/train'
VALID_DIR = '../dataset/test'

CLASSES = ['background', 'Signature']
NUM_CLASSES = 2

VISUALIZE_TRANSFORMED_IMAGES = False

OUT_DIR = '../output'
SAVE_PLOTS_EPOCH = 2
SAVE_MODEL_EPOCH = 2