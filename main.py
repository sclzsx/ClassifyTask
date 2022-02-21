import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt

import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset

args = ARGS()
model = SimpleCNN(num_classes=20, inp_size=224, c_dim=3)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# scheduler = torch.optim.lr_scheduler.StepLR(...)
scheduler = None
test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
print('test map:', test_map)