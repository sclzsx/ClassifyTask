import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt

import trainer
from utils import ARGS
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset

# # Pre-trained weights up to second-to-last layer
# # final layers should be initialized from scratch!
# class PretrainedResNet(nn.Module):
#     def __init__(self, num_fc_out=20):
#         super().__init__()

#         net = models.resnet18(pretrained=True)
        
#         for param in net.parameters():
#             param.requires_grad = True

#         num_fc_in = net.fc.in_features
#         net.fc = nn.Linear(num_fc_in, num_fc_out)
#         self.net = net()

#     def forward(self, x):
#         x = self.net(x)
#         return x

# model = models.resnet18(pretrained=True)
# net_structure = list(model.children())
# print(net_structure[-1])

# class PretrainedResNet(nn.Module):
#     def __init__(self, num_fc_out=20):
#         super().__init__()

#         model = models.resnet18(pretrained=True)

#         for param in model.parameters():
#             param.requires_grad = True

#         self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
#         # num_fc_in = model.fc.in_features
#         # self.fc = nn.Linear(num_fc_in, num_fc_out)
        
#         # num_fc_in = model.fc.in_features
#         # model.fc = nn.Linear(num_fc_in, num_fc_out)
#         # self.model = model()

#     def forward(self, x):
#         x = self.resnet_layer(x)

#         # print(x.shape)
#         # x = self.fc(x)

#         # x = self.model(x)
#         return x

args = ARGS()
# model = SimpleCNN(num_classes=20, inp_size=224, c_dim=3)
# model = PretrainedResNet(num_fc_out=20)



model = models.resnet18(pretrained=True)
channel_in = model.fc.in_features
class_num = 20
model.fc = nn.Linear(channel_in, class_num)

for param in model.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

# net.load_state_dict(
#     torch.load(osp.join(model_save_root_path, 'resnet50_50_2020-04-09_22-15-11.pth')))



optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# scheduler = torch.optim.lr_scheduler.StepLR(...)
scheduler = None
test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
print('test map:', test_map)


