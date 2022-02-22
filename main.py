import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import trainer
from utils import ARGS, get_data_loader
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import random
import cv2

args = ARGS()


def Q_4():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, args.class_num)
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, gamma=args.gamma)
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)

    with open('test_map.txt', 'w') as f:
        f.write(str(test_map))


def extract_features():
    with torch.no_grad():
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, args.class_num)
        model.load_state_dict(torch.load('model_10'))

        model.eval()
        model.to(args.device)

        all_features = []
        all_labels = []
        test_loader = get_data_loader('voc', train=True, batch_size=args.batch_size, split='test')
        for data, target, _ in test_loader:
            data = data.to(args.device)

            layer = nn.Sequential(*list(model.children())[:-1])
            feature = layer(data)
            feature = np.array(feature.cpu().squeeze())
            feature = feature.reshape(1, -1).squeeze().tolist()
            all_features.extend(feature)

            label = np.array(target.squeeze()).reshape(1, -1).squeeze().tolist()
            all_labels.extend(label)

        all_features = np.array(all_features)
        all_features = all_features.reshape(-1, model.fc.in_features)

        all_labels = np.array(all_labels)
        all_labels = all_labels.reshape(-1, args.class_num)

        pd.DataFrame(all_features).to_csv('all_features.csv', index=False, header=False)
        pd.DataFrame(all_labels).to_csv('all_labels.csv', index=False, header=False)


def Q_5_1():
    all_features = np.array(pd.read_csv('all_features.csv'))
    all_labels = np.array(pd.read_csv('all_labels.csv'))

    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(all_features, all_labels)

    with torch.no_grad():
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, args.class_num)
        model.load_state_dict(torch.load('model_10'))

        model.eval()
        model.to(args.device)

        test_loader = get_data_loader('voc', train=True, batch_size=1, split='test')

        img_num = 3

        previous_label = np.zeros((1, args.class_num), dtype='float32')
        for data, target, _ in test_loader:
            label = np.array(target)

            if (previous_label == label).all():
                continue
            else:
                previous_label = label

            data = data.to(args.device)

            layer = nn.Sequential(*list(model.children())[:-1])
            feature = layer(data)
            feature = np.array(feature.cpu().squeeze().unsqueeze(0))

            knn_pred = knn.predict(feature)
            print('knn_pred', knn_pred)
            print('label:', label)
            print()

            img_num = img_num - 1
            if img_num < 0:
                break


def generate_colors(num):
    random.seed(0)
    colors = []
    for i in range(num):
        color = []
        for i in range(3):
            color_val = random.randint(0, 255)
            color.append(color_val)
        colors.append(color)
    return colors


def Q_5_2():
    features = np.array(pd.read_csv('all_features.csv'))[:1000, :]
    labels = np.array(pd.read_csv('all_labels.csv'))[:1000, :]
    tsne = TSNE()
    tsnes = tsne.fit_transform(features)
    min_ = np.min(tsnes)
    max_ = np.max(tsnes)
    tsnes = (tsnes - min_) / (max_ - min_)

    tsnes = tsnes * 255
    tsnes = tsnes.astype('uint8')

    draw = np.ones([255, 255, 3], dtype='uint8') * 255

    colors = generate_colors(args.class_num)

    for i in range(1000):
        label = labels[i, :]
        # print(label.shape)
        class_idxs = np.argwhere(label > 0).squeeze(1)
        class_idxs = class_idxs.tolist()
        # print(class_idxs)
        selected_colors = []
        for idx in class_idxs:
            selected_colors.append(colors[idx])
        num_colors = len(selected_colors)

        if num_colors > 1:
            color = np.zeros(3)

            for c in selected_colors:
                c = np.array(c)
                color = color + c

            color = color / num_colors

        else:
            assert num_colors == 1
            color = selected_colors[0]

        color = tuple([int(x) for x in color])

        center = (tsnes[i, 0], tsnes[i, 1])

        cv2.circle(draw, center=center, radius=1, color=color, thickness=-1)

    cv2.imwrite("t_sne.png", draw)


def Q_5_3():
    pass
    # 样本数量不平衡：对损失加权，权重为类别数量的倒数
    # 小目标：减小网络层数，增加特征金字塔以融合特征


if __name__ == '__main__':
    Q_4()
    extract_features()
    Q_5_1()
    Q_5_2()
    Q_5_3()
