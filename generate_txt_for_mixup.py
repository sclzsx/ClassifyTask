import os
with open('data/VOCdevkit/VOC2007_mixup/trainval_mixup.txt', 'w') as f:
    for path in os.listdir('data/VOCdevkit/VOC2007_mixup/Annotations'):
        if 'mixup' in path:
            # print(path)
            f.write(path[:-4])
            f.write('\n')