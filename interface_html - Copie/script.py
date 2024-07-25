import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, ConcatDataset
import torchvision.transforms as tt
import rk_dataset as RKDataset
from PIL import Image
from pathlib import Path
from torchvision import transforms as T

from utils import ToDeviceLoader, to_device, get_device, accuracy, train, load_pre_trained_weights, load_weights, Watcher, get_classes_items, predict_image
from models import ResNet


if __name__ == "__main__":


    print("Script python lancé", flush=True)
    BATCH_SIZE = 16

    parser = argparse.ArgumentParser(description='Params of the script')
    parser.add_argument('-image_type', metavar='-t', type=str, 
                        help='type of the input images')


    args = parser.parse_args()

    image_type = args.image_type

    data_path = r'C:\Users\Enzo\Pictures\dataset'
    weigth_path = r"C:\Users\Enzo\Documents\Code_enzo\resnet18_test_\rk-resnet-bs-16-48-project.pth"
    check_path = r'C:\Users\Enzo\Pictures\dataset'
    #check_path = r'C:\Users\Enzo\Documents\Code_enzo'
    class_name = 'RK_fusion'

    epochs = 100
    optimizer = torch.optim.Adam
    max_lr = 5e-4
    grad_clip = 0.1
    weight_decay = 1e-5
    scheduler = torch.optim.lr_scheduler.OneCycleLR

    train_data = RKDataset.RKDataset(data_path, class_name=class_name, is_train=True)
    test_data = RKDataset.RKDataset(data_path, class_name=class_name, is_train=False)

    train_dl = DataLoader(train_data, BATCH_SIZE, num_workers=4, pin_memory=True, shuffle=True)
    test_dl = DataLoader(test_data, BATCH_SIZE, num_workers=4, pin_memory=True)



    test_data_classes = get_classes_items(test_data)

    device = get_device()

    train_dl = ToDeviceLoader(train_dl, device)
    test_dl = ToDeviceLoader(test_dl, device)

    model = ResNet(3,2)
    model = to_device(model, device)
    

    #train(model, epochs, train_dl, test_dl, optimizer, max_lr, grad_clip, weight_decay)


    load_weights(model, weigth_path)
    warm_up_path = r"C:\Users\Enzo\Pictures\image.png"

    with Image.open(Path(warm_up_path)).convert('RGB') as img:
                    transform_x = T.Compose([T.Resize(48, Image.LANCZOS),
                                                T.CenterCrop(48),
                                                T.ToTensor(),
                                                T.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
                    img = transform_x(img)

                    predict_image(img, model, test_data_classes, device)

    print("G;1200", flush=True)
    print(image_type, flush=True)

    #reduce the precision of each weight of the network
    #model.half()
    
    w = Watcher(check_path,model, test_data.classes, device, image_type)
    w.run()