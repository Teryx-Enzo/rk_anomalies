import argparse
import torch
from torch.utils.data import DataLoader
from rk_dataset import RKDataset
from models import ResNet
from utils import ToDeviceLoader, to_device, get_device, train, load_weights
from pathlib import Path



if __name__ == "__main__":
    print("Début de l'entraînement du modèle", flush=True)

    parser = argparse.ArgumentParser(description='Params of the script')
    parser.add_argument('-model_name', metavar='-m', type=str, help='name of the model file')
    parser.add_argument('-data_path', metavar='-d', type=str, help='path to the data directory')

    args = parser.parse_args()

    model_name = args.model_name
    data_path = args.data_path
    class_name = 'RK_fusion'

    epochs = 100
    BATCH_SIZE = 16
    optimizer = torch.optim.Adam
    max_lr = 5e-4
    grad_clip = 0.1
    weight_decay = 1e-5
    scheduler = torch.optim.lr_scheduler.OneCycleLR

    train_data = RKDataset(data_path, class_name=class_name, is_train=True)
    test_data = RKDataset(data_path, class_name=class_name, is_train=False)

    train_dl = DataLoader(train_data, BATCH_SIZE, num_workers=4, pin_memory=True, shuffle=True)
    test_dl = DataLoader(test_data, BATCH_SIZE, num_workers=4, pin_memory=True)
    
    device = get_device()
    train_dl = ToDeviceLoader(train_dl, device)
    test_dl = ToDeviceLoader(test_dl, device)
    
    model = ResNet(3, 2)
    model = to_device(model, device)
    
    weight_path = Path(r"C:\Users\Enzo\Documents\Code_enzo\resnet18_test_") / model_name
    load_weights(model, weight_path)

    print(f"Entraînement avec le modèle {model_name}", flush=True)
    train(model, epochs, train_dl, test_dl, optimizer, max_lr, grad_clip, weight_decay)

    print("Entraînement terminé", flush=True)