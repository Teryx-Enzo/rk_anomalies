from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
from torchvision import transforms 
from utils import ToDeviceLoader, to_device, get_device, accuracy, train, load_pre_trained_weights, load_weights, Watcher, get_classes_items, predict_image
from models import ResNet
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import re

class_names = ['good', 'not good']  # Mettez ici les noms des classes
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}  # Crée un mapping classe -> index


def evaluate_on_test_set(model, test_loader, classes, device):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = to_device(images, device)
            labels = to_device(labels, device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print(cm)


class TripletImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        # Charger les images et les étiquettes
        self.load_images_and_labels()
    
    def load_images_and_labels(self):
        for label in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(class_dir):
                triplets = self.group_triplets(class_dir)
                #print(triplets)
                for triplet in triplets:
                    self.image_paths.append(triplet)
                    self.labels.append(class_to_idx[label])
    def group_triplets(self, class_dir):
        # Obtenir toutes les images du dossier
        images = glob.glob(os.path.join(class_dir, "*.png")) + \
                 glob.glob(os.path.join(class_dir, "*.jpg")) + \
                 glob.glob(os.path.join(class_dir, "*.jpeg")) + \
                 glob.glob(os.path.join(class_dir, "*.bmp"))
        
        # Grouper les images par préfixe commun
        triplets = {}
        for img_path in images:
            # Extraire le suffixe et le préfixe
            match = re.match(r".*&Cam1Img\.(\w+)", os.path.basename(img_path))
            #print(img_path, match)
            if match:
                suffix = match.group(1).split('_')[0]  # Par exemple "Shape1"
                prefix = match.group(1).split('_')[1]
                
                #print(suffix, prefix)
                if prefix not in triplets:
                    triplets[prefix] = [None, None, None]  # Trois places pour les trois types d'images
            
                if suffix == "GlossRatio":
                    triplets[prefix][0] = img_path
                elif suffix == "Normal":
                    triplets[prefix][1] = img_path
                elif suffix == "Shape1":
                    triplets[prefix][2] = img_path
        
        return [paths for paths in triplets.values() if None not in paths]  # Retirer les groupes incomplets

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        triplet_paths = self.image_paths[idx]
        images = [Image.open(path).convert("L") for path in triplet_paths]  # Convertir en niveaux de gris
        
        merged_image = Image.merge("RGB", images)  # Fusionner les images en une seule avec trois canaux
        
        if self.transform:
            merged_image = self.transform(merged_image)
        
        label = self.labels[idx]
        return merged_image, int(label)

def load_test_data(test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = TripletImageDataset(root_dir=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader, test_dataset.labels


if __name__ == "__main__":

    # Charger les données de test
    test_loader, class_names = load_test_data(r'C:\Users\Enzo\Pictures\image_27_08_2024')


    #print(class_names)
    # Charger le modèle
    model = ResNet(3,2)  # Remplacez YourModelClass par votre classe de modèle
    load_weights(model, r"C:\Users\Enzo\Documents\Code_enzo\resnet18_test_\rk-resnet-project.pth")

    # Déplacer le modèle sur le périphérique adéquat
    device = get_device()
    model = model.to(device)

    # Test du modèle
    y_true, y_pred = evaluate_on_test_set(model, test_loader, class_names, device)

    #print(y_true, y_pred)

    plot_confusion_matrix(y_true, y_pred, class_names)