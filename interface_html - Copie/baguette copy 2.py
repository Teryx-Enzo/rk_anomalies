from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
from torchvision import transforms 
from torchvision.transforms import v2
from utils import ToDeviceLoader, to_device, get_device, accuracy, train, load_pre_trained_weights, load_weights, Watcher, get_classes_items, predict_image
from models import ResNetWithGradCAM
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import re
import cv2


class_names = ['good', 'not good']  # Noms des classes
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}  # Mapping classe -> index



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
                for triplet in triplets:
                    self.image_paths.append(triplet)
                    self.labels.append(class_to_idx[label])  # Convertir l'étiquette en entier

    def group_triplets(self, class_dir):
        # Obtenir toutes les images du dossier
        images = glob.glob(os.path.join(class_dir, "*.png")) + \
                 glob.glob(os.path.join(class_dir, "*.jpg")) + \
                 glob.glob(os.path.join(class_dir, "*.jpeg")) + \
                 glob.glob(os.path.join(class_dir, "*.bmp"))
        
        # Grouper les images par préfixe commun
        triplets = {}
        for img_path in images:
            # Extraire le préfixe et le suffixe
            match = re.match(r".*&Cam1Img\.(\w+)", os.path.basename(img_path))
            #print(img_path, match)
            if match:
                #print(match.group(1).split('_'))
                suffix = match.group(1).split('_')[0]  # Par exemple "Shape1"
                prefix = match.group(1).split('_')[1]

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
        return merged_image, label  # Renvoie l'image et l'index de classe (entier)

def load_data(train_dir, test_dir, batch_size=32):

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = TripletImageDataset(root_dir=train_dir, transform=train_transform)
    test_dataset = TripletImageDataset(root_dir=test_dir, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


import torch
import torch.nn as nn
import torch.optim as optim

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, test_loader, device, epochs=10, max_lr=0.01, weight_decay=1e-4, grad_clip=None):
    optimizer = optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # Calcul de la perte moyenne pour l'entraînement
        train_loss_avg = torch.tensor(train_losses).mean().item()

        # Évaluation sur le set de validation
        val_result = evaluate(model, test_loader, device)
        val_loss = val_result['loss']
        val_accuracy = val_result['accuracy']

        if epoch > 10:
            if val_loss < np.min(history['val_loss']) or train_loss_avg < np.min(history['train_loss']):
                torch.save(model.state_dict(), 'rk-resnet-bs-16-48-'+str(epoch)+'.pth')

        # Sauvegarder les métriques dans l'historique
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    return history

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = torch.tensor(losses).mean().item()
    return {'accuracy': accuracy, 'loss': avg_loss}


def evaluate_on_test_set(model, test_loader, classes, device):
    y_true = []
    y_pred = []
    outputs_source = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = to_device(images, device)
            labels = to_device(labels, device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            outputs_source.extend(outputs.cpu().numpy())

    return outputs_source, y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    
    
    print(cm)



def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Tracer la perte
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Tracer l'exactitude
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_accuracy'], 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

def preprocess_image(img, transform):
    """
    Preprocess the input image (normalize, resize, etc.) using torchvision transforms.
    """
    return transform(img).unsqueeze(0)  # Add batch dimension

def generate_heatmap(model, image_tensor, class_idx=None):
    """
    Generates a heatmap for a single input image and the selected class index.
    
    Args:
    - model: The trained model with Grad-CAM support.
    - image_tensor: The preprocessed image tensor.
    - class_idx: Index of the class to generate the heatmap for. If None, use the predicted class.
    
    Returns:
    - heatmap: A normalized heatmap (numpy array) that can be visualized.
    - predicted_class: The predicted class index.
    """
    # Forward pass through the model
    output = model(image_tensor)
    
    # Get the predicted class
    predicted_class = torch.argmax(output, dim=1).item()
    
    # If class index is not provided, use the predicted class
    if class_idx is None:
        class_idx = predicted_class
    
    # Generate the heatmap for the selected class
    heatmap_tensor = model.get_heatmap(class_idx)
    heatmap = heatmap_tensor.squeeze().cpu().detach().numpy()
    
    # Resize the heatmap to match the input image size
    heatmap = cv2.resize(heatmap, (image_tensor.shape[2], image_tensor.shape[3]))
    
    return heatmap, predicted_class

def overlay_heatmap_on_image(heatmap, original_image, alpha=0.5):
    """
    Overlays the heatmap on the original image.
    
    Args:
    - heatmap: The heatmap (numpy array) normalized between 0 and 1.
    - original_image: The original image (numpy array in range [0, 1]).
    - alpha: Transparency factor for the heatmap overlay.
    
    Returns:
    - overlay_image: The image with heatmap overlaid.
    """
    # Remove NaNs or Infs by replacing them with 0
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Normalize the heatmap to the range [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Convert heatmap to RGB format using a colormap (e.g., JET)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = np.float32(heatmap_colored) / 255.0
    
    # Convert original image from (1, H, W) to (H, W, 3) and normalize to [0, 1]
    original_image = original_image.permute(1, 2, 0).cpu().numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    
    # Overlay the heatmap on the original image
    overlay_image = heatmap_colored * alpha + original_image * (1 - alpha)
    
    return overlay_image



if __name__ == "__main__":

    # Charger les données
    train_loader, test_loader = load_data(train_dir=r'C:\Users\Enzo\Pictures\dataset_on_tente_des_trucs\train', 
                                          test_dir=r'C:\Users\Enzo\Pictures\dataset_on_tente_des_trucs\test')

    # Charger votre modèle (par exemple ResNet)
    model = ResNetWithGradCAM(3, len(class_names))
    
    device = get_device()
    model.to(device)

    
    # Charger des poids pré-entraînés si nécessaire
    #load_pre_trained_weights(model, r"C:\Users\Enzo\Documents\Code_enzo\resnet18_test_\cifar100-resnet-project.pth")

    # Entraîner le modèle et obtenir l'historique
    #history = train(model, train_loader, test_loader, device, epochs=200, max_lr=0.001, weight_decay=1e-4, grad_clip=0.1)

    # Tracer l'historique de l'entraînement
    #plot_training_history(history)


    load_weights(model, r"C:\Users\Enzo\Documents\GitHub\poids_resnet_256_256_no_ transformation\rk-resnet-bs-16-48-181.pth")

    model.eval()  # Set model to evaluation mode

    # Load and preprocess a single image
    # Assume you have an image loaded as a PIL Image and a `transform` defined for preprocessing
    image = Image.open(r"C:\Users\Enzo\Pictures\image.jpg")  # Load your image here (e.g., PIL.Image.open('path_to_image'))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = preprocess_image(image, transform).to(device)

        
    # Generate the heatmap for the input image (and get the predicted class)
    heatmap, predicted_class = generate_heatmap(model, image_tensor)

    # Visualize the heatmap superimposed on the original image
    overlay_image = overlay_heatmap_on_image(heatmap, image_tensor.squeeze(0))  # Remove batch dimension for visualization

    # Display the image with heatmap and prediction
    plt.imshow(overlay_image)
    plt.title(f'Predicted Class: {predicted_class}')
    plt.axis('off')
    plt.show()

    # Test du modèle sur l'ensemble de test
    #outputs_source, y_true, y_pred = evaluate_on_test_set(model, test_loader, class_names, device)

    # Afficher la matrice de confusion
    #plot_confusion_matrix(y_true, y_pred, class_names)

    #same_list = []
    #diff_list = []
    #same_relat_diff = []
    #diff_relat_diff = []

    """
    for a,b,c in zip(outputs_source, y_true, y_pred):

        if b != c:
            print('Erreur ', max(a), abs(a[0]-a[1]))
            diff_list.append((max(a), abs(a[0]-a[1])))

        else :
            print('Les memes ', max(a), abs(a[0]-a[1]))
            same_list.append((max(a), abs(a[0]-a[1])))


    plt.figure()
    plt.scatter(*zip(*same_list))
    plt.scatter(*zip(*diff_list))
    plt.show()"""