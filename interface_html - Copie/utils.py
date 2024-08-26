import torch
import torch.nn as nn
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image
from torchvision import transforms as T
from pathlib import Path
from io import BytesIO
from server_http import update_value
import csv


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)


def accuracy(predicted, actual):
    _, predictions = torch.max(predicted, dim=1)
    return torch.tensor(torch.sum(predictions==actual).item()/len(predictions))

class ToDeviceLoader:
    def __init__(self,data,device):
        self.data = data
        self.device = device
        
    def __iter__(self):
        for batch in self.data:
            yield to_device(batch,self.device)
            
    def __len__(self):
        return len(self.data)
    

@torch.no_grad()
def evaluate(model,test_dl):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_dl]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit (epochs, train_dl, test_dl, model, optimizer, max_lr, weight_decay, scheduler, grad_clip=None):
    torch.cuda.empty_cache()
    
    history = []
    
    optimizer = optimizer(model.parameters(), max_lr, weight_decay = weight_decay)
    
    scheduler = scheduler(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_dl))
    
    for epoch in range(epochs):
        model.train()
        
        train_loss = []
        
        lrs = []
        
        for batch in train_dl:
            loss = model.training_step(batch)
            
            train_loss.append(loss)
            
            loss.backward()
            
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step()
            lrs.append(get_lr(optimizer))
        result = evaluate(model, test_dl)
        result["train_loss"] = torch.stack(train_loss).mean().item()
        result["lrs"] = lrs
        
        model.epoch_end(epoch,result)
        history.append(result)
        
    return history

def get_classes_items(test_data):

    test_classes_items = dict()
    for test_item in test_data:
        label = test_data.classes[test_item[1]]
        if label not in test_classes_items:
            test_classes_items[label] = 1
        else:
            test_classes_items[label] += 1

    return test_classes_items


def train(model_courant, epochs, train_dl, test_dl, optimizer, max_lr, grad_clip, weight_decay):

    history = fit(epochs=epochs, train_dl=train_dl, test_dl=test_dl, model=model_courant, 
                optimizer=optimizer, max_lr=max_lr, grad_clip=grad_clip,
                weight_decay=weight_decay, scheduler=torch.optim.lr_scheduler.OneCycleLR)
    

    torch.save(model_courant.state_dict(), 'rk-resnet-project_new.pth')
    return history

def load_pre_trained_weights(model, weights_path):


    pretrained_dict = torch.load(weights_path)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in ['classifier.2.weight', 'classifier.2.bias']}

    for k, v in model_dict.items():
        if k in  ['classifier.2.weight', 'classifier.2.bias']:
            pretrained_dict[k] = v
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)

def load_weights(model, weights_path):

    trained_dict = torch.load(weights_path)
    model_dict = model.state_dict()

    # 2. overwrite entries in the existing state dict
    model_dict.update(trained_dict) 
    # 3. load the new state dict
    model.load_state_dict(trained_dict)


def predict_image(img, model, test_data_classes, device):


    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)

    TRUC = preds[0].item()

    return TRUC




class Watcher:
    
    def __init__(self,directory_to_watch, model, test_data_classes, device, image_type):
        self.DIRECTORY_TO_WATCH = directory_to_watch
        self.model = model
        self.observer = Observer()
        self.test_data_classes = test_data_classes
        self.device = device
        self.image_type = image_type


    def run(self):
        event_handler = Handler(self.model, self.test_data_classes, self.device, self.image_type)
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=False)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            self.observer.stop()
        self.observer.join()

class Handler(FileSystemEventHandler):

    def __init__(self, model, test_data_classes, device, image_type):
        self.model = model
        self.test_data_classes = test_data_classes
        self.device = device
        self.image_paths = []
        self.image_type = image_type
        self.t0 = 0
        self.output = ""
        self.compteur = 0

        self.suffixes = ["&Cam1Img.GlossRatio{imagetype}".format(imagetype=self.image_type),"&Cam1Img.Normal{imagetype}".format(imagetype=self.image_type),"&Cam1Img.Shape1{imagetype}".format(imagetype=self.image_type)]

        self.transform = T.Compose([
                T.Resize(48, Image.LANCZOS),
                T.CenterCrop(48),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def process(self, event):
        if not self.t0:
            self.t0 = time.time()
        if event.event_type == 'created' and event.src_path.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            self.image_paths.append(Path(event.src_path))
            pred = self.process_images()
            
            if pred or pred==0 : 
                update_value(pred)
                status= self.test_data_classes[pred]
                elapsed_time = int((time.time()-self.t0)*1000)
                self.t0 = 0
                
                
                self.output += f'Temps d\'exécution pour itération {self.compteur + 1}: {elapsed_time:.2f} ms \n'
                self.compteur += 1

                if self.compteur % 10 == 0:
                    print(self.output, flush = True)
                else:
                    print(f'{status};{elapsed_time}', flush=True)   

    
    def process_images(self):

        if len(self.image_paths) >= 3:
            triplet_images = [None, None, None]
            
            for path in self.image_paths[:3]:
                fin =  str(path).split('_')[-1]
                if fin == self.suffixes[0]:
                    index = 0
                elif fin == self.suffixes[1]:
                    index = 1
                elif fin == self.suffixes[2]:
                    index = 2
        
                try:
                    with Image.open(path).convert('L') as img:  # Convertir en niveau de gris
                        triplet_images[index] = img
                except Exception as e:
                    print(f"Erreur lors de la lecture du fichier {path}: {e}")
                    return
                
            
            
            # Fusionner les trois images en une seule avec trois canaux
            merged_image = Image.merge('RGB', triplet_images)

            # Appliquer les transformations et prédire
            
            img_tensor = self.transform(merged_image)
            
            pred= predict_image(img_tensor, self.model, self.test_data_classes, self.device)

            # Sauvegarder la prediction et le nom des images dans un csv

            # Chemin du fichier CSV
            filename = "output.csv"

            data = self.image_paths[:3] + [pred]

            print(data, flush = True)
            for i in range(len(data)):

                data[i] = str(data[i]).split("\\")[-1]
                print(data[i], flush = True)
            # Écriture dans le fichier CSV
            with open(filename, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(data)


            # Retirer les trois premières images de la liste
            self.image_paths = self.image_paths[3:]

            return pred
            

    def on_created(self, event):
        self.process(event)
    
