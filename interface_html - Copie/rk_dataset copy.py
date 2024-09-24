import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASS_NAMES = ['RK_fusion']

class RKDataset(Dataset):
    def __init__(self, dataset_path=r'C:\Users\Enzo\Pictures\dataset', class_name='rk', is_train=True,
                 resize=48, cropsize=48):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.classes = {0 : 'G', 1 : 'NG'}

        # load dataset
        self.x, self.y, self.names, self.labels = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.LANCZOS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):

        
        x, y, name = self.x[idx], self.y[idx], self.names[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        # return a zero mask for all images since ground truth is not available
        mask = torch.zeros([1, self.cropsize, self.cropsize])

        if name.startswith('NG'):
            label = 1
        else:
            label = 0


        


        return x, label

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, names, label = [], [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.jpg')])
            x.extend(img_fpath_list)
            names.extend([os.path.basename(f) for f in img_fpath_list])

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                
            else:
                y.extend([1] * len(img_fpath_list))

            label.append(img_type)

            

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(names), list(label)

