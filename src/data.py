import torch.utils.data as data
import pandas as pd
import numpy as np
from PIL import Image
import os, glob
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode

class SquareDataset(data.Dataset):
    def __init__(
            self, 
            data_dir,
            size=(128, 128),
            mean=None,
            std=None,
        ):
        self.data_dir = data_dir
        self.label_mapping = {
            'a': 0,
            'b': 1,
            'c': 2,
        }

        self.init_csv()
        if (mean is None) and (std is None):
            self.transform = Compose([
                ToTensor(),
                Resize(size, interpolation=InterpolationMode.NEAREST),
            ])
        else:
            self.transform = Compose([
                ToTensor(),
                Resize(size, interpolation=InterpolationMode.NEAREST),
                Normalize(mean, std),
            ])
        
    def init_csv(self):
        info = {'image': [], 'label': []}
        classes = [f for f in os.listdir(self.data_dir) if not f.startswith('.')]
        for cls in classes:
            for img in [f for f in os.listdir(os.path.join(self.data_dir, cls)) if not f.startswith('.')]:
                img_dir = os.path.join(self.data_dir, cls, img)
                info['image'].append(img_dir)
                info['label'].append(self.label_mapping[cls])
        self.df = pd.DataFrame(info)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image']
        img = Image.open(img_path)
        img = self.transform(img)
        label = self.df.iloc[idx]['label']
        return img, label
    
# unit test
if __name__ == '__main__':
    data_dir = '/home/fang/Square_Task/squares/val'
    dataset = SquareDataset(data_dir)
    for i in range(10):
        img, label = dataset[i]
        print(img.shape, label)
