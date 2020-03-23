import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import re
import torch
class ListDataSet(Dataset):
    def __init__(self,phrase,image_path,transforms=None):
        self.phrase=phrase
        with open(image_path,"r") as f:
            self.imgs = f.readlines()
            self.labels=[int(re.search("/\d/",path).group()[1]) for path in self.imgs]
        self.transforms=transforms
        
    
    def __getitem__(self,index):
        img_path=self.imgs[index].rstrip()
        pil_img = Image.open(img_path).convert("RGB")
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        label=self.labels[index]
        sample = (data,label)
        return sample
    def __len__(self):
        return len(self.imgs)

    
            
            
