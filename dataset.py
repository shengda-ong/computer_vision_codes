# assuming (file_path, label) in csv
import os
import csv

import cv2
import torch
from torch.utils.data import Dataset

class ImageClassifcationDataset(Dataset):
    def __init__(self,csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                img_path, img_label = row[0], row[1] # relative path, label
                label = int(img_label)
                self.samples.append((img_path,label))
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in csv file: {csv_file}")
        
    def __len__(self):
        return len(self.samples)
    
    def _load_image(self,img_path):
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
             
        img = cv2.cvtColor(img, cv2.BGR2RGB)
        img = img.astype("float32")/255.0 
        img = torch.from_numpy(img) # (H,w,C)
        img = img.permute(2,0,1) # (C,H,W)

        return img
    
    def __getitem__(self,idx):
        img_path, label = self.samples[idx]
        img_path = os.path.join(self.root_dir,img_path)
        
        image = self._load_image(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image,label

