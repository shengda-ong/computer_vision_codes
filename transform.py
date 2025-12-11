import torch
import torch.nn.functional as F
# chaining transform
class Compose:
    def __init__(self,transforms):
        self.transforms = transforms
    def __call__(self,x):
        for t in self.transforms:
            x = t(x)
        return x

class RandomHorizontalFlip:
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self,img):
        if torch.rand(1).item.item() < self.p:
            img = torch.flip(img,dims=[2])
        return img
    
class Resize:
    def __init__(self,size):
        self.size = size # (new_h, new_w)

    def __call__(self):
        img = img.unsqueeze(0) # (1, C, H, W)
        img = F.interpolate(img, size=self.size, mode='bilinear', align_corners=False)
        img = img.squeeze(0) # (C, H, W)
        return img
    
class Normalise:
    def __init__(self,mean,std):
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)

    def __call__(self, img):
        return (img - self.mean)/ self.std
        
               
        