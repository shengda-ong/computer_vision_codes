from dataset import ImageClassifcationDataset
from torch.utils.data import DataLoader
from transform import Compose, RandomHorizontalFlip, Resize, Normalise

root_dir = ""
train_csv = ""
val_csv = ""


train_transform = Compose([
    Resize((244,244)),
    RandomHorizontalFlip(p=0.5),
    Normalise(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet values
])

val_transform = Compose([
    Resize((244,244)),
    Normalise(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet values
])

train_dataset = ImageClassifcationDataset(
    csv_file=train_csv,
    root_dir=root_dir,
    transform=train_transform
)

val_dataset = ImageClassifcationDataset(
    csv_file=val_csv,
    root_dir=root_dir,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle = True,
    num_workers = 4,
    pin_memory = True # enable faster cpu to gpu transfer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle = True,
    num_workers = 4,
    pin_memory = True # enable faster cpu to gpu transfer
)
