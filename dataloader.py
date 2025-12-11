from dataset import ImageClassifcationDataset
from torch.utils.data import DataLoader
from transform import Compose, RandomHorizontalFlip, Resize, Normalise
csv_file = ""
root_dir = ""

transform = Compose([
    Resize((244,244)),
    RandomHorizontalFlip(p=0.5),
    Normalise(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet values
])

dataset = ImageClassifcationDataset(
    csv_file=csv_file,
    root_dir=root_dir,
    transform=transform
)

loader = DataLoader(dataset,batch_size=32, shuffle=True)
for image, label in loader:
    break
