from dataset import ImageClassifcationDataset
from torch.utils.data import DataLoader

csv_file = ""
root_dir = ""

dataset = ImageClassifcationDataset(
    csv_file=csv_file,
    root_dir=root_dir
)

loader = DataLoader(dataset,batch_size=32, shuffle=True)
for image, label in loader:
    break
