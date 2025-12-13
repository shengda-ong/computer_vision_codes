import torch
from torch import nn
from torch import optim
from dataloader import train_loader
from simple_cnn import SimpleCNN
import shutil
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASS = 10
NUM_EPOCH = 30
LR = 0.01

model = SimpleCNN(num_classes=NUM_CLASS).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=LR)

terminal_width = shutil.get_terminal_size().columns
message = "STARTING TRAINING"
print(f"--- {message} ".center(terminal_width, "-"))

for epoch in range(NUM_EPOCH):
    model.train()
    running_loss = 0.0

    for i, (images,labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # zero gradient parameter
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs,labels)

        # backward propagation
        loss.backwards()
        optimizer.step()

        running_loss += loss.items()

    print(f"Epoch {epoch+1}/{NUM_EPOCH}, loss: {running_loss/len(train_loader):.4f} ")

message = "END TRAINING"
print(f"--- {message} ".center(terminal_width, "-"))