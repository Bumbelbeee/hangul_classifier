from PIL import Image, ImageOps
import os 
import pandas as pd
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset
from preprocess import preprocess
EPOCHS = 10
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        conv1 = nn.Conv2d(in_channels=1, out_channels=32,kernel_size=3, padding=1)
        conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3, padding=1)
        conv3 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3, padding=1)
        batchnorm1 = nn.BatchNorm2d(num_features=32)
        batchnorm2 = nn.BatchNorm2d(num_features=64)
        batchnorm3 = nn.BatchNorm2d(num_features=128)
        relu = nn.ReLU()
        maxpool = nn.MaxPool2d(kernel_size=2)
        flatten = nn.Flatten()
        fc = nn.Linear(8192, num_classes)
        
        self.layers = nn.Sequential(
            conv1, batchnorm1, relu, maxpool,
            conv2, batchnorm2, relu, maxpool, 
            conv3, batchnorm3, relu, maxpool, 
            flatten, fc)

    def forward(self, x):
        return self.layers(x)
        

        

def main():
    if not(os.path.exists("data.pt")):
        preprocess()
    
    data = torch.load("data.pt")
    classes = data["classes"]
    dataset = TensorDataset(data["images"], data["labels"])
    # Train Test Split

    train_data, test_data = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers= os.cpu_count() // 2, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=64, num_workers= os.cpu_count() // 2, persistent_workers=True, pin_memory=True)

    # Test if the transformer is correct
    """
    1. The first try was very bad you cannot recognize the symbols the rezise may be to harsh I comment it out
    2. Without the rezise it is much better but every pixel has large borders i might be able to strech it first into the borders and then rezise it
    3. I Changed the Dataset now and created a box around everything that has pixels and then expanded it a couple of pixels to 64, 64
    4. You can also see noise in these images but this might makes it even more robust
    5. I optimized it further usung mutliple processes to load the data and training it on my graphics card
    """

    model = CNN(len(classes))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Train using: {device}")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # TRAINING
    model.train()
    losses = []
    evals = []
    for _ in range(EPOCHS):
        for step, (image, label) in enumerate(train_loader):
            image = image.to(device).float() / 255
            label = label.to(device)
            # Predict the Label based on image
            prediction = model(image)
            # Calculate error
            loss = loss_fn(prediction, label)
            # Delete Gradient
            optimizer.zero_grad()
            # Calculate Gradient
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(loss.item())
                losses.append(loss)
        evals.append(evaluate(model=model, loader=test_loader, device=device))
    torch.save(model.state_dict(), "model.pt")


def evaluate(model, loader, device):
    model.eval()
    hits = 0
    with torch.no_grad():
        for image, label in loader:
            image = image.to(device).float() / 32
            label = label.to(device)
            prediction = model(image)
            prediction = prediction.argmax(dim=1)
            hits += (prediction == label).sum().item()
    eval = hits / (len(loader.dataset))
    print(f"Evaluation: {eval}")
    return eval

if __name__ == "__main__":
    main()