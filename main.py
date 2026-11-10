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
EPOCHS = 40
class CNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        conv1 = nn.Conv2d(in_channels=1, out_channels=32,kernel_size=3, padding=1)
        conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3, padding=1)
        conv3 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3, padding=1)
        conv4 = nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3, padding=1)
        batchnorm1 = nn.BatchNorm2d(num_features=32)
        batchnorm2 = nn.BatchNorm2d(num_features=64)
        batchnorm3 = nn.BatchNorm2d(num_features=128)
        batchnorm4 = nn.BatchNorm2d(num_features=256)
        relu = nn.ReLU()
        maxpool = nn.MaxPool2d(kernel_size=2)
        flatten = nn.Flatten()
        dropout = nn.Dropout(0.5)
        #fc = nn.Linear(8192, num_classes) with 64 x 64
        fc = nn.Linear(16384, num_classes) # 128 × 16 × 16

        
        self.layers = nn.Sequential(
            conv1, batchnorm1, relu, maxpool,
            conv2, batchnorm2, relu, maxpool, 
            conv3, batchnorm3, relu, maxpool, 
            conv4, batchnorm4, relu, maxpool,
            flatten, dropout, fc)

    def forward(self, x):
        return self.layers(x)
        

        

def main():
    if not(os.path.exists("data.pt")):
        print("Preprocessing")
        preprocess()
    
    data = torch.load("data.pt")
    classes = data["classes"]
    dataset = TensorDataset(data["images"], data["labels"])
    # Train Test Split

    train_data, validation_data, test_data = random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers= os.cpu_count() // 2, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=64, num_workers= os.cpu_count() // 2, persistent_workers=True, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=64, num_workers= os.cpu_count() // 2, persistent_workers=True, pin_memory=True)
    # Test if the transformer is correct


    model = CNN(len(classes))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Train using: {device}")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # TRAINING
    
    losses = []
    test_losses = []
    evals = []
    for i in range(EPOCHS):
        model.train()
        epoch_loss = 0
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
            epoch_loss += loss.item()
            #äif step % 100 == 0:
                #print(loss.item())


        losses.append(epoch_loss / len(train_loader))
        acc, test_loss = evaluate(model=model, loader=validation_loader, device=device, loss_fn=loss_fn)
        if (i == 0):
            torch.save(model.state_dict(), "model.pt")
        elif(test_loss < min(test_losses)):
            torch.save(model.state_dict(), "model.pt")
        evals.append(acc)
        test_losses.append(test_loss)
        if(i > 2):
            if((test_losses[-4] - test_losses[-1]) < 0.001):
                break
        
    model.load_state_dict(torch.load("model.pt"))
    acc, test_loss = evaluate(model=model, loader=test_loader, device=device, loss_fn=loss_fn)

    # Visualisiations

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(losses, label="Train")
    axes[0].plot(test_losses, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()


    axes[1].plot(evals)
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_ylabel("Accuracy")

    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    fig.savefig("images/training_earlystopping_2.png")
    plt.show()


def evaluate(model, loader, device, loss_fn):
    model.eval()
    hits = 0
    loss = 0
    with torch.no_grad():
        for image, label in loader:
            image = image.to(device).float() / 255
            label = label.to(device)
            prediction = model(image)
            loss += loss_fn(prediction, label).item()
            prediction = prediction.argmax(dim=1)
            hits += (prediction == label).sum().item()
    loss = loss/len(loader)
    eval = hits / (len(loader.dataset))
    print(f"Evaluation: {eval}")
    return eval, loss

if __name__ == "__main__":
    main()