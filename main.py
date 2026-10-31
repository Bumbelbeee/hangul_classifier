from PIL import Image, ImageOps
import os 
import pandas as pd
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn

EPOCHS = 1
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(10368, num_classes)


        self.layers = [self.conv1, self.relu, self.maxpool, 
                       self.conv2, self.relu, self.maxpool, 
                       self.conv3, self.relu, self.maxpool, 
                       self.flatten, self.fc]      

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
        
class HangulDataset(Dataset):
    def __init__(self, data, labels, path_data, transformer, maping):
        self.data = data
        self.labels = labels
        self.path_data = path_data
        self.transformer = transformer
        self.maping = maping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = Image.open(os.path.join(self.path_data, self.data[index])).convert("L")
        bbox = sample.getbbox()
        crop = sample.crop(bbox)
        sample = ImageOps.pad(crop, (74, 74))
        sample = self.transformer(sample)


        label = self.maping[self.labels[index]]
        return sample, label
        

def main():
    # Current Dir
    cwd = os.getcwd()
    path = os.path.join(cwd, "hangul_characters_250k")
    path_data = os.path.join(path, "labels-map.csv")
    data = pd.read_csv(path_data, header=None, names=["path", "symbol"])

    transform = v2.Compose([
    v2.ToImage(), # Transforms into a tensor
    #v2.Resize((64,64)), # Resize the tensor from 255, 255 to 64, 64
    v2.ToDtype(torch.float32, scale=True), # Transforms the Tensor into scaled float32
    ])

    # Label Mapping
    classes = sorted(data["symbol"].unique())

    label_maping = {symbol: idx for idx, symbol in enumerate(classes)}


    dataset = HangulDataset(data["path"], data["symbol"], path, transform, label_maping)

    # Train Test Split
    train_data, test_data = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    # Test if the transformer is correct
    """
    1. The first try was very bad you cannot recognize the symbols the rezise may be to harsh I comment it out
    2. Without the rezise it is much better but every pixel has large borders i might be able to strech it first into the borders and then rezise it
    3. I Changed the Dataset now and created a box around everything that has pixels and then expanded it a couple of pixels to 64, 64
    4. You can also see noise in these images but this might makes it even more robust
    """

    model = CNN(len(classes))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Train using: {device}")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    model.train()

    for _ in range(EPOCHS):
        for step, (image, label) in enumerate(train_loader):
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



if __name__ == "__main__":
    main()