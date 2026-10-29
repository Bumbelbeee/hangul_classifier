from PIL import Image, ImageOps
import os 
import pandas as pd
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, random_split

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
    v2.ToImage(),
    v2.Resize((64,64)),
    v2.ToDtype(torch.float32, scale=True),
    ])

    # Label Mapping
    classes = sorted(data["symbol"].unique())

    label_maping = {symbol: idx for idx, symbol in enumerate(classes)}


    dataset = HangulDataset(data["path"], data["symbol"], path, transform, label_maping)

    # Train Test Split
    train_data, test_data = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)








if __name__ == "__main__":
    main()