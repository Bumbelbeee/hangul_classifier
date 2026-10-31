from PIL import Image, ImageOps
import os 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.transforms import v2
import torch

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
        sample = ImageOps.pad(crop, (56, 56))
        sample = ImageOps.expand(sample, border=4)
        sample = self.transformer(sample)


        label = self.maping[self.labels[index]]
        return sample, label


def preprocess():
    cwd = os.getcwd()
    path = os.path.join(cwd, "hangul_characters_250k")
    path_data = os.path.join(path, "labels-map.csv")
    data = pd.read_csv(path_data, header=None, names=["path", "symbol"])

    transform = v2.Compose([
    v2.ToImage(), # Transforms into a tensor

    #v2.ToDtype(torch.float32, scale=True), # Transforms the Tensor into scaled float32
    ])

    # Label Mapping
    classes = sorted(data["symbol"].unique())

    label_maping = {symbol: idx for idx, symbol in enumerate(classes)}


    dataset = HangulDataset(data["path"], data["symbol"], path, transform, label_maping)
    loader = DataLoader(dataset, batch_size=248, num_workers=os.cpu_count() // 2)
    images = []
    labels = []
    for image, label in loader:
        images.append(image)
        labels.append(label)
    tensors_image = torch.cat(images)
    tensors_label = torch.cat(labels)
    torch.save({"images": tensors_image, "labels": tensors_label, "classes": classes}, "data.pt")