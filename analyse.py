from main import CNN
import os
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import matplotlib.pyplot as plt

def analyse():
    if not(os.path.exists("data.pt")):
        print("You need preprocess and train first.")
        return

    # Load the Data
    data = torch.load("data.pt")
    classes = data["classes"]
    dataset = TensorDataset(data["images"], data["labels"])
    _, _, test_data = random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_data, batch_size=64, num_workers= os.cpu_count() // 2, persistent_workers=True, pin_memory=True)

    # Initialize the Model
    model = CNN(len(classes))
    model.load_state_dict(torch.load("model.pt"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    wrong_images = []
    real_labels = []
    predictions = []
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device).float() / 255
            label = label.to(device)
            prediction = model(image)
            prediction = prediction.argmax(dim=1)
            wrong = prediction != label
            # Store alle Images, Labels and Predictions
            wrong_images.append(image[wrong].cpu())
            real_labels.append(label[wrong].cpu())
            predictions.append(prediction[wrong].cpu())

    wrong_images = torch.cat(wrong_images)
    real_labels = torch.cat(real_labels)
    predictions = torch.cat(predictions)
    #print(f"Wrong Images: {len(wrong_images)}")
    #print(f"Real Labels: {len(real_labels)}")
    #print(f"Predictions: {len(predictions)}")

    pairs = zip(real_labels.tolist(), predictions.tolist())
    # Count the most common pairs
    counted_pairs = Counter(pairs).most_common(20)
    for (real, guess), count in counted_pairs:
        print(f"{classes[real]} ---> {classes[guess]}: {count}")


    # Sample Pictures

    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, axes = plt.subplots(4, 5, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(wrong_images[i][0], cmap="gray")
        ax.axis("off")
        ax.set_title(f"{classes[real_labels[i].item()]} --- >{classes[predictions[i].item()]}")
    fig.tight_layout() 
    fig.savefig("images/errors_128.png")
    plt.show()

        





if __name__ == "__main__":
    analyse()