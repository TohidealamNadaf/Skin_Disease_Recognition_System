import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from datetime import datetime

# ✅ CONFIGURATION
CONFIG = {
    "dataset_path": "dataset",
    "image_size": (224, 224),
    "batch_size": 32,
    "epochs": 15,
    "model_path": "models/skin_disease_model.pth",
    "labels_pkl": "models/class_indices.pkl",
    "plot_path": "models/training_plot.png"
}

# ✅ GPU SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using Device: {device}")

# ✅ DATA TRANSFORMATION & LOADING
transform = transforms.Compose([
    transforms.Resize(CONFIG["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(CONFIG["dataset_path"] + "/train", transform=transform)
val_dataset = ImageFolder(CONFIG["dataset_path"] + "/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])

num_classes = len(train_dataset.classes)
joblib.dump(train_dataset.class_to_idx, CONFIG["labels_pkl"])  # Save class indices

# ✅ MODEL BUILDING (MobileNetV2)
model = models.mobilenet_v2(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(model.last_channel, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes),
    nn.Softmax(dim=1)
)

model = model.to(device)

# ✅ LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ TRAINING FUNCTION
def train_and_save():
    history = {"train_acc": [], "val_acc": []}
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        correct, total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        history["train_acc"].append(train_acc)

        # ✅ VALIDATION
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), CONFIG["model_path"])
    print(f"💾 Model saved to {CONFIG['model_path']}")

    plot_history(history)

# ✅ PLOT TRAINING RESULTS
def plot_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(history["val_acc"], label="Val Accuracy", marker="x")
    plt.title("Training Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(CONFIG["plot_path"])
    plt.close()
    print(f"📈 Training plot saved to {CONFIG['plot_path']}")

# ✅ MAIN EXECUTION
if __name__ == "__main__":
    print(f"🚀 Started Training @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    train_and_save()