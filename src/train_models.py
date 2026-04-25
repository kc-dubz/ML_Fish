import os
import torch
import kagglehub
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# -----------------------
# 1. Load dataset
# -----------------------
DATA_PATH = kagglehub.dataset_download("jorritvenema/affine")
DATA_PATH = os.path.join(DATA_PATH, "dataset") 

print("Dataset path:", DATA_PATH)

# -----------------------
# 2. Transforms
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)

print("Classes:", dataset.classes)

# -----------------------
# 3. Train / Val split
# -----------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# -----------------------
# 4. Model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

# Replace final layer for fish classes
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# -----------------------
# 5. Loss + Optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# 6. Training loop
# -----------------------
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # -----------------------
    # Validation
    # -----------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total

    print(f"\nEpoch {epoch+1} | Loss: {train_loss:.4f} | Val Acc: {acc:.4f}\n")

# -----------------------
# 7. Save model
# -----------------------
torch.save({
    "model_state": model.state_dict(),
    "classes": dataset.classes
}, "../models/fish_model_resnet34.pth")

print("Model saved as fish_model_resnet34.pth")