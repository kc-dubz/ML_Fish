import os
import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import kagglehub


# -----------------------
# CONFIG: change this
# -----------------------
MODEL_PATH = "../models/fish_model_resnet18.pth"   # <-- change to fish_model_resnet34.pth if needed
BATCH_SIZE = 32


# -----------------------
# 1. Load dataset (same as training for now)
# -----------------------
DATA_PATH = kagglehub.dataset_download("jorritvenema/affine")
DATA_PATH = os.path.join(DATA_PATH, "dataset")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", dataset.classes)
print("Dataset size:", len(dataset))

# -----------------------
# 2. Load model checkpoint
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)

num_classes = len(checkpoint["classes"])


# -----------------------
# 3. Rebuild model (Depends on wich one you load in)
# -----------------------
model = models.resnet18(weights=None)  # < ------------------- change this to resnet18 or resnet34 depending on wich one is used
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

print("Model loaded from:", MODEL_PATH)


# -----------------------
# 4. Evaluation loop
# -----------------------
correct = 0
total = 0

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total

print("\n====================")
print(f"TEST ACCURACY: {accuracy:.4f}")
print("====================")