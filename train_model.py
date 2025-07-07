# train_model.py (Fixed + Enhanced for accurate Parkinson’s handwriting detection)
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from collections import Counter
import numpy as np

# Paths
data_dir = "dataset"  # Folder should have 'Static/' and 'Dynamic/'
model_save_path = "model_weights.pth"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transforms (with augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load dataset
dataset = ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes
print("Classes:", class_names)  # ['Dynamic', 'Static']

# Count class samples
label_counts = Counter([label for _, label in dataset])
print("Class distribution:", label_counts)

# Balanced sampling
class_weights = 1. / torch.tensor([label_counts[i] for i in range(len(class_names))], dtype=torch.float)
sample_weights = [class_weights[label] for _, label in dataset]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Split train/test
from torch.utils.data import random_split
train_len = int(0.8 * len(dataset))
test_len = len(dataset) - train_len
train_data, test_data = random_split(dataset, [train_len, test_len])

train_loader = DataLoader(train_data, batch_size=16, sampler=WeightedRandomSampler(
    [sample_weights[i] for i in train_data.indices], len(train_data), replacement=True))
test_loader = DataLoader(test_data, batch_size=16)

# Define model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 15
print("\nTraining started...\n")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Save model for app use
torch.save(model.state_dict(), model_save_path)
print(f"\n✅ Model saved to {model_save_path}")
