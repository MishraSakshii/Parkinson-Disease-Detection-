import torch
import torch.nn as nn
import torchvision.models as models

# Create dummy model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Static, Dynamic

# Save dummy model to disk
torch.save(model, "model.pth")
print("Dummy model saved as model.pth")
