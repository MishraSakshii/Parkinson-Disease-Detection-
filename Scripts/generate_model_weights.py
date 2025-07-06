import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

# Save weights only
torch.save(model.state_dict(), "model_weights.pth")
print("Saved model_weights.pth")
