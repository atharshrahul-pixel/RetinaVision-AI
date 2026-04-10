import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate architecture
model = efficientnet_b3(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)

# Load weights
model.load_state_dict(torch.load("best_retina_model.pth", map_location=device))

model.to(device)
model.eval()

print("✅ Model loaded successfully. Backup confirmed.")