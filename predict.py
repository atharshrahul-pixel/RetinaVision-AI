import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import sys

# --------- Class Names ----------
class_names = ["Normal", "Mild", "Moderate", "Severe"]

# --------- Load Model ----------
model = models.efficientnet_b3(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load("best_retina_model.pth", map_location="cpu"))
model.eval()

# --------- Image Transform ----------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

# --------- Get Image Path ----------
image_path = sys.argv[1]

# --------- Load Image ----------
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# --------- Inference ----------
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = F.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

# --------- Print Results ----------
print("\nPrediction:", class_names[predicted.item()])
print("Confidence:", round(confidence.item() * 100, 2), "%")

print("\nAll Class Probabilities:")
for i, prob in enumerate(probabilities[0]):
    print(f"{class_names[i]}: {prob.item() * 100:.2f}%")