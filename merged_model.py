import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =========================
# DEVICE
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LABELS
# =========================
multi_labels = ['N','D','G','C','A','H','M','O']

dr_severity_labels = [
    'Normal',
    'Mild',
    'Moderate',
    'Severe'
]

# =========================
# LOAD MULTI-DISEASE MODEL
# =========================
multi_model = models.efficientnet_b3(weights=None)
multi_model.classifier[1] = nn.Linear(
    multi_model.classifier[1].in_features, 8
)

multi_model.load_state_dict(
    torch.load("multi_disease_b3.pth", map_location=DEVICE)
)

multi_model.to(DEVICE)
multi_model.eval()

# =========================
# LOAD DR SEVERITY MODEL
# =========================
dr_model = models.efficientnet_b3(weights=None)
dr_model.classifier[1] = nn.Linear(
    dr_model.classifier[1].in_features, 4
)

dr_model.load_state_dict(
    torch.load("best_retina_model.pth", map_location=DEVICE)
)

dr_model.to(DEVICE)
dr_model.eval()

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

# =========================
# PREDICTION FUNCTION
# =========================
def predict(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        # ---- MULTI DISEASE ----
        multi_outputs = multi_model(image)
        multi_probs = torch.sigmoid(multi_outputs)[0]

        print("\n===== MULTI DISEASE PROBABILITIES =====")
        for i, label in enumerate(multi_labels):
            print(f"{label}: {multi_probs[i].item():.4f}")

        # Demo thresholds
        global_threshold = 0.4
        dr_threshold = 0.45

        detected = []

        for i, label in enumerate(multi_labels):
            prob = multi_probs[i].item()

            if label == 'D':
                if prob > dr_threshold:
                    detected.append(label)
            else:
                if prob > global_threshold:
                    detected.append(label)

        print("\n===== MULTI DISEASE RESULT =====")
        if not detected:
            detected = ['N']
        print("Detected:", detected)

        # ---- DR SEVERITY ----
        if 'D' in detected:

            dr_outputs = dr_model(image)
            dr_probs = torch.softmax(dr_outputs, dim=1)[0]
            dr_class = torch.argmax(dr_probs).item()

            print("\n===== DR SEVERITY RESULT =====")
            print("Prediction:", dr_severity_labels[dr_class])

            print("\nClass Probabilities:")
            for i, label in enumerate(dr_severity_labels):
                print(f"{label}: {dr_probs[i].item():.4f}")

        else:
            print("\nNo DR detected. Severity model not triggered.")


# =========================
# RUN TEST
# =========================
if __name__ == "__main__":
    predict("test_retina.jpg.jpeg")  # change to your test image