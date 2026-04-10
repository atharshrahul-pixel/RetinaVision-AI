import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from gradcam import GradCAM, overlay_heatmap

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="RetinaVision AI", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

multi_labels = [
    'Normal',
    'Diabetic Retinopathy',
    'Glaucoma',
    'Cataract',
    'AMD',
    'Hypertension',
    'Myopia',
    'Other'
]

dr_severity_labels = ['Normal', 'Mild', 'Moderate', 'Severe']

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():

    # Multi-disease model
    multi_model = models.efficientnet_b3(weights=None)
    multi_model.classifier[1] = nn.Linear(
        multi_model.classifier[1].in_features, 8
    )
    multi_model.load_state_dict(
        torch.load("multi_disease_b3.pth", map_location=DEVICE)
    )
    multi_model.to(DEVICE)
    multi_model.eval()

    # DR severity model (4 classes)
    dr_model = models.efficientnet_b3(weights=None)
    dr_model.classifier[1] = nn.Linear(
        dr_model.classifier[1].in_features, 4
    )
    dr_model.load_state_dict(
        torch.load("best_retina_model.pth", map_location=DEVICE)
    )
    dr_model.to(DEVICE)
    dr_model.eval()

    return multi_model, dr_model

multi_model, dr_model = load_models()

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

# =========================
# UI
# =========================
st.title("🧠 RetinaVision AI")
st.subheader("AI-Based Ocular Disease Screening & DR Severity Grading")

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Fundus Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # ================= MULTI-DISEASE PREDICTION =================
    with torch.no_grad():
        multi_outputs = multi_model(image_tensor)
        multi_probs = torch.sigmoid(multi_outputs)[0]

    global_threshold = 0.4
    dr_display_threshold = 0.45
    dr_trigger_threshold = 0.2

    detected = []

    for i, label in enumerate(multi_labels):
        prob = multi_probs[i].item()

        if label == "Diabetic Retinopathy":
            if prob > dr_display_threshold:
                detected.append(label)
        else:
            if prob > global_threshold:
                detected.append(label)

    if not detected:
        detected = ["Normal"]

    # ================= DISPLAY RESULTS =================
    with col2:
        st.success("Screening Complete")

        st.markdown("### 🩺 Screening Result")
        for d in detected:
            st.write(f"- {d}")

        st.markdown("### 📊 Multi-Disease Probabilities")
        for i, label in enumerate(multi_labels):
            st.write(f"{label}: {multi_probs[i].item():.4f}")

    # ================= DR SEVERITY =================
    dr_index = multi_labels.index("Diabetic Retinopathy")
    dr_probability = multi_probs[dr_index].item()

    if dr_probability > dr_trigger_threshold:

        with torch.no_grad():
            dr_outputs = dr_model(image_tensor)
            dr_probs = torch.softmax(dr_outputs, dim=1)[0]
            dr_class = torch.argmax(dr_probs).item()

        st.markdown("---")
        st.markdown("## 🔬 DR Severity Analysis")

        st.info(f"Predicted Severity: {dr_severity_labels[dr_class]}")

        st.markdown("### 📊 DR Severity Probabilities")
        for i, label in enumerate(dr_severity_labels):
            st.write(f"{label}: {dr_probs[i].item():.4f}")

    # ================= GRAD-CAM =================
    st.markdown("---")
    st.markdown("## 🔥 Model Attention (Grad-CAM)")

    # Create fresh tensor WITH gradients enabled
    image_tensor_grad = transform(image).unsqueeze(0).to(DEVICE)
    image_tensor_grad.requires_grad = True

    # Re-run model WITHOUT no_grad
    multi_outputs_grad = multi_model(image_tensor_grad)
    multi_probs_grad = torch.sigmoid(multi_outputs_grad)[0]

    top_class = torch.argmax(multi_probs_grad).item()

    # Create GradCAM instance
    target_layer = multi_model.features[-1]
    grad_cam = GradCAM(multi_model, target_layer)

    heatmap = grad_cam.generate(image_tensor_grad, top_class)

    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    cam_image = overlay_heatmap(image_np, heatmap)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)

    st.image(cam_image, caption="Grad-CAM Visualization", use_container_width=True)

st.markdown("---")
st.caption("RetinaVision AI © 2026 | Academic Demonstration System")