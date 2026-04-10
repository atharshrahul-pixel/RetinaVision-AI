# RetinaVision AI – Deep Learning-Based Ocular Disease Detection System

RetinaVision AI is an AI-powered retinal screening system that analyzes fundus images to detect multiple eye diseases and assess Diabetic Retinopathy (DR) severity. It combines Deep Learning, Computer Vision, and explainable AI to provide accurate predictions with visual insights through an interactive web interface.

⚠️ **Disclaimer**: RetinaVision AI is developed for educational and research purposes only and does not provide medical diagnosis.

---

# Features

1. Multi-Disease Detection – Identifies 8 retinal conditions (Normal, DR, Glaucoma, Cataract, AMD, etc.)
2. DR Severity Classification – Classifies Diabetic Retinopathy into Normal, Mild, Moderate, Severe
3. Grad-CAM Visualization – Highlights affected retinal regions for interpretability
4. Real-Time Prediction – Instant results through image upload
5. Dual-Model System – Combines disease detection + severity classification
6. Streamlit UI – Clean, interactive, and user-friendly interface

---

# Tech Stack

1. Artificial Intelligence (AI) – Decision-making and classification
2. Deep Learning (EfficientNet-B3) – Feature extraction and prediction
3. Computer Vision (OpenCV, PIL) – Image preprocessing
4. Explainable AI (Grad-CAM) – Visual interpretation of predictions
5. PyTorch – Model development and training
6. Streamlit – Web application interface
7. Python – Core programming language

---

# How It Works
1. User uploads a retinal fundus image
2. Image is preprocessed (resized & normalized)
3. Multi-disease model predicts possible eye conditions
4. If DR is detected → severity model is triggered
5. Grad-CAM highlights important regions
6. Final results with probabilities are displayed

---

# Installation & Setup

1. Clone the repository
git clone https://github.com/your-username/retinavision-ai.git

2. Navigate to project folder
cd RetinaVisionAI_Test

3. Install dependencies
pip install -r requirements.txt

---

# Run the application
streamlit run app.py

---

# Requirements
Python 3.10+
PyTorch
Torchvision
Streamlit
OpenCV
NumPy
Pillow

---

# Usage
1. Upload a retinal image (.jpg/.png)
2. View detected diseases and probabilities
3. Check DR severity (if detected)
4. Visualize affected regions using Grad-CAM
5. Upload another image for re-analysis

---

# Use Cases
1. AI-based medical screening demos
2. Ophthalmology research support
3. Healthcare awareness tools
4. Academic and hackathon projects
5. Telemedicine prototypes

---

# Future Improvements
1. Multi-modal diagnosis (image + patient data)
2. More disease categories (e.g., retinal detachment)
3. Cloud deployment for remote access
4. Mobile app integration
5. Clinical validation with real-world datasets

---

# Conclusion

RetinaVision AI demonstrates the power of deep learning in medical imaging by enabling early detection of retinal diseases. With EfficientNet-based models and Grad-CAM visualization, the system provides both accurate predictions and meaningful insights, making it a strong tool for AI-driven healthcare innovation.
