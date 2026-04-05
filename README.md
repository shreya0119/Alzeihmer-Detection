# 🧠Alzheimer Detection

A Flask-based web application that uses deep learning to detect early signs of Alzheimer's disease from MRI brain scans. The model achieves ~80% validation accuracy and provides clinical-grade sensitivity for moderate dementia detection.

# 📌Features

- **Web Interface**: User-friendly Flask application for uploading MRI scans
- **Deep Learning Model**: MobileNetV2-based classifier trained on OASIS-protocol MRI data
- **Multi-class Classification**: Detects 4 dementia severity levels:
  - Non Demented
  - Very Mild Dementia
  - Mild Dementia
  - Moderate Dementia
- **Clinical Sensitivity**: Enhanced detection thresholds for early intervention
- **Real-time Predictions**: Instant confidence scores for uploaded scans

# 📝Prerequisites
- Python 3.8+
- pip (Python package manager)

# 🖥️Setup Instructions

1. **Clone the repo**
```bash
git clone https://github.com/shreya0119/Alzheimer-Detection.git
cd Alzheimer-Detection
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the pre-trained weights file** from https://drive.google.com/file/d/1b3vrvXOXgelaJb9SRXQnuGrAQEpWJrgv/view?usp=drive_link and place it in the 
`models/` folder as `alzheimer_pro_weights.weights.h5`

5. **Run the app**
```bash
python app.py
```
- App opens automatically at http://127.0.0.1:5000
- Upload an MRI brain scan to get a diagnosis.

# 🗂️Project Structure
```
Alzheimer-Detection/
├── app.py                           # Flask application entry point
├── model_utils.py                   # Model loading & prediction logic
├── requirements.txt                 # Python dependencies
├── models/
│   └── alzheimer_pro_weights.weights.h5  # Pre-trained model weights
├── templates/
│   └── index.html                   # Web interface
├── static/
│   └── uploads/                     # Uploaded MRI images
└── notebook/                        # Training & analysis notebooks
```

# 📍Model Architecture

- **Base Model**: MobileNetV2 (pretrained on ImageNet, frozen layers)
- **Input Size**: 224×224×3 (RGB)
- **Architecture**:
    - MobileNetV2 backbone with GlobalAveragePooling2D
    - Dropout layer (0.4) for regularization
    - Dense output layer (4 classes) with softmax activation
- **Training**: 10 epochs (2-phase fine-tuning approach)
- **Training accuracy**: 77.60%
- **Validation Accuracy**: 80/89% (Epoch 7)
- **Overfitting Gap**:2.45%

# 📊 Training performance


| Metric | Initial | Final | Best | Status |
|--------|---------|-------|------|--------|
| **Training Loss** | 0.7962 | 0.8982 | 0.6990 @ Ep5 | Converged |
| **Validation Loss** | 0.5032 | 0.4920 | 0.4920 @ Ep10 |  Improved |
| **Training Accuracy** | 76.32% | 77.60% | 81.42% @ Ep7 | ↑ 1.28% |
| **Validation Accuracy** | 79.42% | 80.05% | 80.89% @ Ep7 | ↑ 0.63% |


# 📍Model Behavior & Clinical Sensitivity

The model implements **sensitivity thresholds** for early detection:

| Condition/THreshold                | Action                              |
| ---------------------------------- | ----------------------------------- |
| Moderate Dementia confidence > 10% | Report as Moderate (priority alert) |
| Mild Dementia confidence > 10%     | Report as Mild                      |
| Otherwise                          | Report highest confidence class     |

This ensures early-stage dementia cases are not missed due to model uncertainty.

# 📍Input Requirements
- **Image Format** : JPEG,PNG
- **Optimal Source**: OASIS-protocol T1 axial MRI scans
- **Image Preprocessing**: Automatically resized to 224 x 224 and normalized to [0,1]

# ⚠️Key Limitations
- **Training data size**: Optimized for OASIS dataset; performance may vary with different dataset sizes
- **Epoch convergence**: Model converged at epoch 7; training continued to epoch 10 showed minor fluctuations
- **Validation protocol**: Model validated on held-out OASIS-protocol MRI scans
- **Cross-scanner generalization**: Performance may vary significantly with:
    - Different MRI scanner models
    - Alternative imaging protocols
    - Different acquisition parameters
- **Clinical use**: This is a research tool. Results should be validated by medical professionals
- **Data imbalance**: Class weights are applied during training due to dataset imbalance.


# ⚠️Disclaimer

This tool is for educational and research purposes only. It should not be used for clinical diagnosis without professional medical review and validation.