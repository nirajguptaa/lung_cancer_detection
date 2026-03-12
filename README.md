

# AI-Powered Lung Cancer Detection System

An AI-powered web platform for detecting lung cancer from CT scan images using **Deep Learning (EfficientNetB0)**, **Explainable AI (Grad-CAM)**, and an **AI Medical Assistant**. The system also generates **PDF medical reports** and maintains **patient scan history** using a Supabase database.

---

## Project Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection significantly improves survival rates. This project demonstrates how Artificial Intelligence can assist healthcare professionals by analyzing CT scan images and providing explainable predictions.

The platform allows users to:

- Upload CT scan images
- Detect lung abnormalities using a deep learning model
- Visualize model attention using Grad-CAM
- Interact with an AI assistant to understand results
- Generate automated medical reports
- Track scan history through a patient dashboard

---

## Key Features

### AI-Based Lung Cancer Detection
A deep learning model analyzes CT scan images and predicts whether the scan belongs to one of three categories:

- Normal
- Benign
- Malignant

The system provides prediction confidence to help interpret model reliability.

---

### Explainable AI (Grad-CAM)
Grad-CAM visualization highlights the regions of the CT scan that influenced the model's prediction. This improves transparency and helps verify that the model focuses on medically relevant lung regions.

---

### AI Medical Assistant
The platform integrates a chatbot powered by the **Groq LLM API**. The assistant can:

- Explain prediction results
- Provide medical guidance
- Suggest next diagnostic steps

⚠️ The assistant provides informational guidance only and does not replace professional medical diagnosis.

---

### Automated PDF Medical Reports
Users can generate structured medical reports containing:

- Patient information
- Prediction results
- Confidence scores
- Risk assessment
- Grad-CAM visualizations

These reports can be downloaded and shared with healthcare professionals.

---

### Patient Dashboard & Scan History
Using **Supabase database**, the system stores patient data and scan results.

The dashboard allows patients to:

- View previous scans
- Track prediction history
- Download reports

---

## System Architecture

User

↓

Web Interface (HTML / CSS / JavaScript)

↓

Flask Backend

↓

• Image Preprocessing
• EfficientNetB0 Deep Learning Model
• Grad-CAM Explainability
• AI Chatbot (Groq API)
• PDF Report Generator

↓

Supabase Database

• Patients Table
• Scans Table

---

## Technologies Used

### Machine Learning
- TensorFlow
- Keras
- EfficientNetB0
- Grad-CAM

### Backend
- Python
- Flask

### Database
- Supabase

### AI Integration
- Groq API

### Frontend
- HTML
- CSS
- JavaScript
- Chart.js

### Report Generation
- ReportLab

---

## Project Structure

lung-cancer-detection

│

├── app

│   ├── app.py

│   ├── chatbot.py

│   ├── database.py

│   ├── gradcam.py

│   └── report_generator.py

│

├── templates

│   ├── index.html

│   └── dashboard.html

│

├── static

│   ├── style.css

│   ├── uploaded images

│   └── heatmaps

│

├── models

│   └── efficientnet_final.h5

│

├── requirements.txt

└── README.md

---

## Installation

Clone the repository

```
git clone https://github.com/yourusername/lung-cancer-detection.git
```

Navigate to project directory

```
cd lung-cancer-detection
```

Create virtual environment

```
python -m venv venv
```

Activate environment

Mac / Linux

```
source venv/bin/activate
```

Windows

```
venv\\Scripts\\activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the root directory and add the following:

```
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_api_key
GROQ_API_KEY=your_groq_api_key
FLASK_SECRET_KEY=your_secret_key
```

---

## Running the Application

Start the Flask server:

```
python app.py
```

Open the application in your browser:

```
http://127.0.0.1:5001
```

---

## Model Details

The system uses **EfficientNetB0**, a convolutional neural network architecture optimized for high performance with fewer parameters.

Key techniques used:

- Transfer Learning
- Fine-tuning
- Image preprocessing
- Grad-CAM explainability

---

## Model Evaluation

Typical evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Grad-CAM visualization was used to validate model interpretability.

---

## Limitations

This system is intended for **research and educational purposes only**.

Limitations include:

- Limited training dataset
- Not trained on full clinical datasets
- Should not replace medical professionals

---

## Future Improvements

Possible improvements include:

- Training on larger medical datasets
- Integration with hospital PACS systems
- Long-term patient risk monitoring
- Cloud deployment
- Multi-disease detection

---

## Medical Disclaimer

This project is developed for **academic and research purposes only** and should not be used for real medical diagnosis.

Always consult qualified healthcare professionals for medical decisions.

---

