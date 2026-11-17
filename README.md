# PhishGuardDetection
AI-Based Email Phishing Detection
# 🛡️ PhishGuard

### AI-Powered Email Phishing Detection System

PhishGuard is an email classification application powered by a fine-tuned DistilBERT Transformer model.

## Model Information

- Model: SaeedaKhan/PhishGuard
- Architecture: DistilBERT
- Task: Binary Text Classification
- Class 0: Safe Email
- Class 1: Phishing Email

## Features

- Email text analysis
- Safe and phishing classification
- Prediction confidence
- Class probability visualization
- Transformer-based inference

## Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- Streamlit

## Model Repository

https://huggingface.co/SaeedaKhan/PhishGuard

## Run Locally

Install dependencies:

    pip install -r requirements.txt

Start the application:

    streamlit run app.py

## Deployment

This application can be deployed using Streamlit Community Cloud through its GitHub integration.

## Important Note

PhishGuard is an educational machine learning project. Its predictions are not a guarantee that an email is safe or malicious. Always verify suspicious links and sender identities independently.
