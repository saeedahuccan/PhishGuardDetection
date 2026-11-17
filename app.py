import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Application configuration
# -----------------------------
st.set_page_config(
    page_title="PhishGuard | Email Security",
    page_icon="🛡️",
    layout="centered"
)

MODEL_ID = "SaeedaKhan/PhishGuard"
MAX_LENGTH = 512

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource(show_spinner="Loading PhishGuard AI model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


st.title("🛡️ PhishGuard")
st.subheader("AI-Powered Email Phishing Detection")
st.write(
    "Analyze an email message using your fine-tuned "
    "DistilBERT classification model."
)

st.info(
    "Paste the email content below. PhishGuard will "
    "classify it as Safe Email or Phishing Email."
)

# -----------------------------
# Email input
# -----------------------------
email_text = st.text_area(
    "Enter email content",
    height=220,
    placeholder="Paste your email message here..."
)

analyze = st.button(
    "🔍 Analyze Email",
    type="primary",
    use_container_width=True
)

# -----------------------------
# Prediction
# -----------------------------
if analyze:
    if not email_text.strip():
        st.warning("Please enter an email message first.")

    else:
        try:
            tokenizer, model = load_model()

            with st.spinner("Analyzing email..."):
                inputs = tokenizer(
                    email_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=True
                )

                with torch.inference_mode():
                    outputs = model(**inputs)
                    probabilities = F.softmax(
                        outputs.logits,
                        dim=-1
                    )[0]

                predicted_class = int(
                    torch.argmax(probabilities).item()
                )

                confidence = float(
                    probabilities[predicted_class].item()
                ) * 100

                safe_probability = float(probabilities[0]) * 100
                phishing_probability = float(probabilities[1]) * 100

            st.divider()
            st.subheader("Analysis Result")

            if predicted_class == 1:
                st.error("🚨 Phishing Email Detected")
            else:
                st.success("✅ Safe Email Predicted")

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.write("### Class probabilities")

            st.write(f"Safe Email: {safe_probability:.2f}%")
            st.progress(safe_probability / 100)

            st.write(
                f"Phishing Email: {phishing_probability:.2f}%"
            )
            st.progress(phishing_probability / 100)

            st.caption(
                "Prediction is based on the model's learned "
                "patterns and is not a guarantee of email safety."
            )

        except Exception as error:
            st.error("Unable to load or run the model.")
            st.exception(error)

st.divider()
st.caption(
    "PhishGuard | Fine-tuned DistilBERT | "
    "Educational Email Security Project"
)
