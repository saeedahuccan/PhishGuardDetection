import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

MODEL_REPO = "SaeedaKhan/PhishGuard"

st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ PhishGuard")
st.subheader("AI-Powered Email Phishing Detection")

st.write(
    "Analyze emails using a fine-tuned DistilBERT model "
    "with additional suspicious-email checks."
)


@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPO
    )

    model.eval()

    return tokenizer, model


try:

    tokenizer, model = load_model()

    email_text = st.text_area(
        "Enter Email Content",
        height=220,
        placeholder="Paste your email here..."
    )

    if st.button("Analyze Email", type="primary"):

        if not email_text.strip():

            st.warning("Please enter email content.")

        else:

            inputs = tokenizer(
                email_text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )

            with torch.inference_mode():

                outputs = model(**inputs)

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=1
                )[0]

                prediction = torch.argmax(
                    probabilities
                ).item()

            confidence = float(probabilities[prediction]) * 100

            safe_probability = float(probabilities[0]) * 100

            phishing_probability = float(probabilities[1]) * 100

            # ADDITIONAL SUSPICIOUS EMAIL CHECKS

            email_lower = email_text.lower()

            financial_terms = [
                "banking details",
                "payment details",
                "bank account",
                "vendor payment",
                "invoice",
                "wire transfer",
                "payment information",
                "banking information"
            ]

            change_terms = [
                "changed",
                "updated",
                "new details",
                "change of",
                "may have changed",
                "updated banking"
            ]

            credential_terms = [
                "password",
                "login credentials",
                "account verification",
                "login details"
            ]

            urgency_terms = [
                "urgent",
                "immediately",
                "deadline",
                "today",
                "as soon as possible"
            ]

            financial_flag = any(
                word in email_lower
                for word in financial_terms
            )

            change_flag = any(
                word in email_lower
                for word in change_terms
            )

            credential_flag = any(
                word in email_lower
                for word in credential_terms
            )

            urgency_flag = any(
                word in email_lower
                for word in urgency_terms
            )

            requires_review = (
                (financial_flag and change_flag)
                or (credential_flag and urgency_flag)
            )

            # DISPLAY RESULTS

            st.divider()

            st.subheader("Analysis Result")

            st.write("### Model Prediction")

            if prediction == 1:

                st.error("🚨 Phishing Email Detected")

            else:

                st.success("✅ Safe Email Predicted")

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.write("### Class Probabilities")

            st.write(f"Safe Email: {safe_probability:.2f}%")
            st.progress(safe_probability / 100)

            st.write(
                f"Phishing Email: {phishing_probability:.2f}%"
            )

            st.progress(phishing_probability / 100)

            st.divider()

            st.subheader("Additional Security Assessment")

            if requires_review:

                st.warning(
                    "⚠️ Suspicious Activity Detected! "
                    "Manual verification is recommended."
                )

                if financial_flag and change_flag:

                    st.write(
                        "Reason: Financial information change request."
                    )

                if credential_flag and urgency_flag:

                    st.write(
                        "Reason: Urgent account or credential request."
                    )

            elif prediction == 1:

                st.error(
                    "High-risk email: Do not trust links or attachments "
                    "without verification."
                )

            else:

                st.success(
                    "No additional warning triggered. "
                    "Still verify the sender when necessary."
                )

            st.caption(
                "PhishGuard provides AI-based predictions and "
                "additional rule-based warnings. Neither guarantees "
                "that an email is safe or malicious."
            )

except Exception as e:

    st.error("Unable to load or run the model.")

    st.exception(e)


st.divider()

st.caption(
    "PhishGuard | Fine-tuned DistilBERT | "
    "Educational Email Security Project"
)
