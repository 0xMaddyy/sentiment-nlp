"""
app.py
Streamlit web app for real-time sentiment prediction.
Run: streamlit run app.py
"""
import streamlit as st
import joblib
from preprocess import clean_text

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_artifacts()

st.title("Sentiment Analysis on Reviews")
st.write("Enter a product review or social media post. The model predicts positive or negative sentiment.")

text = st.text_area("Your text:", "This product exceeded my expectations!")

if st.button("Analyze"):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "Positive 😀" if pred == 1 else "Negative 😞"
    confidence = prob[pred]

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2%}")
    st.progress(float(confidence))
