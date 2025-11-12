
import streamlit as st
import io
import joblib
import re
from sentence_transformers import SentenceTransformer

# ============================================================
# Load trained ML models and SBERT embedder (cached for performance)
# ============================================================
@st.cache_resource
def load_prediction_resources():
    models = {
        "O": joblib.load("RandomForest_O.pkl"),
        "C": joblib.load("RandomForest_C.pkl"),
        "E": joblib.load("LinearRegression_E.pkl"),
        "A": joblib.load("LinearRegression_A.pkl"),
        "N": joblib.load("RandomForest_N.pkl")
    }
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return models, embedder

models, embedder = load_prediction_resources()

# ------------------------------------------------------------
# Text cleanup (copied from previous steps)
# ------------------------------------------------------------
def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)       # remove special chars / numbers
    text = re.sub(r"\s+", " ", text).strip()      # remove extra spaces
    return text

# ------------------------------------------------------------
# Personality Prediction Function (copied from previous steps)
# ------------------------------------------------------------
def predict_personality(text: str):
    cleaned = clean_text(text)

    # Convert text into SBERT embeddings
    embeddings = embedder.encode([cleaned])

    result = {}

    # Predict using trained models
    for trait, model in models.items():
        raw_pred = float(model.predict(embeddings)[0])

        # Force trait output to range [0,1]
        result[trait] = max(0, min(1, raw_pred))

    return result

# ------------------------------------------------------------
# PDF Text Extraction Function (copied from previous steps)
# ------------------------------------------------------------
def extract_text_from_pdf(pdf_file_object):
    """Extracts text from a PDF file object."""
    import pdfplumber # Import here to avoid dependency if not used
    all_text = []
    try:
        with pdfplumber.open(pdf_file_object) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""
    return " ".join(all_text).strip()


# ============================================================
# Streamlit App Layout
# ============================================================
st.title("Personality Trait Predictor")
st.write("Upload a PDF or enter text to predict OCEAN personality traits.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Text input area, dynamically populated or left empty
text_input = ""
if uploaded_file is not None:
    # Read the file as bytes and pass to the extraction function
    bytes_data = uploaded_file.getvalue()
    # Using io.BytesIO to simulate a file object for pdfplumber
    text_from_pdf = extract_text_from_pdf(io.BytesIO(bytes_data))
    if text_from_pdf:
        st.success("Text extracted from PDF successfully!")
        text_input = st.text_area("Extracted text from PDF (edit if needed):", text_from_pdf, height=300)
    else:
        st.error("Could not extract text from PDF. Please try another file or enter text manually.")
        text_input = st.text_area("Enter text here:", "", height=200)
else:
    text_input = st.text_area("Enter text here:", "", height=200)

# Prediction button
if st.button("Predict Personality Traits"):
    if text_input:
        predictions = predict_personality(text_input)
        st.subheader("Predicted OCEAN Traits:")

        # Display results with progress bars for better visualization
        for trait, score in predictions.items():
            st.write(f"**{trait}:** {score:.2f}")
            st.progress(score)
    else:
        st.warning("Please upload a PDF or enter text to make a prediction.")
