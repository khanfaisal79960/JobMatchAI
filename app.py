# streamlit_app.py
import streamlit as st
import os
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Streamlit UI Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Job Recommendation Engine",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- NLTK Downloads ---
# Ensure these are downloaded when the Streamlit app runs.
# Streamlit caches these downloads, so they only happen once.
@st.cache_resource
def download_nltk_data():
    # Directly call nltk.download() without a try-except block for DownloadError
    # Streamlit's cache_resource will ensure this runs only once.
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_data()

# --- Configuration ---
# Define the path where models are stored
# In a deployed Streamlit app, ensure 'model' directory is accessible.
MODEL_DIR = 'model'
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
JOB_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'job_classifier.joblib')
JOB_CATEGORIES_PATH = os.path.join(MODEL_DIR, 'job_categories.joblib')

# --- Load Trained Models ---
# Use Streamlit's caching to load models only once
@st.cache_resource
def load_models():
    try:
        tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        job_classifier = joblib.load(JOB_CLASSIFIER_PATH)
        job_categories = joblib.load(JOB_CATEGORIES_PATH)
        # st.success("Models loaded successfully!") # Removed to avoid calling Streamlit command before set_page_config
        return tfidf_vectorizer, job_classifier, job_categories
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.error("Please ensure 'tfidf_vectorizer.joblib', 'job_classifier.joblib', and 'job_categories.joblib' are in the 'model/' directory.")
        st.stop() # Stop the app if models are not found
    except Exception as e:
        st.error(f"An unexpected error occurred while loading models: {e}")
        st.stop()

tfidf_vectorizer, job_classifier, job_categories = load_models()

# Display success message after models are loaded and page config is set
st.success("Models loaded successfully!")


# --- Text Preprocessing Function (must match the one used during training) ---
def preprocess_text(text):
    """
    Cleans and preprocesses a given text string.
    Steps include: lowercase conversion, URL removal, HTML tag removal,
    punctuation removal, newline removal, numeric word removal,
    stopwords removal, and lemmatization.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# --- Streamlit UI ---
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: #fff;
        font-family: 'Inter', sans-serif;
    }
    .stTextInput>div>div>textarea {
        border-radius: 10px;
        border: 1px solid #ced4da;
        padding: 15px;
        font-size: 1.1rem;
        min-height: 250px;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(45deg, #2575fc, #6a11cb);
        color: white;
        border: none;
        padding: 15px 40px;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 50px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #6a11cb, #2575fc);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }
    .stAlert {
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #fff; /* White for headings */
    }
    .stMarkdown p {
        color: #eee; /* Lighter white for paragraphs */
    }
    .recommended-category {
        font-size: 2.5rem;
        font-weight: bold;
        background: -webkit-linear-gradient(45deg, #28a745, #007bff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💼 Job Recommendation Engine")
st.markdown("Paste your resume below and let our AI recommend the best job category for you!")

resume_text = st.text_area(
    "Paste your resume content here:",
    height=300,
    placeholder="e.g., 'Experienced software engineer with strong skills in Python, machine learning, and cloud computing...'"
)

if st.button("Get Recommendation"):
    if resume_text:
        with st.spinner("Analyzing your resume..."):
            try:
                cleaned_resume = preprocess_text(resume_text)
                resume_vector = tfidf_vectorizer.transform([cleaned_resume])
                predicted_category = job_classifier.predict(resume_vector)[0]
                st.markdown(f"<p class='recommended-category'>Recommended Job Category: {predicted_category}</p>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during recommendation: {e}")
    else:
        st.warning("Please paste your resume content to get a recommendation.")

st.markdown("---")
st.markdown("Created by Faisal Khan")
st.markdown(
    """
    <div style="text-align: center; margin-top: 10px;">
        <a href="https://khanfaisal.netlify.app" target="_blank" style="color: white; margin: 0 10px;"><i class="fas fa-globe"></i></a>
        <a href="https://medium.com/@khanfaisal79960" target="_blank" style="color: white; margin: 0 10px;"><i class="fab fa-medium"></i></a>
        <a href="https://www.linkedin.com/in/khanfaisal79960" target="_blank" style="color: white; margin: 0 10px;"><i class="fab fa-linkedin"></i></a>
        <a href="https://github.com/khanfaisal79960" target="_blank" style="color: white; margin: 0 10px;"><i class="fab fa-github"></i></a>
        <a href="https://instagram.com/mr._perfect_1004" target="_blank" style="color: white; margin: 0 10px;"><i class="fab fa-instagram"></i></a>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """, unsafe_allow_html=True)
