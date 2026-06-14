import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuration & Page Setup
st.set_page_config(
    page_title="Smart House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Intern Metadata
INTERN_ID = "CITS3911"
DEVELOPER_GITHUB = "https://github.com/zakkyaj"
MODEL_PATH = os.path.join("models", "model.pkl")

# Custom CSS for Modern, Premium UI
st.markdown("""
    <style>
    /* Main Layout Styling */
    .reportview-container {
        background: #f8f9fa;
    }
    
    /* Header & Branding */
    .title-text {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .subtitle-text {
        font-size: 1.0rem;
        color: #64748b;
        margin-top: 0px;
        margin-bottom: 25px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 12px;
    }
    
    /* Predict Output Card */
    .prediction-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-top: 20px;
        margin-bottom: 25px;
    }
    .prediction-title {
        font-size: 1.1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.9;
        margin-bottom: 8px;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Metric Section */
    .metric-box {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    
    /* Sidebar Details */
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0f172a;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .sidebar-info {
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Design
with st.sidebar:
    st.markdown("### 🏠 Project Dashboard")
    st.markdown(f"**Internship Submission**")
    
    st.markdown("<div class='sidebar-header'>Developer Info</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sidebar-info'>🧑‍💻 <b>Intern ID:</b> {INTERN_ID}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sidebar-info'>🌐 <b>GitHub:</b> <a href='{DEVELOPER_GITHUB}' target='_blank'>{DEVELOPER_GITHUB.split('/')[-1]}</a></div>", unsafe_allow_html=True)
    
    st.markdown("---")

    # Load Model (if exists) to display metrics in sidebar
    model_loaded = False
    payload = None
    if os.path.exists(MODEL_PATH):
        try:
            payload = joblib.load(MODEL_PATH)
            model_loaded = True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            
    if model_loaded and payload:
        st.markdown("<div class='sidebar-header'>Model Performance</div>", unsafe_allow_html=True)
        metrics = payload['metrics']
        st.markdown(f"<div class='sidebar-info'>📈 <b>R² Accuracy:</b> {metrics['r2'] * 100:.2f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sidebar-info'>📉 <b>MAE:</b> ${metrics['mae']:,.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sidebar-info'>📊 <b>RMSE:</b> ${metrics['rmse']:,.2f}</div>", unsafe_allow_html=True)
        st.caption("Random Forest Regressor")
    else:
        st.markdown("<div class='sidebar-header'>Model Status</div>", unsafe_allow_html=True)
        st.warning("⚠️ Model not trained yet. Please run 'train_model.py' first.")

# Main Page Header
st.markdown(f"<div class='title-text'>🏠 Smart House Price Predictor</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle-text'>Internship Project | Intern ID: {INTERN_ID}</div>", unsafe_allow_html=True)

# Main workflow
if not model_loaded:
    st.info("### Welcome to the Smart House Price Predictor")
    st.markdown(f"""
        This machine learning web application is built for the internship project under Intern ID **{INTERN_ID}**.
        
        To run predictions, we need to train the machine learning model first.
        
        **Instructions to start:**
        1. Run the model training script locally to generate the dataset and train the model:
           ```bash
           python train_model.py
           ```
        2. Once the script finishes, refresh this page to begin predicting house prices.
    """)
    st.stop()

# Model is loaded, let's render the prediction form
model = payload['model']

st.markdown("### Enter Property Characteristics")
st.write("Adjust the fields below to obtain a real-time price estimation from the trained Random Forest model.")

# Grid of input controls
col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (Square Feet)",
        min_value=500,
        max_value=10000,
        value=1800,
        step=50,
        help="Total usable area of the house in square feet."
    )
    
    bedrooms = st.slider(
        "Bedrooms",
        min_value=1,
        max_value=6,
        value=3,
        step=1,
        help="Number of bedrooms in the house."
    )

with col2:
    bathrooms = st.slider(
        "Bathrooms",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        help="Number of bathrooms in the house."
    )
    
    parking = st.selectbox(
        "Parking Spaces",
        options=[0, 1, 2, 3],
        index=1,
        help="Number of reserved parking spaces."
    )

# Prediction Calculation
try:
    # Prepare input DataFrame (ensuring feature names match training features)
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, parking]], columns=payload['features'])
    
    # Calculate Prediction
    prediction = model.predict(input_data)[0]
    
    # Format and Output Result
    st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-title">Estimated Property Value</div>
            <div class="prediction-value">${prediction:,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred while calculating the prediction: {e}")
    st.info("Please verify the model training and structure.")

# Footer Section
st.markdown("---")
col_footer1, col_footer2 = st.columns(2)
with col_footer1:
    st.caption(f"Internship Project | Intern ID: {INTERN_ID}")
with col_footer2:
    st.markdown(f"<div style='text-align: right;'><a href='{DEVELOPER_GITHUB}' target='_blank' style='text-decoration: none; color: gray; font-size: 0.85rem;'>Developer GitHub Profile</a></div>", unsafe_allow_html=True)
