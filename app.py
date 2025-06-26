import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import urllib.request
import os

# =============================================
# MODEL LOADING (FULLY WORKING VERSION)
# =============================================

def load_model():
    """Robust model loader with multiple fallback mechanisms"""
    # Configuration - REPLACE WITH YOUR ACTUAL GITHUB URL
    GITHUB_RAW_URL = "https://github.com/phimas781/madnessgwamz/blob/main/gwamz_streams_predictor.pkl"
    MODEL_NAME = "gwamz_predictor.pkl"
    MODEL_DIR = "models"
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Possible model paths to check
    possible_paths = [
        Path(MODEL_DIR) / MODEL_NAME,  # models/gwamz_predictor.pkl
        Path(MODEL_NAME),              # gwamz_predictor.pkl
        Path(__file__).parent / MODEL_DIR / MODEL_NAME,  # Absolute path
    ]
    
    # 1. Try local paths first
    for path in possible_paths:
        if path.exists():
            try:
                model = joblib.load(path)
                st.success(f"Successfully loaded model from: {path}")
                return model
            except Exception as e:
                st.warning(f"Found but couldn't load {path}: {str(e)}")
                continue
    
    # 2. Try downloading from GitHub (without progress bar)
    try:
        download_path = Path(MODEL_DIR) / MODEL_NAME
        st.warning("Attempting to download model from GitHub...")
        
        # Simple download without progress reporting
        urllib.request.urlretrieve(GITHUB_RAW_URL, download_path)
            
        if download_path.exists():
            try:
                model = joblib.load(download_path)
                st.success("Download successful!")
                return model
            except Exception as e:
                st.error(f"Downloaded file corrupted: {str(e)}")
                os.remove(download_path)  # Clean up corrupted file
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
    
    # 3. Ultimate fallback - manual upload
    st.error("""
    ### Automatic loading failed. Please:
    1. Download the model from: [GitHub](%s)
    2. Upload it below:
    """ % GITHUB_RAW_URL)
    
    uploaded_file = st.file_uploader(f"Upload {MODEL_NAME}", type="pkl")
    if uploaded_file is not None:
        try:
            with open(Path(MODEL_DIR) / MODEL_NAME, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Upload successful! Reloading...")
            st.rerun()
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")
    
    st.stop()

# Initialize model
try:
    model = load_model()
except Exception as e:
    st.error(f"Critical error loading model: {str(e)}")
    st.stop()

# =============================================
# STREAMLIT APP UI
# =============================================

# Page configuration
st.set_page_config(
    page_title="Gwamz Track Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("🎤 Gwamz Song Performance Predictor")
st.markdown("""
Predict how many streams a new Gwamz track will get based on historical data.
Adjust the settings in the sidebar and click **Predict**.
""")

# =============================================
# USER INPUT SECTION
# =============================================

st.sidebar.header("Track Configuration")

def user_input_features():
    """Collect user inputs from sidebar"""
    # Release Date
    release_date = st.sidebar.date_input("Release Date", datetime.now())
    
    # Album Info
    album_type = st.sidebar.selectbox("Album Type", ["single", "album", "compilation"])
    total_tracks = st.sidebar.slider("Total Tracks in Album", 1, 20, 1)
    markets = st.sidebar.slider("Available Markets", 1, 200, 185)
    
    # Track Metadata
    track_num = st.sidebar.slider("Track Number", 1, 20, 1)
    disc_num = st.sidebar.slider("Disc Number", 1, 5, 1)
    explicit = st.sidebar.checkbox("Explicit Content", True)
    
    # Track Version
    version = st.sidebar.radio("Track Version", 
                             ["Original", "Sped Up", "Remix", "Instrumental", "Jersey Club"])
    
    # Popularity Metrics
    followers = st.sidebar.number_input("Artist Followers", min_value=0, value=7937)
    artist_pop = st.sidebar.slider("Artist Popularity (0-100)", 0, 100, 41)
    track_pop = st.sidebar.slider("Expected Track Popularity (0-100)", 0, 100, 50)
    
    # Feature Engineering
    release_year = release_date.year
    release_month = release_date.month
    release_day = release_date.day
    release_weekday = release_date.weekday()  # Monday=0, Sunday=6
    
    # Days since first Gwamz release (April 29, 2021)
    first_release = datetime(2021, 4, 29).date()
    days_since_first = (release_date - first_release).days
    
    # Encode categoricals
    album_type_encoded = 0 if album_type == "single" else (1 if album_type == "album" else 2)
    explicit_encoded = 1 if explicit else 0
    
    # Track version flags
    is_sped_up = 1 if version == "Sped Up" else 0
    is_remix = 1 if version == "Remix" else 0
    is_instrumental = 1 if version == "Instrumental" else 0
    is_jersey = 1 if version == "Jersey Club" else 0
    
    # Create feature dict
    features = {
        'artist_followers': followers,
        'artist_popularity': artist_pop,
        'album_type_encoded': album_type_encoded,
        'release_year': release_year,
        'total_tracks_in_album': total_tracks,
        'available_markets_count': markets,
        'track_number': track_num,
        'disc_number': disc_num,
        'explicit_encoded': explicit_encoded,
        'track_popularity': track_pop,
        'release_month': release_month,
        'release_day': release_day,
        'release_day_of_week': release_weekday,
        'is_sped_up': is_sped_up,
        'is_remix': is_remix,
        'is_instrumental': is_instrumental,
        'is_jersey': is_jersey,
        'days_since_first_release': days_since_first,
        'track_age_days': 0  # New release
    }
    
    return pd.DataFrame([features])

# Get user input
input_df = user_input_features()

# =============================================
# PREDICTION SECTION
# =============================================

st.subheader("Selected Parameters")
st.write(input_df)

if st.button("Predict Streams"):
    try:
        prediction = model.predict(input_df)
        predicted_streams = int(prediction[0])
        
        st.success(f"### Predicted Streams: **{predicted_streams:,}**")
        
        # Performance analysis
        avg_streams = 500_000  # Gwamz's historical average
        performance_ratio = (predicted_streams / avg_streams) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Streams", f"{predicted_streams:,}")
        with col2:
            st.metric("Vs. Average", f"{performance_ratio:.1f}%", 
                     delta=f"{performance_ratio-100:.1f}%")
        
        # Interpretation
        if predicted_streams > avg_streams * 1.5:
            st.success("🔥 **Hit Potential!** This track is predicted to perform **significantly better** than average.")
        elif predicted_streams > avg_streams:
            st.success("👍 **Above Average** - Strong performance expected.")
        else:
            st.warning("⚠️ **Below Average** - Consider optimizing release strategy.")
        
        # Top influencing factors
        st.subheader("Key Factors Affecting Prediction")
        feature_importance = model.feature_importances_
        top_3 = sorted(zip(input_df.columns, feature_importance), 
                      key=lambda x: x[1], reverse=True)[:3]
        
        for feature, importance in top_3:
            value = input_df[feature].values[0]
            st.write(f"- **{feature.replace('_', ' ').title()}**: {value} (Impact: {importance*100:.1f}%)")
    
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# =============================================
# HISTORICAL DATA VISUALIZATION
# =============================================

st.markdown("---")
st.subheader("Gwamz's Historical Performance")

@st.cache_data
def load_historical_data():
    """Load and preprocess historical data"""
    try:
        df = pd.read_csv('gwamz_data.csv')
        df['release_date'] = pd.to_datetime(df['release_date'], format='%d/%m/%Y')
        df['version'] = 'Original'
        df.loc[df['track_name'].str.contains('Sped Up'), 'version'] = 'Sped Up'
        df.loc[df['track_name'].str.contains('Remix'), 'version'] = 'Remix'
        df.loc[df['track_name'].str.contains('Instrumental'), 'version'] = 'Instrumental'
        df.loc[df['track_name'].str.contains('Jersey'), 'version'] = 'Jersey Club'
        return df
    except Exception as e:
        st.error(f"Couldn't load historical data: {str(e)}")
        return pd.DataFrame()

hist_data = load_historical_data()

if not hist_data.empty:
    # Streams over time
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        data=hist_data, 
        x='release_date', 
        y='streams', 
        hue='version',
        marker='o',
        ax=ax1
    )
    ax1.set_title("Streams Over Time by Version")
    ax1.set_xlabel("Release Date")
    ax1.set_ylabel("Streams")
    st.pyplot(fig1)

    # Version comparison
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.boxplot(
        data=hist_data,
        x='version',
        y='streams',
        order=["Original", "Sped Up", "Remix", "Jersey Club", "Instrumental"],
        ax=ax2
    )
    ax2.set_title("Streams Distribution by Track Version")
    st.pyplot(fig2)
