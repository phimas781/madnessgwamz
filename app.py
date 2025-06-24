import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Gwamz Track Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model (make sure 'gwamz_predictor.pkl' is in the same folder)
import os
import streamlit as st
import joblib

@st.cache_resource
def load_model():
    # Try multiple possible paths
    possible_paths = [
        'models/gwamz_predictor.pkl',  # Relative path
        os.path.join(os.path.dirname(__file__), 'models/gwamz_predictor.pkl'),  # Absolute path
        'gwamz_predictor.pkl'  # Root directory fallback
    ]
    
    for path in possible_paths:
        try:
            st.write(f"Trying path: {path}")
            return joblib.load(path)
        except FileNotFoundError:
            continue
    
    # If all paths fail, show error and stop
    st.error(f"Model file not found in any of these locations: {possible_paths}")
    st.stop()

model = load_model()

# App title
st.title("🎤 Gwamz Song Performance Predictor")
st.markdown("""
Predict how many streams a new Gwamz track will get based on historical data.
Adjust the settings in the sidebar and click **Predict**.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Track Configuration")

def user_input_features():
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

# --- Main Panel ---
st.subheader("Selected Parameters")
st.write(input_df)

# Prediction
if st.button("Predict Streams"):
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

# --- Historical Data Visualization ---
st.markdown("---")
st.subheader("Gwamz's Historical Performance")

@st.cache_data
def load_historical_data():
    df = pd.read_csv('gwamz_data.csv')
    df['release_date'] = pd.to_datetime(df['release_date'], format='%d/%m/%Y')
    df['version'] = 'Original'
    df.loc[df['track_name'].str.contains('Sped Up'), 'version'] = 'Sped Up'
    df.loc[df['track_name'].str.contains('Remix'), 'version'] = 'Remix'
    df.loc[df['track_name'].str.contains('Instrumental'), 'version'] = 'Instrumental'
    df.loc[df['track_name'].str.contains('Jersey'), 'version'] = 'Jersey Club'
    return df

hist_data = load_historical_data()

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
