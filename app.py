import streamlit as st
import numpy as np
import pickle

# Set page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# Load models and scalers
@st.cache_resource
def load_models():
    model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
    return model, sc, mx

model, sc, mx = load_models()

# Crop dictionary
CROP_DICT = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .prediction-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .prediction-text {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2E7D32;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
        width: 100%;
        margin-top: 10px;
        height: 3rem;
    }
    .section-header {
        font-size: 1.2rem;
        color: #2E7D32;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .info-text {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🌱 Crop Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Find the perfect crop for your soil and climate conditions</p>', unsafe_allow_html=True)

# Main form
with st.form(key="crop_recommendation_form"):
    # Soil nutrient section
    st.markdown('<p class="section-header">Soil Nutrients</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Enter the nutrient content of your soil:</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input('Nitrogen (N)', min_value=0.0, help="Nitrogen content in soil (mg/kg)")
    with col2:
        P = st.number_input('Phosphorus (P)', min_value=0.0, help="Phosphorus content in soil (mg/kg)")
    with col3:
        K = st.number_input('Potassium (K)', min_value=0.0, help="Potassium content in soil (mg/kg)")
    
    # Environmental conditions section
    st.markdown('<p class="section-header">Environmental Conditions</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Enter the climate and environmental conditions of your area:</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        temp = st.number_input('Temperature (°C)', min_value=0.0, help="Average temperature in Celsius")
    with col2:
        humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, help="Relative humidity percentage")
    with col3:
        ph = st.number_input('pH', min_value=0.0, max_value=14.0, help="Soil pH level (0-14)")
    with col4:
        rainfall = st.number_input('Rainfall (mm)', min_value=0.0, help="Annual rainfall in millimeters")
    
    # Predict button
    predict_button = st.form_submit_button('Find Best Crop')

# Prediction section
if predict_button:
    with st.spinner("Analyzing soil and climate data..."):
        # Create feature list and reshape for prediction
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)
        
        # Apply transformations
        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)
        prediction = model.predict(sc_mx_features)
        
        # Show result
        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        if prediction[0] in CROP_DICT:
            crop_name = CROP_DICT[prediction[0]]
            # Display image with prediction
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image("static/crop.png", use_column_width=True)
            
            st.markdown(f'<p class="prediction-text">✅ {crop_name}</p>', unsafe_allow_html=True)
            st.markdown(f'<p>is the best crop to cultivate based on your soil and climate conditions.</p>', unsafe_allow_html=True)
            
            # Display some basic information about the crop
            st.markdown("### About this crop:")
            crop_info = {
                "Rice": "Staple food crop that thrives in wet, humid conditions with consistent temperatures.",
                "Maize": "Versatile grain crop that prefers moderate temperatures and well-drained soil.",
                "Jute": "Fiber crop that grows well in high humidity and rainfall conditions.",
                "Cotton": "Important fiber crop that requires warm temperatures and moderate rainfall.",
                "Coconut": "Tropical crop that thrives in coastal areas with high humidity.",
                "Papaya": "Tropical fruit that prefers warm temperatures and well-drained soil.",
                "Orange": "Citrus fruit that grows best in subtropical climates.",
                "Apple": "Temperate fruit that requires cold winters and moderate summers.",
                "Muskmelon": "Summer fruit that needs warm temperatures and moderate watering.",
                "Watermelon": "Heat-loving fruit that requires well-drained soil and consistent moisture.",
                "Grapes": "Fruit crop that grows well in various climates with good drainage.",
                "Mango": "Tropical fruit that prefers warm, dry winters and hot summers.",
                "Banana": "Tropical fruit that thrives in humid conditions with consistent moisture.",
                "Pomegranate": "Fruit that prefers semi-arid regions with hot summers.",
                "Lentil": "Cool-season pulse crop that grows well in well-drained soil.",
                "Blackgram": "Pulse crop that prefers warm temperatures and moderate rainfall.",
                "Mungbean": "Short-season pulse crop that thrives in warm temperatures.",
                "Mothbeans": "Drought-resistant pulse crop suitable for semi-arid regions.",
                "Pigeonpeas": "Tropical legume that tolerates dry conditions once established.",
                "Kidneybeans": "Legume that prefers moderate temperatures and consistent moisture.",
                "Chickpea": "Cool-season legume that grows well in semi-arid regions.",
                "Coffee": "Tropical crop that prefers shade, consistent moisture, and specific altitude ranges."
            }
            
            if crop_name in crop_info:
                st.info(crop_info[crop_name])
            
        else:
            st.markdown('<p class="prediction-text" style="color: #d32f2f;">⚠️ Could not determine a suitable crop</p>', unsafe_allow_html=True)
            st.markdown('<p>Please check your input values and try again.</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)