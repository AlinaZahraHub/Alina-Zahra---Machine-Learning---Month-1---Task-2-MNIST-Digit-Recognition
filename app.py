import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import os

# Page Configuration
st.set_page_config(
    page_title="MNIST Digit Recognizer | AI Workspace",
    page_icon="🤖",
    layout="wide"
)

# Premium Dark Neon Cyberpunk UI Design with Custom Progress Bars CSS
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #070913;
        background-image: 
            linear-gradient(rgba(255, 101, 132, 0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 198, 255, 0.02) 1px, transparent 1px);
        background-size: 35px 35px;
        color: #f8fafc;
    }
    
    /* Hide Default Header Elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Extra Large Prominent Title Styling */
    .app-title {
        font-size: 4.8rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #00c6ff 0%, #ff6584 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
        letter-spacing: -2px;
        line-height: 1.1;
    }
    .app-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.5rem !important;
        margin-bottom: 45px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #0d111c;
        border: 1px solid rgba(0, 198, 255, 0.3);
        border-radius: 12px;
        color: #94a3b8;
        padding: 12px 30px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 198, 255, 0.2) 0%, rgba(255, 101, 132, 0.2) 100%);
        border: 1px solid rgba(255, 101, 132, 0.7);
        color: #ffffff;
    }

    /* Result Card Container */
    .result-container {
        background: rgba(13, 17, 28, 0.9);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 198, 255, 0.4);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 198, 255, 0.15);
        margin-top: 20px;
    }
    .predicted-digit {
        font-size: 6.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00c6ff 0%, #ff6584 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin: 10px 0;
    }
    .confidence-text {
        font-size: 1.3rem;
        color: #34d399;
        font-weight: 700;
    }

    /* Custom Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00c6ff 0%, #ff6584 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        font-weight: 800;
        font-size: 1.05rem;
        width: 100%;
        box-shadow: 0 0 20px rgba(255, 101, 132, 0.4);
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 0 30px rgba(255, 101, 132, 0.8);
    }
    </style>
""", unsafe_allow_html=True)

# Header Section with Extra Large Title
st.markdown('<p class="app-title">MNIST Digit Recognizer</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Real-time Deep Learning Classification Workspace</p>', unsafe_allow_html=True)

# Model Loading
MODEL_PATH = 'mnist_digit_model.h5'

@st.cache_resource
def load_cnn_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return None

model = load_cnn_model()

if model is None:
    st.error("❌ Error: 'mnist_digit_model.h5' model file not found in directory!")
else:
    # Interface Tabs
    tab1, tab2 = st.tabs(["✏️ Draw Digit", "☁️ Upload Image"])

    # ---------------- TAB 1: DRAW DIGIT ----------------
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns([1, 1], gap="large")
        
        with col_left:
            st.markdown("<h3 style='color: #ff6584;'>✏️ Draw a Digit (0-9)</h3>", unsafe_allow_html=True)
            st.markdown("<span style='color: #94a3b8; font-size: 1rem;'>Sketch a clear, bold digit inside the box below.</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=20,
                stroke_color="white",
                background_color="black",
                width=260,
                height=260,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            predict_draw_btn = st.button("⚡ Run Prediction", key="btn_draw")

        with col_right:
            st.markdown("<h3 style='color: #00c6ff;'>📊 Class Probabilities (0-9)</h3>", unsafe_allow_html=True)
            st.markdown("<span style='color: #94a3b8; font-size: 1rem;'>Live confidence breakdown per digit class.</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            prob_placeholder_draw = st.empty()
            with prob_placeholder_draw.container():
                st.info("👈 Draw a digit on the left and click **Run Prediction** to view probabilities.")

        # Logic for Drawing Prediction
        if canvas_result.image_data is not None and predict_draw_btn:
            with st.spinner("Processing drawing pixels..."):
                input_image = canvas_result.image_data
                processed_image = Image.fromarray(input_image.astype('uint8'), mode="RGBA").convert('L')
                
                # Image Preprocessing Pipeline
                image = ImageOps.fit(processed_image, (28, 28), Image.Resampling.LANCZOS)
                img_array = np.array(image)
                
                if img_array.mean() > 127:
                    img_array = 255 - img_array

                img_array = img_array / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)

                # Prediction
                preds = model.predict(img_array)[0]
                predicted_digit = int(np.argmax(preds))
                confidence = float(np.max(preds)) * 100

                # Render Result on Left (beneath canvas)
                with col_left:
                    st.markdown(f"""
                        <div class="result-container">
                            <span style="color: #94a3b8; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px;">Predicted Result</span>
                            <div class="predicted-digit">{predicted_digit}</div>
                            <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)

                # Render Probabilities on Right
                with col_right:
                    with prob_placeholder_draw.container():
                        st.markdown("<div style='padding-top: 5px;'></div>", unsafe_allow_html=True)
                        for i, prob in enumerate(preds):
                            prob_val = float(prob)
                            percentage = prob_val * 100
                            is_predicted = (i == predicted_digit)
                            
                            label_color = "#34d399" if is_predicted else "#94a3b8"
                            font_weight = "800" if is_predicted else "400"
                            
                            bar_html = f"""
                            <div style="margin-bottom: 12px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                    <span style="color: {label_color}; font-weight: {font_weight}; font-size: 1.05rem;">Digit {i}:</span>
                                    <span style="color: {label_color}; font-weight: {font_weight}; font-size: 0.95rem;">{percentage:.1f}%</span>
                                </div>
                                <div style="background-color: #0d111c; border: 1px solid rgba(0, 198, 255, 0.2); border-radius: 10px; height: 14px; width: 100%; overflow: hidden;">
                                    <div style="background: linear-gradient(135deg, #00c6ff 0%, #ff6584 100%); width: {percentage}%; height: 100%; border-radius: 8px; box-shadow: 0 0 10px rgba(255, 101, 132, 0.6);"></div>
                                </div>
                            </div>
                            """
                            st.markdown(bar_html, unsafe_allow_html=True)

    # ---------------- TAB 2: UPLOAD IMAGE ----------------
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        col_up_left, col_up_right = st.columns([1, 1], gap="large")
        
        with col_up_left:
            st.markdown("<h3 style='color: #ff6584;'>☁️ Upload an Image</h3>", unsafe_allow_html=True)
            st.markdown("<span style='color: #94a3b8; font-size: 1rem;'>Browse or drop a digit image file below.</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"], key="upload_file_box")
            
            uploaded_image = None
            if uploaded_file is not None:
                uploaded_image = Image.open(uploaded_file).convert('L')
                st.image(uploaded_image, caption="Source Image Preview", width=220)
            
            st.markdown("<br>", unsafe_allow_html=True)
            predict_up_btn = st.button("🚀 Analyze Upload", key="btn_up")

        with col_up_right:
            st.markdown("<h3 style='color: #00c6ff;'>📊 Upload Analysis & Probabilities</h3>", unsafe_allow_html=True)
            st.markdown("<span style='color: #94a3b8; font-size: 1rem;'>Live confidence breakdown for uploaded image.</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            prob_placeholder_upload = st.empty()
            with prob_placeholder_upload.container():
                st.info("👈 Upload an image on the left and click **Analyze Upload** to view results.")

        # Logic for Upload Prediction inside Tab 2
        if uploaded_image is not None and predict_up_btn:
            with st.spinner("Processing uploaded image pixels..."):
                # Image Preprocessing Pipeline
                image_up = ImageOps.fit(uploaded_image, (28, 28), Image.Resampling.LANCZOS)
                img_array_up = np.array(image_up)
                
                if img_array_up.mean() > 127:
                    img_array_up = 255 - img_array_up

                img_array_up = img_array_up / 255.0
                img_array_up = img_array_up.reshape(1, 28, 28, 1)

                # Prediction
                preds_up = model.predict(img_array_up)[0]
                predicted_digit_up = int(np.argmax(preds_up))
                confidence_up = float(np.max(preds_up)) * 100

                # Render Result on Left (beneath preview in tab2)
                with col_up_left:
                    st.markdown(f"""
                        <div class="result-container">
                            <span style="color: #94a3b8; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px;">Predicted Result</span>
                            <div class="predicted-digit">{predicted_digit_up}</div>
                            <div class="confidence-text">Confidence: {confidence_up:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)

                # Render Probabilities on Right in tab2
                with col_up_right:
                    with prob_placeholder_upload.container():
                        st.markdown("<div style='padding-top: 5px;'></div>", unsafe_allow_html=True)
                        for i, prob in enumerate(preds_up):
                            prob_val = float(prob)
                            percentage = prob_val * 100
                            is_predicted = (i == predicted_digit_up)
                            
                            label_color = "#34d399" if is_predicted else "#94a3b8"
                            font_weight = "800" if is_predicted else "400"
                            
                            bar_html = f"""
                            <div style="margin-bottom: 12px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                                    <span style="color: {label_color}; font-weight: {font_weight}; font-size: 1.05rem;">Digit {i}:</span>
                                    <span style="color: {label_color}; font-weight: {font_weight}; font-size: 0.95rem;">{percentage:.1f}%</span>
                                </div>
                                <div style="background-color: #0d111c; border: 1px solid rgba(0, 198, 255, 0.2); border-radius: 10px; height: 14px; width: 100%; overflow: hidden;">
                                    <div style="background: linear-gradient(135deg, #00c6ff 0%, #ff6584 100%); width: {percentage}%; height: 100%; border-radius: 8px; box-shadow: 0 0 10px rgba(255, 101, 132, 0.6);"></div>
                                </div>
                            </div>
                            """
                            st.markdown(bar_html, unsafe_allow_html=True)
