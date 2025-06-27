import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# Attempt to load environment variables for secret keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ----- Protobuf Compatibility Fix -----
try:
    import google.protobuf
    # Check protobuf version
    from packaging import version
    pb_version = version.parse(google.protobuf.__version__)
    if pb_version > version.parse("3.20.3"):
        raise ImportError("Protobuf version too high")
except:
    # Attempt to force compatible version
    try:
        import subprocess
        import sys
        subprocess.check_call([
            sys.executable, 
            "-m", "pip", "install", 
            "protobuf==3.20.3", 
            "--no-deps", 
            "--force-reinstall"
        ])
        import google.protobuf
    except Exception as e:
        print(f"Protobuf installation failed: {e}", file=sys.stderr)
        sys.exit(1)

# Now import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import models, layers, applications
    from tensorflow.keras.preprocessing import image
    print(f"TensorFlow {tf.__version__} loaded successfully")
except ImportError as e:
    print(f"TensorFlow import error: {e}", file=sys.stderr)
    sys.exit(1)

# Rest of your imports...
import json
import requests
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import folium
from streamlit_folium import st_folium
import time

# Try to import modern UI packages with fallbacks
try:
    from streamlit_extras.card import card
    from streamlit_extras.stylable_container import stylable_container
    modern_ui = True
except ImportError:
    modern_ui = False

try:
    import plotly.express as px
    plotly_available = True
except ImportError:
    plotly_available = False

try:
    from annotated_text import annotated_text
    annotated_text_available = True
except ImportError:
    annotated_text_available = False

# ──────── Helper Functions ─────────────────────────────────────────────────────

def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]
    grads = tape.gradient(loss, last_conv)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = last_conv[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_img = image.array_to_img(jet_heatmap)
    jet_img = jet_img.resize((img.shape[1], img.shape[0]))
    jet_arr = image.img_to_array(jet_img)
    superimposed = jet_arr * alpha + img
    return image.array_to_img(superimposed)

def load_hospital_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, "hospital.csv")
    return pd.read_csv(csv_path)

# ──────── Streamlit Page Config ───────────────────────────────────────────────

st.set_page_config(
    page_title="OrthoScan: Knee OA Diagnostic Suite",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────── Custom CSS ──────────────────────────────────────────────────────────

st.markdown("""
<style>
:root {
    --primary: #0d5699;
    --secondary: #00c0b4;
    --light: #f0f9ff;
    --dark: #0a2d4d;
    --danger: #e63946;
    --warning: #ff9e00;
    --success: #2a9d8f;
}

* {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f0f9ff 0%, #e6f7ff 100%);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--dark);
    font-weight: 600;
}

.stTabs > div > div > div > div {
    background-color: white;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
}

.stTab {
    padding: 10px 20px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease;
}

.stTab:hover {
    background-color: rgba(13, 86, 153, 0.1) !important;
}

.stTab[aria-selected="true"] {
    color: var(--primary) !important;
    font-weight: 600;
    border-bottom: 3px solid var(--primary) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, #00a3e0 100%);
    color: white !important;
    border-radius: 8px;
    border: none;
    padding: 10px 24px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(13, 86, 153, 0.25);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(13, 86, 153, 0.35);
}

.card {
    background: white;
    border-radius: 16px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
    padding: 24px;
    transition: all 0.3s ease;
    border: 1px solid rgba(0, 0, 0, 0.05);
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
}

.chat-message-user {
    background: var(--primary) !important;
    color: white !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 16px !important;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
}

.chat-message-assistant {
    background: #edf7ff !important;
    color: var(--dark) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 12px 16px !important;
    margin: 8px 0;
    max-width: 80%;
    margin-right: auto;
    border: 1px solid rgba(13, 86, 153, 0.1);
}

.stSpinner > div {
    border: 4px solid rgba(13, 86, 153, 0.2);
    border-radius: 50%;
    border-top: 4px solid var(--primary);
    width: 36px;
    height: 36px;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.stMetric {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    border-left: 4px solid var(--primary);
}

.st-emotion-cache-1v0mbdj {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
}

.stDataFrame {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ──────── Load Models ──────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    try:
        model = keras.models.load_model("./src/models/model_Xception_ft.hdf5")
        gm = keras.models.clone_model(model)
        gm.set_weights(model.get_weights())
        gm.layers[-1].activation = None
        grad_model = keras.models.Model(
            inputs=gm.inputs,
            outputs=[gm.get_layer("global_average_pooling2d_1").input, gm.output]
        )
        return model, grad_model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.error("Please check the model file path and format")
        st.stop()

try:
    model, grad_model = load_models()
    hospitals_df = load_hospital_data()
except Exception as e:
    st.error(f"❌ Initialization error: {str(e)}")
    st.stop()

class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
target_size = (224, 224)

# ──────── Header ──────────────────────────────────────────────────────────────

st.title("🦴 OrthoScan: Knee OA Diagnostic Suite")
st.caption("Advanced radiographic assessment and clinical decision support system")

# Show TensorFlow status
st.sidebar.caption(f" AI Healthcare hackathon")

# ──────── Tab Interface ───────────────────────────────────────────────────────

tab_xray, tab_chat = st.tabs(["📊 Radiographic Assessment", "💬 Clinical Consultation"])

# ──────── Radiographic Assessment Tab ─────────────────────────────────────────

with tab_xray:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        with st.container(border=True):
            st.subheader("📸 Image Upload")
            st.markdown("Upload a knee radiograph for analysis")
            uploaded_file = st.file_uploader("Choose an X-ray image", 
                                             type=["png", "jpg", "jpeg"], 
                                             label_visibility="collapsed")
            
            if uploaded_file:
                st.session_state["uploaded_file"] = uploaded_file
                st.image(uploaded_file, use_container_width=True)
            else:
                st.info("Please upload a knee X-ray to begin analysis")
                st.image("app/img/Sample1.png", use_container_width=True)

    with col2:
        if "uploaded_file" in st.session_state:
            if st.button("🔍 Initiate Analysis", use_container_width=True):
                with st.spinner("Performing deep radiographic assessment..."):
                    try:
                        img = image.load_img(
                            st.session_state["uploaded_file"], target_size=target_size
                        )
                        img_arr = image.img_to_array(img)
                        st.session_state["orig_img"] = img_arr.copy()
                        inp = np.expand_dims(img_arr, 0).astype("float32")
                        inp = applications.xception.preprocess_input(inp)

                        preds = model.predict(inp, verbose=0)[0]
                        idx = int(np.argmax(preds))
                        prob = preds[idx] * 100

                        st.session_state.update({
                            "preds": preds,
                            "grade_idx": idx,
                            "prob": prob,
                            "input_arr": inp,
                            "pred_done": True
                        })
                        st.toast("Analysis complete!", icon="✅")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

        if st.session_state.get("pred_done", False):
            with st.container(border=True):
                st.subheader("🧬 Pathological Focus Analysis")
                with st.spinner("Generating tissue-level visualization..."):
                    try:
                        heatmap = make_gradcam_heatmap(
                            grad_model, st.session_state["input_arr"], st.session_state["grade_idx"]
                        )
                        cam_img = save_and_display_gradcam(st.session_state["orig_img"], heatmap)
                        st.image(cam_img, use_container_width=True)
                        st.caption("Heatmap showing pathological regions contributing to diagnostic assessment")
                    except Exception as e:
                        st.error(f"Heatmap generation failed: {str(e)}")

    # ──────── Diagnostic Results ──────────────────────────────────────────────
    if st.session_state.get("pred_done", False):
        st.divider()
        
        col3, col4 = st.columns([1, 2])
        
        with col3:
            with st.container(border=True):
                st.subheader("📋 Diagnostic Summary")
                grade = class_names[st.session_state['grade_idx']]
                severity_colors = {
                    "Healthy": "#2a9d8f",
                    "Doubtful": "#8ac926",
                    "Minimal": "#ff9e00",
                    "Moderate": "#fb8500",
                    "Severe": "#e63946"
                }
                
                if annotated_text_available:
                    annotated_text(
                        ("Severity Grade: ", "", "#0d9249"),
                        (grade, "", severity_colors[grade])
                    )
                    
                    annotated_text(
                        ("Confidence Level: ", "", "#0f6faf"),
                        (f"{st.session_state['prob']:.1f}%", "", "#0d5699")
                    )
                else:
                    st.markdown(f"**Severity Grade:** <span style='color:{severity_colors[grade]};font-weight:bold'>{grade}</span>", 
                                unsafe_allow_html=True)
                    st.markdown(f"**Confidence Level:** {st.session_state['prob']:.1f}%")
                
                st.progress(int(st.session_state['prob']), text="Diagnostic Confidence")
                
                st.subheader("📈 Severity Distribution")
                if plotly_available:
                    fig = px.bar(
                        x=class_names,
                        y=st.session_state["preds"] * 100,
                        labels={'x': 'KL Grade', 'y': 'Probability (%)'},
                        color=class_names,
                        color_discrete_sequence=['#2a9d8f', '#8ac926', '#ff9e00', '#fb8500', '#e63946']
                    )
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=250
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback to matplotlib
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(class_names, st.session_state["preds"] * 100, 
                            color=['#2a9d8f', '#8ac926', '#ff9e00', '#fb8500', '#e63946'])
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('Probability (%)')
                    ax.set_title('Severity Distribution')
                    st.pyplot(fig)

        with col4:
            with st.container(border=True):
                st.subheader("🩺 Clinical Recommendations")
                
                grade = class_names[st.session_state["grade_idx"]]
                recommendations = {
                    "Healthy": [
                        "✅ No evidence of degenerative changes",
                        "Maintain healthy lifestyle with regular low-impact exercise",
                        "Consider periodic monitoring every 2-3 years"
                    ],
                    "Doubtful": [
                        "⚠️ Minimal degenerative changes observed",
                        "Implement preventive measures: weight management, strengthening exercises",
                        "Follow up in 12-18 months with orthopedic evaluation"
                    ],
                    "Minimal": [
                        "🟡 Early osteoarthritic changes detected",
                        "Begin conservative management: physical therapy, NSAIDs as needed",
                        "Orthopedic consultation recommended within 6 months"
                    ],
                    "Moderate": [
                        "🔴 Moderate joint degeneration confirmed",
                        "Urgent orthopedic evaluation required",
                        "Consider advanced interventions: viscosupplementation, bracing"
                    ],
                    "Severe": [
                        "❗ Severe joint space narrowing and osteophyte formation",
                        "Immediate orthopedic consultation indicated",
                        "Surgical options evaluation: osteotomy or arthroplasty"
                    ]
                }
                
                for rec in recommendations[grade]:
                    st.markdown(f"- {rec}")
                
                st.divider()
                
                if grade != "Healthy":
                    st.subheader("🏥 Orthopedic Care Network")
                    st.info("Specialized facilities near you:")
                    
                    state = st.selectbox("Select State", sorted(hospitals_df['State'].unique()))
                    district = st.selectbox("Select District", sorted(hospitals_df[hospitals_df['State'] == state]['District'].unique()))

                    filtered_hospitals = hospitals_df[
                        (hospitals_df['State'] == state) & 
                        (hospitals_df['District'] == district)
                    ].head(3)

                    if filtered_hospitals.empty:
                        st.warning("No specialized facilities in selected region")
                    else:
                        for _, row in filtered_hospitals.iterrows():
                            with st.expander(f"🏨 **{row['Hospital Name']}**", expanded=True):
                                cols = st.columns([1, 3])
                                with cols[0]:
                                    st.metric("Specialty", row['Specialty'])
                                with cols[1]:
                                    st.caption(f"📍 {row['Address']}")
                                
                                # Map with fallback
                                try:
                                    st.map(
                                        pd.DataFrame({
                                            'lat': [row['Lat']],
                                            'lon': [row['Lon']]
                                        }),
                                        zoom=12,
                                        use_container_width=True
                                    )
                                except:
                                    st.warning("Map display unavailable")

# ──────── Clinical Consultation Tab ───────────────────────────────────────────

with tab_chat:
    st.subheader("💬 Patient Consultation Portal")
    st.caption("AI-assisted clinical decision support for knee osteoarthritis")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    for role, content in st.session_state.chat_history:
        avatar = "🧑‍⚕️" if role == "assistant" else "👤"
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
    
    # User input
    if prompt := st.chat_input("Ask about knee OA symptoms or management..."):
        # Add user message to chat history
        st.session_state.chat_history.append(("user", prompt))
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://orthoscan-system.streamlit.app",
            "X-Title": "OrthoScan Medical Assistant"
        }
        
        # Build message history
        messages = [{"role": role, "content": content} 
                   for role, content in st.session_state.chat_history]
        
        payload = {
            "model": "deepseek/deepseek-r1:free",
            "messages": messages,
            "stream": True
        }
        
        # Display assistant response
        with st.chat_message("assistant", avatar="🧑‍⚕️"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Show processing indicator
                with st.spinner("Analyzing your query..."):
                    # Make API request
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        stream=True,
                        timeout=400  # Increased timeout for streaming responses
                    )
                
                # Handle non-200 responses
                if response.status_code != 200:
                    error_data = response.json().get("error", {})
                    error_msg = error_data.get("message", f"API Error {response.status_code}")
                    raise Exception(error_msg)
                
                # Create a status indicator
                status_container = st.empty()
                status_container.info("🔍 Processing response...")
                
                # Stream the response
                for chunk in response.iter_lines():
                    if chunk:
                        chunk_str = chunk.decode("utf-8")
                        if chunk_str.startswith("data: "):
                            try:
                                data = json.loads(chunk_str[6:])
                                if data.get("choices"):
                                    content = data["choices"][0]["delta"].get("content", "")
                                    if content:
                                        full_response += content
                                        # Update the response incrementally
                                        message_placeholder.markdown(full_response + "▌")
                            except json.JSONDecodeError:
                                continue
                
                # Remove the processing status
                status_container.empty()
                
                # Add final response
                message_placeholder.markdown(full_response)
                st.session_state.chat_history.append(("assistant", full_response))
                
            except Exception as e:
                # Remove processing status if error occurs
                if 'status_container' in locals():
                    status_container.empty()
                    
                error_msg = f"🚨 Connection error: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.chat_history.append(("assistant", error_msg))

    if st.button("🔄 Start New Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
# ──────── Footer ──────────────────────────────────────────────────────────────

st.divider()
st.caption("© 2024 OrthoScan Diagnostics | For clinical use only | v2.2.0")