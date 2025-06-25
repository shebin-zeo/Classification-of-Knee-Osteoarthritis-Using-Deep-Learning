import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import folium
from streamlit_folium import st_folium

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
    jet_img = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_img = jet_img.resize((img.shape[1], img.shape[0]))
    jet_arr = tf.keras.preprocessing.image.img_to_array(jet_img)
    superimposed = jet_arr * alpha + img
    return tf.keras.preprocessing.image.array_to_img(superimposed)

def load_hospital_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, "hospital.csv")
    return pd.read_csv(csv_path)

# ──────── Streamlit Page Config ───────────────────────────────────────────────

st.set_page_config(
    page_title="Knee OA Severity Detector",
    page_icon="🦵",
    layout="wide"
)

# ──────── Load Models ──────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    model = tf.keras.models.load_model("./src/models/model_Xception_ft.hdf5")
    gm = tf.keras.models.clone_model(model)
    gm.set_weights(model.get_weights())
    gm.layers[-1].activation = None
    grad_model = tf.keras.models.Model(
        inputs=gm.inputs,
        outputs=[gm.get_layer("global_average_pooling2d_1").input, gm.output]
    )
    return model, grad_model

model, grad_model = load_models()
hospitals_df = load_hospital_data()

class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
target_size = (224, 224)

# ──────── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("app/img/Sample1.png", width=150)
    st.subheader("Knee OA Severity Predictor")
    uploaded_file = st.file_uploader("Choose an X-ray image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file

# ──────── Main Interface ───────────────────────────────────────────────────────

st.title("🦵 Knee Osteoarthritis Detection & Hospital Finder")
st.markdown("Analyze X-ray for OA severity and locate expert hospitals across India.")

col1, col2 = st.columns(2)

if "pred_done" not in st.session_state:
    st.session_state["pred_done"] = False

with col1:
    st.subheader("📷 Upload X-ray Image")
    if "uploaded_file" in st.session_state:
        st.image(st.session_state["uploaded_file"], use_container_width=True)
    else:
        st.info("Upload a knee X-ray to begin analysis.")

    if st.button("🔄 Analyze Severity") and "uploaded_file" in st.session_state:
        img = tf.keras.preprocessing.image.load_img(
            st.session_state["uploaded_file"], target_size=target_size
        )
        img_arr = tf.keras.preprocessing.image.img_to_array(img)
        st.session_state["orig_img"] = img_arr.copy()
        inp = np.expand_dims(img_arr, 0).astype("float32")
        inp = tf.keras.applications.xception.preprocess_input(inp)

        preds = model.predict(inp)[0]
        idx = int(np.argmax(preds))
        prob = preds[idx] * 100

        st.session_state.update({
            "preds": preds,
            "grade_idx": idx,
            "prob": prob,
            "input_arr": inp,
            "pred_done": True
        })

with col2:
    if st.session_state["pred_done"]:
        st.subheader("🔍 Grad-CAM Heatmap")
        heatmap = make_gradcam_heatmap(
            grad_model, st.session_state["input_arr"], st.session_state["grade_idx"]
        )
        cam_img = save_and_display_gradcam(st.session_state["orig_img"], heatmap)
        st.image(cam_img, use_container_width=True)

# ──────── Prediction & Analysis ────────────────────────────────────────────────

if st.session_state["pred_done"]:
    st.markdown("---")
    st.subheader("✅ Prediction Result")
    st.metric(
        label="OA Severity Grade",
        value=class_names[st.session_state['grade_idx']],
        delta=f"{st.session_state['prob']:.2f}%"
    )

    st.subheader("📊 Prediction Probabilities")
    fig, ax = plt.subplots(figsize=(6, 2))
    probs = st.session_state["preds"] * 100
    bars = ax.barh(class_names, probs)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    for bar, p in zip(bars, probs):
        ax.text(p + 1, bar.get_y() + bar.get_height() / 2, f"{p:.1f}%")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # ──────── Hospital Finder ──────────────────────────────────────────────────

    st.markdown("---")
    st.subheader("🏥 Find Nearest OA Specialty Hospitals")
    state = st.selectbox("Select State", sorted(hospitals_df['State'].unique()))
    district = st.selectbox("Select District", sorted(hospitals_df[hospitals_df['State'] == state]['District'].unique()))

    filtered_hospitals = hospitals_df[(hospitals_df['State'] == state) & (hospitals_df['District'] == district)]

    if filtered_hospitals.empty:
        st.warning("No matching hospitals found in the selected location.")
    else:
        st.success(f"Found {len(filtered_hospitals)} hospitals in {district}, {state}:")
        for _, row in filtered_hospitals.iterrows():
            st.markdown(f"**{row['Hospital Name']}**  ")
            st.caption(f"Specialty: {row['Specialty']}  | Address: {row['Address']}")

        st.subheader("📍 Map of Hospitals")
        m = folium.Map(location=[filtered_hospitals.iloc[0]['Lat'], filtered_hospitals.iloc[0]['Lon']], zoom_start=11)
        for _, row in filtered_hospitals.iterrows():
            folium.Marker(
                location=[row['Lat'], row['Lon']],
                popup=row['Hospital Name'],
                tooltip=row['Specialty'],
                icon=folium.Icon(color='red', icon='plus-sign')
            ).add_to(m)
        st_folium(m, width=700)

# ──────── Footer ───────────────────────────────────────────────────────────────

st.caption("💡 Restart app with: `streamlit run app/app.py`")
