import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ──────── Helpers ─────────────────────────────────────────────────────────────

def make_gradcam_heatmap(grad_model, img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]
    grads = tape.gradient(loss, last_conv)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_output = last_conv[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:,:3]
    jet_heatmap = jet_colors[heatmap]
    jet_img = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_img = jet_img.resize((img.shape[1], img.shape[0]))
    jet_arr = tf.keras.preprocessing.image.img_to_array(jet_img)
    superimposed = jet_arr * alpha + img
    return tf.keras.preprocessing.image.array_to_img(superimposed)

# ──────── Page Config ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Severity Analysis of Arthrosis in the Knee",
    page_icon="🦵",
    layout="wide"
)

# ──────── Load & Cache Model ───────────────────────────────────────────────────

@st.cache_resource
def load_models():
    model = tf.keras.models.load_model("./src/models/model_Xception_ft.hdf5")
    # build grad_model
    gm = tf.keras.models.clone_model(model)
    gm.set_weights(model.get_weights())
    gm.layers[-1].activation = None
    grad_model = tf.keras.models.Model(
        inputs=gm.inputs,
        outputs=[
            gm.get_layer("global_average_pooling2d_1").input,
            gm.output
        ]
    )
    return model, grad_model

model, grad_model = load_models()

class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]
target_size = (224, 224)

# ──────── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("app/img/Sample1.png", width=150)
    st.subheader("Knee OA Severity Predictor")
    
    uploaded_file = st.file_uploader("Choose an X-ray image", type=["png","jpg","jpeg"])

    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file

# ──────── Main ─────────────────────────────────────────────────────────────────

st.header("Severity Analysis of Arthrosis in the Knee")
col1, col2 = st.columns(2)

# initialize session_state
if "pred_done" not in st.session_state:
    st.session_state["pred_done"] = False

with col1:
    st.subheader("📷 Input")
    if "uploaded_file" in st.session_state:
        st.image(st.session_state["uploaded_file"], use_container_width=True)
    else:
        st.info("Upload an X-ray in the sidebar to get started.")

    # Prediction button
    if st.button("🔄 Predict Arthrosis in the Knee") and "uploaded_file" in st.session_state:
        # load & preprocess
        img = tf.keras.preprocessing.image.load_img(
            st.session_state["uploaded_file"],
            target_size=target_size
        )
        img_arr = tf.keras.preprocessing.image.img_to_array(img)
        st.session_state["orig_img"] = img_arr.copy()
        inp = np.expand_dims(img_arr, 0).astype("float32")
        inp = tf.keras.applications.xception.preprocess_input(inp)

        # predict
        preds = model.predict(inp)[0]
        idx = int(np.argmax(preds))
        prob = preds[idx] * 100

        # store results
        st.session_state.update({
            "preds": preds,
            "grade_idx": idx,
            "prob": prob,
            "input_arr": inp,
            "pred_done": True
        })

with col2:
    if st.session_state["pred_done"]:
        st.subheader("🔍 Explainability")
        heatmap = make_gradcam_heatmap(
            grad_model,
            st.session_state["input_arr"],
            st.session_state["grade_idx"]
        )
        cam_img = save_and_display_gradcam(
            st.session_state["orig_img"], heatmap
        )
        st.image(cam_img, use_container_width=True)

# ──────── Prediction & Analysis (full width) ────────────────────────────────────

if st.session_state["pred_done"]:
    st.markdown("---")
    st.subheader("✅ Prediction")
    st.metric(
        label="Severity Grade",
        value=f"{class_names[st.session_state['grade_idx']]}",
        delta=f"{st.session_state['prob']:.2f}%"
    )

    st.subheader("📊 Analysis")
    fig, ax = plt.subplots(figsize=(6,2))
    probs = st.session_state["preds"] * 100
    bars = ax.barh(class_names, probs)
    ax.set_xlim(0,100); ax.invert_yaxis()
    for bar,p in zip(bars, probs):
        ax.text(p + 1, bar.get_y() + bar.get_height()/2, f"{p:.1f}%")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    st.pyplot(fig)

# ──────── Footer ────────────────────────────────────────────────────────────────

st.caption(
    "💡 Keep your terminal open. To restart:\n"
    "`streamlit run app/app.py`  (Ctrl+C to stop)"
)
