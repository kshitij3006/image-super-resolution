
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import io

# ---------------- Page & Styles ----------------
st.set_page_config(page_title="AI Super-Resolution Enhancer", page_icon="🖼️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.main-header{font-size:3rem;font-weight:bold;background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;padding:1rem 0;}
.sub-header{text-align:center;color:#666;font-size:1.2rem;margin-bottom:2rem}
.metric-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:1.5rem;border-radius:10px;color:#fff;text-align:center;box-shadow:0 4px 6px rgba(0,0,0,.1)}
.stButton>button{width:100%;background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);color:#fff;border:none;padding:.75rem;font-size:1.1rem;border-radius:8px;font-weight:bold}
.info-box{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:1.5rem;border-radius:12px;margin:1rem 0;box-shadow:0 4px 12px rgba(102,126,234,.3);color:#fff}
.info-box h3{color:#fff;font-size:1.5rem;margin-bottom:.5rem;font-weight:bold}
.info-box p{color:rgba(255,255,255,.95);font-size:1rem;margin:0}
</style>
""", unsafe_allow_html=True)

#  Session State 
st.session_state.setdefault('processed', False)
st.session_state.setdefault('lr_img', None)      # (1,128,128,3) float32 [0,1] RGB
st.session_state.setdefault('results', {})       # {name: {'image':(1,128,128,3), 'psnr', 'ssim'}}

#  Model Loading 
@st.cache_resource
def load_models():
    with st.spinner("Loading AI models..."):
        srcnn = load_model('srcnn_super_resolution_model.h5', compile=False)
        edsr  = load_model('edsr_best.h5', compile=False)
        vdsr  = load_model('vdsr_super_resolution_model.h5', compile=False)
    return srcnn, edsr, vdsr

try:
    srcnn, edsr, vdsr = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_loaded = False


def preprocess_image(img_pil, input_size=128):
    """
    PIL -> center-crop to square (no padding/mirroring), resize to input_size,
    normalize to [0,1], add batch dimension. RGB all the way.
    """
    rgb = np.array(img_pil.convert("RGB"))
    h, w = rgb.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    square = rgb[y0:y0+side, x0:x0+side]
    resized = cv2.resize(square, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    x = resized.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)  # (1,128,128,3) RGB

def to_uint8_rgb(x01):
    return (np.clip(x01[0], 0, 1) * 255).astype(np.uint8)

def png_bytes_from_rgb(rgb_uint8):
    img = Image.fromarray(rgb_uint8)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

def bicubic_baseline(x01):
    # For fixed 128 inputs, baseline is identical resolution; keep a copy.
    return x01.copy()

# ---------------- Metrics (RGB) ----------------
def compute_metrics(ref_x01_rgb, cand_x01_rgb, crop=0, win_default=7):
    ref = to_uint8_rgb(ref_x01_rgb); cand = to_uint8_rgb(cand_x01_rgb)
    if crop > 0 and ref.shape[0] > 2*crop and ref.shape[1] > 2*crop:
        ref  = ref[crop:-crop, crop:-crop, :]
        cand = cand[crop:-crop, crop:-crop, :]
    min_dim = min(ref.shape[:2])
    win_size = win_default if min_dim >= win_default else (min_dim if min_dim % 2 else max(min_dim-1, 1))
    psnr = peak_signal_noise_ratio(ref, cand, data_range=255)
    ssim = structural_similarity(ref, cand, channel_axis=-1, win_size=win_size, data_range=255)
    return psnr, ssim

# Header 
st.markdown('<p class="main-header">🖼️ AI Super-Resolution Enhancer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transform low-resolution images into high-quality results using deep learning</p>', unsafe_allow_html=True)

#  Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("### 📤 Upload Image")
    uploaded_image = st.file_uploader("", type=["jpg", "png", "jpeg"], help="Upload a low-resolution image")

    st.markdown("---")
    st.markdown("### 🤖 Model Selection")
    model_options = {"SRCNN":"Fast & lightweight","EDSR":"High fidelity (residual)","VDSR":"Very deep (detail)"}
    selected_models = [name for name in model_options if st.checkbox(name, value=(name=="SRCNN"), help=model_options[name])]

    st.markdown("---")
    st.markdown("### 🔄 Combination")
    combine_option = st.radio("How many models to combine", ["1 Model","2 Models","All Selected"], help="Equal-weight average")

    st.markdown("---")
    st.markdown("### 📊 Display & Metrics")
    show_metrics = st.checkbox("Show Metrics Cards/Table", value=True)
    show_graph   = st.checkbox("Show Performance Graphs", value=True)
    show_comp    = st.checkbox("Side-by-Side Comparison", value=True)
    metric_ref   = st.selectbox("Compare SR against", ["Bicubic baseline (demo)","Input (demo)"])
    crop_border  = st.number_input("Metric crop border (px)", min_value=0, max_value=12, value=0)

# Main Logic 
if not models_loaded:
    st.error("⚠️ Models could not be loaded. Check that .h5 files exist and are valid.")
elif uploaded_image is None:
    st.info("👆 Upload an image from the sidebar to begin.")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("<div class='info-box'><h3>1️⃣ Upload</h3><p>Select a low-res image</p></div>", unsafe_allow_html=True)
    with c2: st.markdown("<div class='info-box'><h3>2️⃣ Choose Models</h3><p>Pick SRCNN / EDSR / VDSR</p></div>", unsafe_allow_html=True)
    with c3: st.markdown("<div class='info-box'><h3>3️⃣ Enhance</h3><p>View, compare, download</p></div>", unsafe_allow_html=True)
elif len(selected_models) == 0:
    st.warning("⚠️ Please select at least one model in the sidebar.")
else:
    if st.button("🚀 Enhance Image", type="primary"):
        with st.spinner("Processing image..."):
            img = Image.open(uploaded_image).convert("RGB")
            lr_img = preprocess_image(img, input_size=128)  # (1,128,128,3) RGB
            st.session_state.lr_img = lr_img

            model_objs = {"SRCNN":srcnn, "EDSR":edsr, "VDSR":vdsr}
            picked = [(m, model_objs[m]) for m in selected_models]

            # Predict once per model
            results, preds = {}, {}
            for name, mdl in picked:
                sr = mdl.predict(lr_img, verbose=0)     # (1,128,128,3)
                preds[name] = sr
                ref = bicubic_baseline(lr_img) if metric_ref.startswith("Bicubic") else lr_img
                psnr, ssim = compute_metrics(ref, sr, crop=crop_border)
                results[name] = {'image': sr, 'psnr': psnr, 'ssim': ssim}

            # Ensemble
            def combine(names):
                acc = np.zeros_like(preds[names[0]], dtype=preds[names[0]].dtype)
                for nm in names: acc += preds[nm]
                return acc / len(names)

            if combine_option == "1 Model":
                names = [selected_models[0]]
            elif combine_option == "2 Models":
                names = selected_models[:2] if len(selected_models) >= 2 else [selected_models[0]]
            else:
                names = selected_models[:]

            combined_img = combine(names)
            ref = bicubic_baseline(lr_img) if metric_ref.startswith("Bicubic") else lr_img
            c_psnr, c_ssim = compute_metrics(ref, combined_img, crop=crop_border)
            results[f"Combined ({' + '.join(names)})"] = {'image': combined_img, 'psnr': c_psnr, 'ssim': c_ssim}

            st.session_state.results = results
            st.session_state.processed = True
            st.success("✅ Image enhanced successfully!")

    # ---------------- Results UI ----------------
    if st.session_state.processed and st.session_state.results:
        st.markdown("---")
        st.markdown("## 📊 Results")

        if show_metrics:
            cols = st.columns(len(st.session_state.results))
            for i, (name, data) in enumerate(st.session_state.results.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                      <h3>{name}</h3>
                      <p style="font-size:2rem;margin:.5rem 0;">{data['psnr']:.2f}</p>
                      <p style="margin:0;">PSNR (dB)</p>
                      <p style="font-size:1.5rem;margin:.5rem 0;">{data['ssim']:.4f}</p>
                      <p style="margin:0;">SSIM</p>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        if show_comp:
            st.markdown("### 🔍 Visual Comparison")
            cols = st.columns(len(st.session_state.results) + 1)
            with cols[0]:
                st.image(to_uint8_rgb(st.session_state.lr_img), caption="Input", use_container_width=True)
            for idx, (name, data) in enumerate(st.session_state.results.items()):
                with cols[idx + 1]:
                    rgb = to_uint8_rgb(data['image'])
                    st.image(rgb, caption=name, use_container_width=True)
                    st.download_button(
                        label="⬇️ Download",
                        data=png_bytes_from_rgb(rgb),
                        file_name=f"{name.replace(' ','_').replace('(','-').replace(')','-')}.png",
                        mime="image/png",
                        key=f"dl_{idx}_{name}"
                    )

        if show_graph:
            st.markdown("---")
            st.markdown("### 📈 Performance Metrics")
            df = pd.DataFrame({
                'Model': list(st.session_state.results.keys()),
                'PSNR': [d['psnr'] for d in st.session_state.results.values()],
                'SSIM': [d['ssim'] for d in st.session_state.results.values()]
            })
            c1, c2 = st.columns(2)
            with c1:
                fig1, ax1 = plt.subplots(figsize=(8,5))
                ax1.bar(df['Model'], df['PSNR'])
                ax1.set_ylabel('PSNR (dB)'); ax1.set_title('Peak Signal-to-Noise Ratio')
                ax1.grid(axis='y', alpha=0.3); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
                st.pyplot(fig1); plt.close(fig1)
            with c2:
                fig2, ax2 = plt.subplots(figsize=(8,5))
                ax2.bar(df['Model'], df['SSIM'])
                ax2.set_ylabel('SSIM'); ax2.set_title('Structural Similarity Index')
                ax2.grid(axis='y', alpha=0.3); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
                st.pyplot(fig2); plt.close(fig2)

            st.markdown("---")
            best_psnr = df.loc[df['PSNR'].idxmax()]
            best_ssim = df.loc[df['SSIM'].idxmax()]
            c1, c2 = st.columns(2)
            with c1: st.success(f"🏆 **Best PSNR**: {best_psnr['Model']} ({best_psnr['PSNR']:.2f} dB)")
            with c2: st.success(f"🏆 **Best SSIM**: {best_ssim['Model']} ({best_ssim['SSIM']:.4f})")

            if show_metrics:
                st.markdown("### 📋 Detailed Metrics")
                st.dataframe(df.style.highlight_max(axis=0, subset=['PSNR','SSIM']), use_container_width=True)

#  Footer
st.markdown("---")
_, mid, _ = st.columns([1,2,1])
with mid:
    st.markdown("""
    <div style='text-align:center;color:#666;'>
      <p>Created by <strong>Vishesh Arora</strong>, <strong>Manan Sethi</strong>, <strong>Kshitij Gupta</strong></p>
      <p style='font-size:.9rem;'>Powered by TensorFlow & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


