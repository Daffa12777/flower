"""
Week 3 ML — Model Deployment
Aplikasi klasifikasi gambar bunga menggunakan MobileNetV2 (hasil Week 2).
Framework: Streamlit. UI: Apple-inspired (Inter, maroon/white, minimalis).

Alur: user -> interface (Streamlit) -> model (.keras) -> output (kelas + probabilitas)
Jalankan:  streamlit run app.py
"""
import time
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ================= Konfigurasi (JANGAN diubah – logika model) =================
MODEL_PATH  = "mobilenetv2_flowers.keras"
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
IMG_SIZE    = (224, 224)

st.set_page_config(page_title="Flower Classification", layout="wide",
                   initial_sidebar_state="expanded")


# ================= Logika Model (tidak diubah) =================
@st.cache_resource
def load_flower_model(path):
    return tf.keras.models.load_model(path)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image):
    batch = preprocess(image)
    t0 = time.perf_counter()
    probs = model.predict(batch, verbose=0)[0]
    infer_ms = (time.perf_counter() - t0) * 1000
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs, infer_ms


# ================= CSS (Apple-inspired) =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --maroon:#7A1F2B; --maroon-hover:#651722;
  --gray:#F5F5F7; --white:#FFFFFF; --text:#1D1D1F; --muted:#6E6E73; --border:#E5E5E5;
}
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"]{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif !important; color:var(--text);
}
.stApp{ background:var(--gray); }
.block-container{ max-width:1200px; padding-top:2.2rem; padding-bottom:3rem; }
#MainMenu, footer{ visibility:hidden; }
/* Header dibuat transparan (hilangkan bar hitam) tapi tombol sidebar tetap tampil */
[data-testid="stHeader"], header{ background:transparent !important; box-shadow:none !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"]{ display:none !important; }
/* Tombol buka sidebar dibuat jelas (maroon) */
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"]{
  visibility:visible !important; display:flex !important; top:14px !important; left:14px !important;
}
[data-testid="stSidebarCollapsedControl"] button, [data-testid="collapsedControl"] button{
  background:var(--maroon) !important; border-radius:12px !important; padding:8px !important;
  box-shadow:0 4px 14px rgba(122,31,43,.30) !important; border:none !important;
}
[data-testid="stSidebarCollapsedControl"] button:hover, [data-testid="collapsedControl"] button:hover{
  background:var(--maroon-hover) !important;
}
[data-testid="stSidebarCollapsedControl"] svg, [data-testid="collapsedControl"] svg{
  color:#fff !important; fill:#fff !important; width:22px !important; height:22px !important;
}

/* Sidebar putih + teks jelas */
[data-testid="stSidebar"]{ background:var(--white); border-right:1px solid var(--border); }
[data-testid="stSidebar"] .block-container{ padding-top:1.6rem; }
[data-testid="stSidebar"] [role="radiogroup"] label p{
  color:var(--text) !important; font-weight:500 !important; font-size:16px !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label{ padding:6px 0; }
[data-testid="stSidebar"] h3{ color:var(--maroon) !important; font-weight:800 !important; font-size:22px !important; }

h1,h2,h3,h4{ font-family:'Inter',sans-serif !important; color:var(--text); letter-spacing:-0.02em; }

/* Card (dipakai sebagai 1 blok HTML utuh) */
.card{
  background:var(--white); border:1px solid var(--border); border-radius:20px;
  padding:30px; box-shadow:0 8px 24px rgba(0,0,0,.06);
  transition:transform .25s ease, box-shadow .25s ease; animation:fade .5s ease; margin-bottom:6px;
}
.card:hover{ transform:translateY(-3px); box-shadow:0 12px 32px rgba(0,0,0,.10); }
@keyframes fade{ from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:none;} }
.card h4{ margin:0 0 18px; font-size:19px; font-weight:700; }

/* Container interaktif (upload) -> jadikan seperti card */
[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--white); border:1px solid var(--border) !important; border-radius:20px !important;
  box-shadow:0 8px 24px rgba(0,0,0,.06); padding:10px 6px;
}

/* Hero */
.hero-title{ font-size:52px; font-weight:800; line-height:1.08; margin:0 0 14px; letter-spacing:-0.03em; }
.hero-sub{ font-size:20px; color:var(--muted); max-width:640px; line-height:1.5; }
.accent{ color:var(--maroon); }
.eyebrow{ color:var(--maroon); font-weight:700; font-size:13px; letter-spacing:.08em;
  text-transform:uppercase; margin-bottom:8px; }

/* Tombol maroon */
.stButton>button, .stDownloadButton>button{
  background:var(--maroon) !important; color:#fff !important; border:none !important;
  border-radius:14px !important; padding:.7rem 1.6rem !important; font-weight:600 !important;
  font-family:'Inter',sans-serif !important; transition:background .2s ease, transform .2s ease;
  box-shadow:0 4px 14px rgba(122,31,43,.25);
}
.stButton>button:hover, .stDownloadButton>button:hover{
  background:var(--maroon-hover) !important; transform:translateY(-1px);
}
[data-testid="stFileUploader"]{ border:1.5px dashed var(--border); border-radius:16px; padding:12px; }
[data-testid="stFileUploader"]:hover{ border-color:var(--maroon); }

/* Baris hasil */
.res-row{ display:flex; justify-content:space-between; align-items:center;
  padding:15px 0; border-bottom:1px solid var(--border); }
.res-row:last-child{ border-bottom:none; }
.res-label{ color:var(--muted); font-size:15px; }
.res-value{ font-weight:700; font-size:17px; }
.badge{ display:inline-block; background:var(--maroon); color:#fff; padding:5px 16px;
  border-radius:999px; font-weight:600; font-size:15px; }

/* Confidence & probabilitas */
.conf-head{ display:flex; justify-content:space-between; align-items:baseline; margin-top:16px; }
.conf-val{ font-weight:700; font-size:17px; color:var(--text); }
.bar-wrap{ background:var(--gray); border-radius:999px; height:12px; overflow:hidden; margin-top:8px; }
.bar-fill{ height:100%; background:var(--maroon); border-radius:999px; }
.prob-item{ margin-top:16px; }
.prob-head{ display:flex; justify-content:space-between; align-items:baseline; }
.prob-name{ font-weight:600; font-size:15px; color:var(--text); text-transform:capitalize; }
.prob-pct{ font-weight:600; font-size:15px; color:var(--muted); }

/* Workflow */
.flow{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; justify-content:center; }
.flow-step{ background:var(--gray); border:1px solid var(--border); border-radius:14px;
  padding:14px 22px; font-weight:600; font-size:15px; }
.flow-arrow{ color:var(--maroon); font-size:20px; font-weight:700; }

/* Info grid */
.info-item{ padding:16px 0; border-bottom:1px solid var(--border); }
.info-item:last-child{ border-bottom:none; }
.info-k{ color:var(--muted); font-size:14px; }
.info-v{ font-weight:600; font-size:16px; margin-top:3px; }

.footer{ text-align:center; color:var(--muted); font-size:14px; padding:34px 0 6px;
  margin-top:26px; border-top:1px solid var(--border); }
.small{ color:var(--muted); font-size:14px; line-height:1.6; }
</style>
""", unsafe_allow_html=True)


# ================= Helper (semua card = 1 blok HTML) =================
FLOW_HTML = """
<div class="flow">
  <div class="flow-step">User</div><span class="flow-arrow">&rarr;</span>
  <div class="flow-step">Upload Image</div><span class="flow-arrow">&rarr;</span>
  <div class="flow-step">MobileNetV2</div><span class="flow-arrow">&rarr;</span>
  <div class="flow-step">Prediction</div><span class="flow-arrow">&rarr;</span>
  <div class="flow-step">Result</div>
</div>"""

def render_footer():
    st.markdown('<div class="footer">Week 3 Machine Learning Deployment<br/>'
                'Created by <b>Muhammad Daffa Fadlurrahman</b></div>', unsafe_allow_html=True)


# ================= Sidebar =================
with st.sidebar:
    st.markdown("### Flower AI")
    st.write("")
    page = st.radio("Navigation", ["Home", "Prediction", "Model Information", "Documentation"],
                    label_visibility="collapsed")

model, model_ok, load_err = None, False, ""
try:
    model = load_flower_model(MODEL_PATH)
    model_ok = True
except Exception as e:
    load_err = str(e)


# ================= HOME =================
if page == "Home":
    st.markdown('<div class="eyebrow">Week 3 · Machine Learning Deployment</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Machine Learning<br/>'
                '<span class="accent">Flower Classification</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload an image and let the trained MobileNetV2 model '
                'classify the flower instantly.</div>', unsafe_allow_html=True)
    st.write("")
    if st.button("Choose Image"):
        st.info("Buka menu **Prediction** lewat tombol menu (☰) merah di pojok kiri atas.")

    st.write("")
    st.markdown('<div class="eyebrow" style="margin-top:22px;">How it works</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card">{FLOW_HTML}</div>', unsafe_allow_html=True)

    if not model_ok:
        st.warning(f"Model belum termuat: pastikan `{MODEL_PATH}` ada di folder aplikasi.")
    render_footer()


# ================= PREDICTION =================
elif page == "Prediction":
    st.markdown('<div class="eyebrow">Prediction</div>', unsafe_allow_html=True)
    st.markdown("## Classify a Flower")
    st.markdown('<div class="small">Supported classes: daisy, dandelion, roses, sunflowers, tulips.</div>',
                unsafe_allow_html=True)
    st.write("")

    if not model_ok:
        st.error(f"Gagal memuat model `{MODEL_PATH}`. Letakkan file model di folder aplikasi.\n\n{load_err}")
    else:
        left, right = st.columns([1, 1], gap="large")

        with left:
            with st.container(border=True):
                st.markdown("#### Upload Image")
                uploaded = st.file_uploader("Upload gambar bunga (JPG/PNG)",
                                            type=["jpg", "jpeg", "png"], label_visibility="collapsed")
                image, go = None, False
                if uploaded:
                    image = Image.open(uploaded)
                    st.image(image, caption="Preview", use_container_width=True)
                    go = st.button("Predict")
                else:
                    st.markdown('<div class="small">Belum ada gambar. Pilih file untuk memulai.</div>',
                                unsafe_allow_html=True)

        with right:
            if uploaded and go:
                with st.spinner("Analyzing..."):
                    pred, conf, probs, infer_ms = predict(model, image)

                # Result card — 1 blok HTML utuh
                result_html = f"""
                <div class="card">
                  <h4>Result</h4>
                  <div class="res-row"><span class="res-label">Prediction</span>
                    <span class="badge">{pred}</span></div>
                  <div class="conf-head"><span class="res-label">Confidence Score</span>
                    <span class="conf-val">{conf*100:.1f}%</span></div>
                  <div class="bar-wrap"><div class="bar-fill" style="width:{conf*100:.1f}%"></div></div>
                  <div class="res-row" style="margin-top:6px;"><span class="res-label">Inference Time</span>
                    <span class="res-value">{infer_ms:.1f} ms</span></div>
                  <div class="res-row"><span class="res-label">Model</span>
                    <span class="res-value">MobileNetV2</span></div>
                </div>"""
                st.markdown(result_html, unsafe_allow_html=True)

                # All probabilities — 1 blok HTML utuh
                order = np.argsort(probs)[::-1]
                rows = ""
                for i in order:
                    pct = probs[i] * 100
                    rows += (f'<div class="prob-item"><div class="prob-head">'
                             f'<span class="prob-name">{CLASS_NAMES[i]}</span>'
                             f'<span class="prob-pct">{pct:.2f}%</span></div>'
                             f'<div class="bar-wrap"><div class="bar-fill" '
                             f'style="width:{pct:.1f}%"></div></div></div>')
                st.markdown(f'<div class="card"><h4>All Class Probabilities</h4>{rows}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="card"><h4>Result</h4>'
                            '<div class="small">Hasil prediksi akan muncul di sini setelah kamu '
                            'mengunggah gambar dan menekan tombol <b>Predict</b>.</div></div>',
                            unsafe_allow_html=True)
    render_footer()


# ================= MODEL INFORMATION =================
elif page == "Model Information":
    st.markdown('<div class="eyebrow">Model Information</div>', unsafe_allow_html=True)
    st.markdown("## About the Model")
    st.write("")
    info = [
        ("Model", "MobileNetV2 (fine-tuned)"),
        ("Framework", "TensorFlow / Keras"),
        ("Dataset", "tf_flowers (5 kelas, ±3.670 gambar RGB)"),
        ("Accuracy", "0.88 (F1 0.87 · AUC 0.99)"),
        ("Input Size", "224 × 224 × 3 (RGB)"),
        ("Number of Classes", "5 — daisy, dandelion, roses, sunflowers, tulips"),
        ("Training Method", "Transfer Learning + Fine-tuning (ImageNet)"),
    ]
    items = "".join(f'<div class="info-item"><div class="info-k">{k}</div>'
                    f'<div class="info-v">{v}</div></div>' for k, v in info)
    st.markdown(f'<div class="card">{items}</div>', unsafe_allow_html=True)
    render_footer()


# ================= DOCUMENTATION =================
elif page == "Documentation":
    st.markdown('<div class="eyebrow">Documentation</div>', unsafe_allow_html=True)
    st.markdown("## How It Works")
    st.write("")

    st.markdown(f'<div class="card"><h4>Deployment Flow</h4>{FLOW_HTML}'
                '<div class="small" style="text-align:center; margin-top:16px;">'
                'User &rarr; Streamlit UI &rarr; Model &rarr; Prediction &rarr; Output</div></div>',
                unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="card"><h4>Deployment Framework</h4>'
                    '<div class="small"><b>Streamlit</b> — dipilih karena cepat, sederhana, dan cocok '
                    'untuk membangun aplikasi demo Machine Learning tanpa kode front-end terpisah.</div></div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h4>ML Framework</h4>'
                    '<div class="small"><b>TensorFlow / Keras</b> — digunakan untuk membangun, melatih, '
                    'dan memuat model MobileNetV2 pada tahap inferensi.</div></div>',
                    unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="card"><h4>Run Locally</h4></div>', unsafe_allow_html=True)
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    render_footer()