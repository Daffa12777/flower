"""
Week 3 ML — Model Deployment
Aplikasi klasifikasi gambar bunga menggunakan MobileNetV2 (hasil Week 2).
Framework: Streamlit. UI: Apple-inspired (Inter, maroon/white, minimalis).

Navigasi memakai tombol menu (hamburger) buatan sendiri berbasis session_state,
sehingga selalu tampil baik di desktop maupun mobile.

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

st.set_page_config(page_title="Flower Classification", layout="centered",
                   initial_sidebar_state="collapsed")


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


# ================= State navigasi =================
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "menu_open" not in st.session_state:
    st.session_state.menu_open = False

def go(page_name):
    st.session_state.page = page_name
    st.session_state.menu_open = False


# ================= CSS =================
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
.block-container{ max-width:900px; padding-top:1.2rem; padding-bottom:3rem; }
#MainMenu, footer{ visibility:hidden; }
[data-testid="stHeader"], header{ background:transparent !important; box-shadow:none !important; }
[data-testid="stToolbar"], [data-testid="stDecoration"]{ display:none !important; }
[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"]{ display:none !important; }

h1,h2,h3,h4{ font-family:'Inter',sans-serif !important; color:var(--text); letter-spacing:-0.02em; }

/* ---------- Tombol umum (maroon) ---------- */
.stButton>button{
  background:var(--maroon) !important; color:#fff !important; border:none !important;
  border-radius:14px !important; padding:.7rem 1.5rem !important; font-weight:600 !important;
  font-size:15px !important; font-family:'Inter',sans-serif !important;
  transition:background .2s ease, transform .2s ease;
  box-shadow:0 4px 14px rgba(122,31,43,.25) !important;
}
.stButton>button:hover{ background:var(--maroon-hover) !important; transform:translateY(-1px); }

/* Tombol menu (hamburger) — kotak maroon kecil */
#menu-btn .stButton>button{
  width:52px !important; height:48px !important; padding:0 !important;
  font-size:22px !important; line-height:1 !important; border-radius:13px !important;
}

/* Tombol navigasi di dalam menu (sekunder / abu) */
#nav-menu .stButton>button{
  background:var(--white) !important; color:var(--text) !important;
  border:1px solid var(--border) !important; box-shadow:none !important;
  width:100% !important; text-align:left !important; padding:.75rem 1.1rem !important;
}
#nav-menu .stButton>button:hover{ background:var(--gray) !important; transform:none; }

/* Brand */
.brand{ font-size:20px; font-weight:800; color:var(--maroon); letter-spacing:-0.02em;
  line-height:48px; }

/* Card */
.card{ background:var(--white); border:1px solid var(--border); border-radius:20px; padding:28px;
  box-shadow:0 8px 24px rgba(0,0,0,.06); transition:transform .25s ease, box-shadow .25s ease;
  animation:fade .5s ease; margin-bottom:6px; }
.card:hover{ transform:translateY(-3px); box-shadow:0 12px 32px rgba(0,0,0,.10); }
@keyframes fade{ from{opacity:0; transform:translateY(8px);} to{opacity:1; transform:none;} }
.card h4{ margin:0 0 18px; font-size:19px; font-weight:700; }

[data-testid="stVerticalBlockBorderWrapper"]{ background:var(--white);
  border:1px solid var(--border) !important; border-radius:20px !important;
  box-shadow:0 8px 24px rgba(0,0,0,.06); padding:10px 6px; }

.hero-title{ font-size:44px; font-weight:800; line-height:1.08; margin:0 0 14px; letter-spacing:-0.03em; }
.hero-sub{ font-size:18px; color:var(--muted); line-height:1.5; }
.accent{ color:var(--maroon); }
.eyebrow{ color:var(--maroon); font-weight:700; font-size:13px; letter-spacing:.08em;
  text-transform:uppercase; margin-bottom:8px; }

[data-testid="stFileUploader"]{ border:1.5px dashed var(--border); border-radius:16px; padding:12px; }
[data-testid="stFileUploader"]:hover{ border-color:var(--maroon); }

.res-row{ display:flex; justify-content:space-between; align-items:center; padding:15px 0;
  border-bottom:1px solid var(--border); }
.res-row:last-child{ border-bottom:none; }
.res-label{ color:var(--muted); font-size:15px; }
.res-value{ font-weight:700; font-size:17px; }
.badge{ display:inline-block; background:var(--maroon); color:#fff; padding:5px 16px;
  border-radius:999px; font-weight:600; font-size:15px; }
.conf-head{ display:flex; justify-content:space-between; align-items:baseline; margin-top:16px; }
.conf-val{ font-weight:700; font-size:17px; color:var(--text); }
.bar-wrap{ background:var(--gray); border-radius:999px; height:12px; overflow:hidden; margin-top:8px; }
.bar-fill{ height:100%; background:var(--maroon); border-radius:999px; }
.prob-item{ margin-top:16px; }
.prob-head{ display:flex; justify-content:space-between; align-items:baseline; }
.prob-name{ font-weight:600; font-size:15px; color:var(--text); text-transform:capitalize; }
.prob-pct{ font-weight:600; font-size:15px; color:var(--muted); }

.flow{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; justify-content:center; }
.flow-step{ background:var(--gray); border:1px solid var(--border); border-radius:14px;
  padding:14px 22px; font-weight:600; font-size:15px; }
.flow-arrow{ color:var(--maroon); font-size:20px; font-weight:700; }

.info-item{ padding:16px 0; border-bottom:1px solid var(--border); }
.info-item:last-child{ border-bottom:none; }
.info-k{ color:var(--muted); font-size:14px; }
.info-v{ font-weight:600; font-size:16px; margin-top:3px; }

.footer{ text-align:center; color:var(--muted); font-size:14px; padding:34px 0 6px;
  margin-top:26px; border-top:1px solid var(--border); }
.small{ color:var(--muted); font-size:14px; line-height:1.6; }
.menu-title{ color:var(--muted); font-size:12px; font-weight:700; letter-spacing:.08em;
  text-transform:uppercase; margin:2px 0 6px; }
</style>
""", unsafe_allow_html=True)


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


# ================= TOP BAR: tombol menu (☰) + brand =================
bar_l, bar_r = st.columns([1, 6])
with bar_l:
    st.markdown('<div id="menu-btn">', unsafe_allow_html=True)
    if st.button("☰", key="btn_menu", help="Buka menu"):
        st.session_state.menu_open = not st.session_state.menu_open
    st.markdown('</div>', unsafe_allow_html=True)
with bar_r:
    st.markdown('<div class="brand">Flower AI</div>', unsafe_allow_html=True)

# Panel menu (muncul saat tombol ☰ ditekan)
if st.session_state.menu_open:
    with st.container(border=True):
        st.markdown('<div id="nav-menu">', unsafe_allow_html=True)
        st.markdown('<div class="menu-title">Menu</div>', unsafe_allow_html=True)
        for label in ["Home", "Prediction", "Model Information", "Documentation"]:
            if st.button(label, key=f"nav_{label}"):
                go(label)
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

st.write("")
page = st.session_state.page

# ================= Load model =================
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
    # Tombol langsung pindah ke halaman Prediction
    if st.button("Choose Image", key="btn_choose"):
        go("Prediction")
        st.rerun()

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
        with st.container(border=True):
            st.markdown("#### Upload Image")
            uploaded = st.file_uploader("Upload gambar bunga (JPG/PNG)",
                                        type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            image, go_predict = None, False
            if uploaded:
                image = Image.open(uploaded)
                st.image(image, caption="Preview", use_container_width=True)
                go_predict = st.button("Predict", key="btn_predict")
            else:
                st.markdown('<div class="small">Belum ada gambar. Pilih file untuk memulai.</div>',
                            unsafe_allow_html=True)

        if uploaded and go_predict:
            with st.spinner("Analyzing..."):
                pred, conf, probs, infer_ms = predict(model, image)
            st.write("")
            st.markdown(f"""
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
            </div>""", unsafe_allow_html=True)

            order = np.argsort(probs)[::-1]
            rows = ""
            for i in order:
                pct = probs[i] * 100
                rows += (f'<div class="prob-item"><div class="prob-head">'
                         f'<span class="prob-name">{CLASS_NAMES[i]}</span>'
                         f'<span class="prob-pct">{pct:.2f}%</span></div>'
                         f'<div class="bar-wrap"><div class="bar-fill" style="width:{pct:.1f}%"></div>'
                         f'</div></div>')
            st.write("")
            st.markdown(f'<div class="card"><h4>All Class Probabilities</h4>{rows}</div>',
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
    st.markdown('<div class="card"><h4>Deployment Framework</h4>'
                '<div class="small"><b>Streamlit</b> — dipilih karena cepat, sederhana, dan cocok '
                'untuk membangun aplikasi demo Machine Learning tanpa kode front-end terpisah.</div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="card"><h4>ML Framework</h4>'
                '<div class="small"><b>TensorFlow / Keras</b> — digunakan untuk membangun, melatih, '
                'dan memuat model MobileNetV2 pada tahap inferensi.</div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="card"><h4>Run Locally</h4></div>', unsafe_allow_html=True)
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    render_footer()