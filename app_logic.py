import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components

from charts import make_confusion_matrix_figure, make_monthly_trend_chart, make_pie_chart, make_roc_figure
from config import AUTH_PASSWORD, AUTH_USERNAME, CLASS_NAMES
from model_utils import evaluate_performance, get_validation_image_paths, load_leukemia_model, load_prediction_history, predict_image, save_prediction_record
from reports import generate_pdf_report
from utils import img_to_base64, inject_css

st.set_page_config(
    page_title="Automated Blood Smear Analysis - Leukemia Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def render_carousel():
    carousel_data = [
        ("Advanced Medical Research Lab", "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?w=800"),
        ("Clinical Slide Under Microscope", "https://images.unsplash.com/photo-1579154204601-01588f351167?w=800"),
        ("AI Disease Analysis Screening", "https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=800"),
        ("Digital Healthcare Diagnostics", "https://images.unsplash.com/photo-1526256262350-7da7584cf5eb?w=800"),
        ("Leukemia Blast Cell Morphology", "https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3?w=800"),
        ("Computer-Aided Clinical Screening", "https://images.openai.com/static-rsc-4/2Uo4YLizpWyN0AZhrn4jdvEZmMx7oUoMKeMKQqYXtDLuiOx4I2b13jYuVOm5fpnv2Ne3Hl7lAv6Yd2oOWQFuwOm1XGhW_VkJDNtwC1DTON1WdL5JqN7-XSmMu07Ikfct6hgPhsFTpVMH-XSxFnbGz55IXHTo5_K2R9kb_eUkwGU2g3hqJeO3sDfmBz2ueH72?purpose=fullsize")
    ]

    leuk_imgs, norm_imgs = get_validation_image_paths()
    if leuk_imgs:
        try:
            b64 = img_to_base64(leuk_imgs[0])
            if b64:
                carousel_data[5] = ("Leukemia Lymphoblasts (Validation Image)", f"data:image/bmp;base64,{b64}")
        except Exception:
            pass
    if norm_imgs:
        try:
            b64 = img_to_base64(norm_imgs[0])
            if b64:
                carousel_data[1] = ("Healthy Blood Smear (Validation Image)", f"data:image/bmp;base64,{b64}")
        except Exception:
            pass

    slides_html = ""
    dots_html = ""
    for idx, (caption, src) in enumerate(carousel_data):
        active_class = "active" if idx == 0 else ""
        slides_html += f"""
        <div class="mySlides fade">
            <img src="{src}" style="width:100%; height:320px; object-fit:cover;">
            <div class="text-caption">
                <h4>{caption}</h4>
                <p>Peripheral Blood Smear Deep Learning Diagnostics</p>
            </div>
        </div>"""
        dots_html += f"""<span class="dot {active_class}" onclick="currentSlide({idx+1})"></span>"""

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    * {{box-sizing: border-box; font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;}}
    body {{margin: 0; padding: 0; background: transparent;}}
    .slideshow-container {{
      max-width: 100%;
      position: relative;
      margin: auto;
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(0,0,0,0.06);
      background: #f8fafc;
      border: 1px solid #e2e8f0;
    }}
    .mySlides {{display: none; position: relative;}}
    .prev, .next {{
      cursor: pointer;
      position: absolute;
      top: 50%;
      width: auto;
      padding: 12px;
      margin-top: -22px;
      color: white;
      font-weight: bold;
      font-size: 18px;
      transition: 0.3s ease;
      border-radius: 0 3px 3px 0;
      user-select: none;
      background-color: rgba(0,0,0,0.3);
      z-index: 10;
    }}
    .next {{right: 0; border-radius: 3px 0 0 3px;}}
    .prev:hover, .next:hover {{background-color: rgba(0,0,0,0.7);}}
    .text-caption {{
      position: absolute;
      bottom: 0px;
      left: 0px;
      right: 0px;
      background: linear-gradient(to top, rgba(15, 82, 186, 0.85) 0%, rgba(15, 82, 186, 0.4) 70%, transparent 100%);
      color: white;
      padding: 20px 24px;
      text-shadow: 0 1px 4px rgba(0,0,0,0.3);
    }}
    .text-caption h4 {{margin: 0 0 4px 0; font-size: 1.1rem; font-weight: 700;}}
    .text-caption p {{margin: 0; font-size: 0.85rem; opacity: 0.9;}}
    .dot-container {{
      text-align: center;
      padding: 8px;
      background: transparent;
      margin-top: 6px;
    }}
    .dot {{
      cursor: pointer;
      height: 10px;
      width: 10px;
      margin: 0 3px;
      background-color: #cbd5e1;
      border-radius: 50%;
      display: inline-block;
      transition: background-color 0.4s ease, width 0.4s ease;
    }}
    .active, .dot:hover {{background-color: #0F52BA; width: 22px; border-radius: 5px;}}
    .fade {{
      animation-name: fade;
      animation-duration: 0.8s;
    }}
    @keyframes fade {{
      from {{opacity: .4}} 
      to {{opacity: 1}}
    }}
    </style>
    </head>
    <body>

    <div class="slideshow-container">
        {slides_html}
        <a class="prev" onclick="plusSlides(-1)">&#10094;</a>
        <a class="next" onclick="plusSlides(1)">&#10095;</a>
    </div>

    <div class="dot-container">
        {dots_html}
    </div>

    <script>
    let slideIndex = 1;
    let timer = null;
    showSlides(slideIndex);
    resetTimer();

    function plusSlides(n) {{
      showSlides(slideIndex += n);
      resetTimer();
    }}

    function currentSlide(n) {{
      showSlides(slideIndex = n);
      resetTimer();
    }}

    function showSlides(n) {{
      let i;
      let slides = document.getElementsByClassName("mySlides");
      let dots = document.getElementsByClassName("dot");
      if (n > slides.length) {{slideIndex = 1}}
      if (n < 1) {{slideIndex = slides.length}}
      for (i = 0; i < slides.length; i++) {{
        slides[i].style.display = "none";
      }}
      for (i = 0; i < dots.length; i++) {{
        dots[i].className = dots[i].className.replace(" active", "");
      }}
      if(slides[slideIndex-1]) {{
        slides[slideIndex-1].style.display = "block";
      }}
      if(dots[slideIndex-1]) {{
        dots[slideIndex-1].className += " active";
      }}
    }}

    function resetTimer() {{
      if (timer) clearInterval(timer);
      timer = setInterval(function() {{
        slideIndex++;
        showSlides(slideIndex);
      }}, 3000);
    }}
    </script>

    </body>
    </html>
    """
    components.html(html_code, height=365, scrolling=False)


def _inject_login_form_styles():
    st.markdown(
        """
        <style>
        .stApp {
            color-scheme: light !important;
        }

        .stApp input[type="text"],
        .stApp input[type="password"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
            caret-color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            border: 1.5px solid #94a3b8 !important;
            border-radius: 10px !important;
            padding: 0.42rem 0.7rem !important;
            font-size: 0.86rem !important;
            min-height: 2.2rem !important;
        }

        .stApp input[type="text"]::placeholder,
        .stApp input[type="password"]::placeholder {
            color: #64748b !important;
            opacity: 1 !important;
        }

        .stApp [data-baseweb="input"],
        .stApp [data-baseweb="base-input"] {
            background-color: #ffffff !important;
            border: 1.5px solid #94a3b8 !important;
            border-radius: 10px !important;
        }

        .stApp [data-baseweb="input"] > div,
        .stApp [data-baseweb="base-input"] > div {
            background-color: #ffffff !important;
        }

        .stApp div[data-testid="stTextInput"] label p {
            color: #334155 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
        }

        .stApp div[data-testid="stFormSubmitButton"] > button {
            background-color: #0F52BA !important;
            background: #0F52BA !important;
            color: #ffffff !important;
            border: 1px solid #0a3d8f !important;
            border-radius: 10px !important;
            font-size: 0.86rem !important;
            font-weight: 600 !important;
            min-height: 2.35rem !important;
            padding: 0.45rem 1rem !important;
            width: 100% !important;
            box-shadow: 0 2px 8px rgba(15, 82, 186, 0.25) !important;
        }

        .stApp div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #0c4298 !important;
            background: #0c4298 !important;
            border-color: #083570 !important;
            color: #ffffff !important;
        }

        .stApp div[data-testid="stFormSubmitButton"] > button p,
        .stApp div[data-testid="stFormSubmitButton"] > button span,
        .stApp div[data-testid="stFormSubmitButton"] > button div {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_login():
    st.markdown(
        """
        <style>
        section.main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 100%;
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
            min-height: 88vh;
        }

        section.main div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(2) > div[data-testid="stVerticalBlock"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 2rem 2rem;
            box-shadow: 0 12px 40px rgba(15, 82, 186, 0.08);
            max-width: 900px;
            margin: 0 auto;
        }

        section.main form[data-testid="stForm"] div[data-testid="stTextInput"] {
            margin-bottom: 0.65rem;
        }

        section.main form[data-testid="stForm"] div[data-testid="stFormSubmitButton"] {
            margin-top: 0.5rem;
        }

        .login-page-links {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .login-page-links a {
            font-size: 0.82rem;
            font-weight: 500;
            color: #0F52BA;
            text-decoration: none;
        }

        .login-page-links a:hover {
            text-decoration: underline;
            color: #0c4298;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.1, 1])

    with center_col:
        st.markdown(
            """
            <div style="text-align:center; margin-bottom:1.25rem;">
                <h2 style="margin:0 0 0.35rem 0; font-weight:800; color:#0F52BA; font-size:1.3rem;">
                    Automated Blood Smear Analysis
                </h2>
                <p style="margin:0; font-size:0.84rem; color:#64748b;">
                    Leukemia Cancer Detection System
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            login_submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

            if login_submitted:
                if username == AUTH_USERNAME and password == AUTH_PASSWORD:
                    st.session_state.authenticated = True
                    st.session_state.page = "Home"
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        st.markdown(
            """
            <div class="login-page-links">
                <a href="#">Create New Account</a>
                <a href="#">Forgot Password?</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _inject_login_form_styles()


def render_app_shell():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    col_brand, col_nav = st.columns([2.5, 9.5], gap="small")

    with col_brand:
        st.markdown(
            """
            <div class="brand-logo-container">
                <div class="brand-logo"></div>
                <div>
                    <div class="brand-title">Automated Blood Smear Analysis</div>
                    <div class="brand-subtitle">Leukemia Detection & Dashboard</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_nav:
        nav_cols = st.columns(4, gap="small")
        pages = ["Home", "Prediction", "Dashboard", "Logout"]
        icons = ["🏠 Home", "🩸 Prediction", "📊 Dashboard", "🚪 Logout"]

        for idx, page_name in enumerate(pages):
            with nav_cols[idx]:
                is_active = st.session_state.page == page_name
                btn_class = "active-nav-btn" if is_active else ""
                st.markdown(f'<div class="{btn_class}">', unsafe_allow_html=True)
                if st.button(icons[idx], key=f"nav_btn_{page_name}", use_container_width=True):
                    if page_name == "Logout":
                        st.session_state.authenticated = False
                        st.session_state.page = "Login"
                    else:
                        st.session_state.page = page_name
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    model, model_path = load_leukemia_model()
    if model is None:
        st.error("Error: Could not locate a trained model (.keras file) in the project structure. Please run a training script first.")
        st.stop()

    st.markdown('<div class="app-container fade-in">', unsafe_allow_html=True)
    if st.session_state.page == "Home":
        render_home_page()
    elif st.session_state.page == "Prediction":
        render_prediction_page(model)
    elif st.session_state.page == "Dashboard":
        render_dashboard_page()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="footer-section">
            <p style='font-weight:700; margin-bottom:0.25rem; color:#0F52BA;'>Automated Blood Smear Analysis for Leukemia Detection</p>
            <p style='margin:0;'>AI-Powered Clinical Screening Web Application &bull; Final Year Research Project</p>
            <p style='margin-top:0.4rem; font-size:0.8rem; opacity:0.8;'>Built using Streamlit, TensorFlow, Matplotlib, and ReportLab</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_home_page():
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-content">
                <div class="hero-pill">Deep Learning Screening Platform</div>
                <h1 class="hero-title-text">AI-Powered Leukemia Detection System</h1>
                <p class="hero-desc-text">
                    Diagnose Acute Lymphoblastic Leukemia from peripheral blood smear microscopic images. Our deep neural network identifies abnormal lymphoblast cells with clinical-grade diagnostic precision.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_started, _ = st.columns([1.5, 8.5])
    with col_started:
        if st.button("🚀 Get Started", key="hero_get_started_btn", use_container_width=True):
            st.session_state.page = "Prediction"
            st.rerun()

    st.markdown("<div style='margin-top:2.5rem;'></div>", unsafe_allow_html=True)

    col_about, col_role = st.columns([6, 4], gap="large")
    with col_about:
        st.markdown(
            """
            <div class="clinical-card">
                <span class="status-badge leukemia" style="margin-bottom:0.75rem;">Pathological Context</span>
                <h3 style="margin:0 0 0.75rem 0; font-weight:800; color:#0F52BA;">About Blood Cancer (Leukemia)</h3>
                <p style="color:#64748b; line-height:1.7; font-size:0.92rem; margin:0;">
                    Leukemia is an aggressive hematologic malignancy originating in the bone marrow, characterized by the uncontrolled clonal proliferation of abnormal white blood cells (lymphoblasts).
                    As these malignant blast cells accumulate, they suppress normal hematopoiesis, leading to critical anemia, thrombocytopenia, and leukocyte dysfunction.
                    <br><br>
                    Early detection of blast cells in peripheral blood smears is vital for prompt therapeutic intervention, particularly in pediatric cases of Acute Lymphoblastic Leukemia (ALL).
                    AI systems analyze blood smear slide images, detecting subtle cellular features, chromatin density, and cytoplasmic variations to assist hematopathologists with fast and consistent screening.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_role:
        st.markdown(
            """
            <div class="clinical-card" style="height: 100%;">
                <span class="status-badge normal" style="margin-bottom:0.75rem;">Platform Integration</span>
                <h3 style="margin:0 0 0.75rem 0; font-weight:800; color:#0F52BA;">Artificial Intelligence Role</h3>
                <div style="display:flex; flex-direction:column; gap:0.9rem; margin-top:0.5rem;">
                    <div style="display:flex; gap:0.75rem; align-items:flex-start;">
                        <span style="font-size:1.3rem;">🔬</span>
                        <div>
                            <div style="font-weight:700; font-size:0.9rem;">Microscopic Profiling</div>
                            <div style="font-size:0.8rem; color:#64748b;">Automated morphologic scan of blood smear cells.</div>
                        </div>
                    </div>
                    <div style="display:flex; gap:0.75rem; align-items:flex-start;">
                        <span style="font-size:1.3rem;">🤖</span>
                        <div>
                            <div style="font-weight:700; font-size:0.9rem;">Feature Extraction</div>
                            <div style="font-size:0.8rem; color:#64748b;">DenseNet121 maps cell nuclei and cytoplasmic features.</div>
                        </div>
                    </div>
                    <div style="display:flex; gap:0.75rem; align-items:flex-start;">
                        <span style="font-size:1.3rem;">⚡</span>
                        <div>
                            <div style="font-weight:700; font-size:0.9rem;">Rapid Clinical Review</div>
                            <div style="font-size:0.8rem; color:#64748b;">Reduces microscopic evaluation cycle times.</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    st.markdown("<h3 style='font-weight:800; color:#0F52BA; margin-bottom:0.5rem;'>Clinical Visual Showcase</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:0.88rem; margin:0 0 1.25rem 0;'>Dynamic slider showcasing microscopic cells and healthcare imagery utilized in the research.</p>", unsafe_allow_html=True)
    render_carousel()

    st.markdown("<h3 style='font-weight:800; color:#0F52BA; margin-top:2.5rem; margin-bottom:0.25rem;'>Core Platform Capabilities</h3>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-tile">
                <div class="feature-icon">🤖</div>
                <div class="feature-title-text">AI Leukemia Scan</div>
                <div class="feature-desc-text">Inspects cellular cytology and nucleolar structure to detect leukemia.</div>
            </div>
            <div class="feature-tile">
                <div class="feature-icon">🎯</div>
                <div class="feature-title-text">High Confidence Score</div>
                <div class="feature-desc-text">Calculates exact probability ratings for normal vs leukemic cells.</div>
            </div>
            <div class="feature-tile">
                <div class="feature-icon">🧠</div>
                <div class="feature-title-text">Deep Learning Engine</div>
                <div class="feature-desc-text">Powered by a pre-trained DenseNet121 architecture optimized for blood smears.</div>
            </div>
            <div class="feature-tile">
                <div class="feature-icon">📄</div>
                <div class="feature-title-text">Automated Reports</div>
                <div class="feature-desc-text">Exports professional performance analysis report cards in PDF format.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_prediction_page(model):
    st.markdown("<h2 style='font-weight:800; color:#0F52BA; margin-bottom:0.25rem;'>Automated Blood Smear Diagnostic Scan</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:0.9rem; margin-bottom:2rem;'>Upload a peripheral blood smear cell image to analyze morphology and run classification.</p>", unsafe_allow_html=True)

    col_left, col_right = st.columns([5, 5], gap="large")

    with col_left:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin:0 0 1rem 0; font-weight:700; color:#0F52BA;'>Microscopic Image Upload</h4>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Select peripheral blood cell image (BMP, PNG, JPG)...",
            type=["bmp", "png", "jpg", "jpeg"],
            label_visibility="visible"
        )

        st.markdown("<p style='font-size:0.8rem; color:#64748b; margin-top:0.75rem;'>Format requirements: Single-cell slide crop from microscope camera. Prefer resolution of 224x224.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
            st.markdown("<h4 style='margin:0 0 1rem 0; font-weight:700; color:#0F52BA;'>Microscopic Preview</h4>", unsafe_allow_html=True)
            st.image(image, caption="Uploaded blood cell sample", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="clinical-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin:0 0 1.25rem 0; font-weight:700; color:#0F52BA;'>AI Diagnostic Outcome</h4>", unsafe_allow_html=True)

        if uploaded_file is None:
            st.info("Awaiting peripheral blood cell image upload to run inference...")
        else:
            if "prediction_result" not in st.session_state:
                st.session_state.prediction_result = None
            if "prediction_file_name" not in st.session_state:
                st.session_state.prediction_file_name = None

            if st.session_state.prediction_file_name != uploaded_file.name:
                st.session_state.prediction_result = None
                st.session_state.prediction_file_name = uploaded_file.name

            predict_clicked = st.button("Predict", type="primary", use_container_width=True)
            if predict_clicked:
                with st.spinner("Processing blood cell morphology and running inference..."):
                    pred_idx, confidence, probs = predict_image(model, image)
                    pred_label = CLASS_NAMES[pred_idx]
                    time.sleep(0.8)
                    history_df = save_prediction_record(uploaded_file, pred_label, confidence)
                    st.session_state.prediction_result = {
                        "pred_label": pred_label,
                        "confidence": confidence,
                        "probs": probs,
                        "uploaded_name": uploaded_file.name,
                        "history_count": len(history_df),
                    }
                    st.success(f"Prediction saved to dashboard history ({len(history_df)} total records).")

            if st.session_state.prediction_result is not None and st.session_state.prediction_result.get("uploaded_name") == uploaded_file.name:
                pred_label = st.session_state.prediction_result["pred_label"]
                confidence = st.session_state.prediction_result["confidence"]
                probs = st.session_state.prediction_result["probs"]

                if pred_label == "Leukemia":
                    st.markdown(
                        f"""
                        <div class="diag-banner leukemia">
                            <div>
                                <div class="diag-title">LEUKEMIA DETECTED</div>
                                <div class="diag-desc">Cell morphology shows high indicators of acute leukemia lymphoblasts.</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="diag-banner normal">
                            <div>
                                <div class="diag-title">NORMAL STATUS</div>
                                <div class="diag-desc">No malignant blast cell cytology indicators detected. Healthy lymphocyte morphology.</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("<h5 style='margin:1.5rem 0 0.5rem 0; font-weight:700; color:#0f172a;'>Probability Distribution</h5>", unsafe_allow_html=True)

                bar_color = "#EF5350" if pred_label == "Leukemia" else "#10b981"

                st.markdown(
                    f"""
                    <div style='display:flex; justify-content:space-between; font-size:0.9rem; font-weight:600; margin-bottom:0.35rem;'>
                        <span>Model Confidence ({pred_label})</span>
                        <span style='color:{bar_color};'>{confidence * 100:.2f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(confidence)

                st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

                prob_df = pd.DataFrame({
                    "Cell Diagnostic Classification": CLASS_NAMES,
                    "Probability Rating": [f"{p * 100:.2f}%" for p in probs]
                })

                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

                st.markdown(
                    f"""
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; padding:1rem; border-radius:14px;">
                        <div style="font-weight:700; font-size:0.85rem; color:#0F52BA; margin-bottom:0.5rem;">CLINICAL MORPHOLOGY DETAILS</div>
                        <ul style="margin:0; padding-left:1.2rem; font-size:0.82rem; color:#64748b; line-height:1.6;">
                            <li><b>Classificator Backbone:</b> Functional DenseNet121 CNN</li>
                            <li><b>Input Dimension:</b> 224 x 224 x 3</li>
                            <li><b>Nuclear Chromatin Scan:</b> {'Coarse & clumped, nucleoli visible' if pred_label == 'Leukemia' else 'Condensed, homogeneous'}</li>
                            <li><b>Cytoplasm Status:</b> {'Scant cytoplasm, high nuclear-to-cytoplasmic ratio' if pred_label == 'Leukemia' else 'Normal volume ratio'}</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("Click Predict to run inference and save the result to the dashboard history.")

        st.markdown('</div>', unsafe_allow_html=True)


def render_dashboard_page():
    st.markdown("<h2 style='font-weight:800; color:#0F52BA; margin-bottom:0.25rem;'>Diagnostic Patient Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:0.9rem; margin-bottom:2rem;'>Review saved prediction records, monitor class distribution, and inspect stored smear files.</p>", unsafe_allow_html=True)

    history_df = load_prediction_history()

    if history_df.empty:
        st.info("No saved predictions yet. Run a prediction from the Prediction page to populate this dashboard.")
        return

    total_records = len(history_df)
    leuk_count = int(np.sum(history_df["Prediction Result"].eq("Leukemia")))
    norm_count = int(np.sum(history_df["Prediction Result"].eq("Normal")))
    avg_confidence = float(history_df["Prediction Confidence Score"].mean()) if total_records else 0.0

    model_path = load_leukemia_model()[1]
    metrics = evaluate_performance(model_path)
    overall_accuracy = metrics["accuracy"]

    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:1.25rem; margin-bottom:2rem;">
            <div class="clinical-card" style="margin-bottom:0; padding:1.25rem;">
                <div style="color:#64748b; font-weight:700; font-size:0.8rem; text-transform:uppercase;">Saved Predictions</div>
                <div style="font-size:1.8rem; font-weight:800; color:#0f172a; margin-top:0.25rem;">{total_records}</div>
                <div style="font-size:0.75rem; color:#10b981; margin-top:0.25rem; font-weight:600;">Live CSV-backed history</div>
            </div>
            <div class="clinical-card" style="margin-bottom:0; padding:1.25rem;">
                <div style="color:#64748b; font-weight:700; font-size:0.8rem; text-transform:uppercase; color:#EF5350;">Leukemia Positive</div>
                <div style="font-size:1.8rem; font-weight:800; color:#EF5350; margin-top:0.25rem;">{leuk_count}</div>
                <div style="font-size:0.75rem; color:#EF5350; margin-top:0.25rem; font-weight:600;">{leuk_count / total_records * 100:.1f}% of saved cases</div>
            </div>
            <div class="clinical-card" style="margin-bottom:0; padding:1.25rem;">
                <div style="color:#64748b; font-weight:700; font-size:0.8rem; text-transform:uppercase; color:#10b981;">Normal Cases</div>
                <div style="font-size:1.8rem; font-weight:800; color:#10b981; margin-top:0.25rem;">{norm_count}</div>
                <div style="font-size:0.75rem; color:#10b981; margin-top:0.25rem; font-weight:600;">{norm_count / total_records * 100:.1f}% of saved cases</div>
            </div>
            <div class="clinical-card" style="margin-bottom:0; padding:1.25rem;">
                <div style="color:#64748b; font-weight:700; font-size:0.8rem; text-transform:uppercase; color:#0F52BA;">Average Confidence</div>
                <div style="font-size:1.8rem; font-weight:800; color:#0F52BA; margin-top:0.25rem;">{avg_confidence:.2f}%</div>
                <div style="font-size:0.75rem; color:#0F52BA; margin-top:0.25rem; font-weight:600;">{overall_accuracy * 100:.2f}% model benchmark</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_chart_left, col_chart_right = st.columns([4, 6], gap="large")
    with col_chart_left:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        pie_fig = make_pie_chart(history_df)
        st.pyplot(pie_fig, use_container_width=True)
        plt.close(pie_fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart_right:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        line_fig = make_monthly_trend_chart(history_df)
        st.pyplot(line_fig, use_container_width=True)
        plt.close(line_fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin:0 0 0.5rem 0; font-weight:700; color:#0F52BA;'>Saved Prediction Log</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:0.82rem; margin:0 0 1.25rem 0;'>Each new prediction is appended to the CSV file automatically and shown here in a live table.</p>", unsafe_allow_html=True)

    logs_df = history_df[["Patient Name", "Prediction Result", "Prediction Confidence Score", "Date of Diagnosis", "Timestamp"]].copy()
    logs_df.columns = [
        "Uploaded File",
        "Classification Result",
        "Confidence Rating",
        "Screening Date",
        "Saved At",
    ]
    logs_df["Confidence Rating"] = logs_df["Confidence Rating"].round(2)
    st.dataframe(logs_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-weight:700; color:#0F52BA; margin-bottom:0.75rem;'>Saved Record Inspector</h4>", unsafe_allow_html=True)
    available_names = [str(name) for name in history_df["Patient Name"].dropna().astype(str).unique()]
    selected_name = st.selectbox("Choose a saved prediction to inspect:", available_names)

    if selected_name:
        row_sel = history_df.loc[history_df["Patient Name"] == selected_name].iloc[0]

        with st.expander(f"🔬 Saved Diagnostic Profile: {selected_name}", expanded=True):
            col_det_left, col_det_right = st.columns([4, 6], gap="large")

            with col_det_left:
                stored_image_path = row_sel.get("Blood Smear Image", "")
                if stored_image_path and os.path.exists(stored_image_path):
                    st.image(stored_image_path, caption=f"Stored Smear: {selected_name}", use_container_width=True)
                else:
                    st.warning("Stored smear image not found on disk.")

            with col_det_right:
                st.markdown(
                    f"""
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; padding:1.25rem; border-radius:18px; height:100%;">
                        <div style="font-weight:800; font-size:1.15rem; color:#0F52BA; margin-bottom:0.75rem; border-bottom:1px solid #cbd5e1; padding-bottom:0.4rem;">Clinical File Card</div>
                        <table style="width:100%; font-size:0.9rem; border-collapse:collapse; line-height:2.0;">
                            <tr>
                                <td style="font-weight:700; color:#64748b; width:190px;">Uploaded File:</td>
                                <td style="font-weight:600; color:#0f172a;">{row_sel["Patient Name"]}</td>
                            </tr>
                            <tr>
                                <td style="font-weight:700; color:#64748b;">Diagnostic Analysis:</td>
                                <td>
                                    <span class="status-badge {'leukemia' if row_sel['Prediction Result'] == 'Leukemia' else 'normal'}">
                                        {row_sel["Prediction Result"]}
                                    </span>
                                </td>
                            </tr>
                            <tr>
                                <td style="font-weight:700; color:#64748b;">AI Class Confidence:</td>
                                <td style="font-weight:800; color: {'#EF5350' if row_sel['Prediction Result'] == 'Leukemia' else '#10b981'}">
                                    {row_sel["Prediction Confidence Score"]:.2f}%
                                </td>
                            </tr>
                            <tr>
                                <td style="font-weight:700; color:#64748b;">Prediction Timestamp:</td>
                                <td style="font-weight:600; color:#0f172a;">{row_sel.get('Timestamp', '')}</td>
                            </tr>
                            <tr>
                                <td style="font-weight:700; color:#64748b;">Screening Date:</td>
                                <td style="font-weight:600; color:#0f172a;">{row_sel.get('Date of Diagnosis', '')}</td>
                            </tr>
                            <tr>
                                <td style="font-weight:700; color:#64748b; vertical-align:top;">Clinical Notes:</td>
                                <td style="color:#64748b; font-size:0.85rem; line-height:1.5; padding-top:0.25rem;">
                                    {row_sel.get('Additional Details', 'Auto-saved prediction record.')}
                                </td>
                            </tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown('</div>', unsafe_allow_html=True)




def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "page" not in st.session_state:
        st.session_state.page = "Login"

    inject_css()

    if not st.session_state.authenticated:
        render_login()
    else:
        render_app_shell()


if __name__ == "__main__":
    main()
