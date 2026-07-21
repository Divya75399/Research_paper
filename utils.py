import base64
import os

import streamlit as st

#UI Design + styling
def get_logo_svg_base64():
    svg_code = """
    <svg width="60" height="60" viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="30" cy="30" r="28" fill="url(#grad)" stroke="#FFFFFF" stroke-width="2"/>
        <ellipse cx="30" cy="30" rx="14" ry="10" fill="#EF5350" opacity="0.9"/>
        <ellipse cx="30" cy="30" rx="9" ry="6" fill="#D32F2F" opacity="0.6"/>
        <rect x="27" y="16" width="6" height="28" rx="2" fill="#FFFFFF"/>
        <rect x="16" y="27" width="28" height="6" rx="2" fill="#FFFFFF"/>
        <defs>
            <linearGradient id="grad" x1="0" y1="0" x2="60" y2="60" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stop-color="#0F52BA" />
                <stop offset="100%" stop-color="#1E88E5" />
            </linearGradient>
        </defs>
    </svg>
    """
    return base64.b64encode(svg_code.encode("utf-8")).decode("utf-8")


def img_to_base64(img_path):
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def inject_css():
    logo_b64 = get_logo_svg_base64()
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {{
        --bg-color: #f8fafc;
        --panel-color: #ffffff;
        --border-color: #e2e8f0;
        --text-color: #0f172a;
        --muted-color: #64748b;
        --primary-color: #0F52BA;
        --accent-color: #EF5350;
        --success-color: #10b981;
    }}
    
    /* Hide Streamlit default styling elements */
    div[data-testid="stHeader"] {{
        display: none !important;
    }}
    div[data-testid="stToolbar"] {{
        display: none !important;
    }}
    footer {{
        visibility: hidden !important;
        height: 0 !important;
        position: absolute !important;
    }}
    
    .stApp {{
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }}
    
    .app-container {{
        max-width: 1600px;
        width: 100%;
        margin: 0 auto;
        padding: 1.5rem 1.5rem 4rem 1.5rem;
    }}

    section.main .block-container {{
        max-width: 1600px !important;
        width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }}
    
    /* Topbar Header styling */
    .topbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(14px);
        border: 1px solid var(--border-color);
        border-radius: 24px;
        padding: 1.25rem 2rem;
        margin-bottom: 2.2rem;
        box-shadow: 0 18px 48px rgba(15, 82, 186, 0.08);
    }}

    .topbar > div {{
        width: 100%;
    }}

    .topbar button {{
        min-width: 180px !important;
        padding: 0.95rem 1.4rem !important;
        font-size: 1rem !important;
        border-radius: 18px !important;
        line-height: 1.1 !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }}

    .topbar button span,
    .topbar button div {{
        white-space: nowrap !important;
    }}

    .brand-logo-container {{
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }}

    .brand-title {{
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--primary-color);
        line-height: 1.15;
    }}
    
    .brand-logo-container {{
        display: flex;
        align-items: center;
        gap: 0.9 rem;
    }}
    
    .brand-logo {{
        width: 48px;
        height: 48px;
        background-image: url('data:image/svg+xml;base64,{logo_b64}');
        background-size: contain;
        background-repeat: no-repeat;
    }}
    
    .brand-title {{
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--primary-color);
        line-height: 1.15;
    }}
    
    .brand-subtitle {{
        font-size: 0.75rem;
        color: var(--muted-color);
    }}
    
    /* Modern Navigation tabs */
    .nav-buttons-container {{
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 0.75rem;
        flex-wrap: nowrap;
        white-space: nowrap;
    }}

    .topbar button {{
        min-width: 150px;
        padding: 0.9rem 1.2rem !important;
        font-size: 1rem !important;
        border-radius: 18px !important;
        line-height: 1.2 !important;
    }}
    
    /* Custom clinical cards */
    .clinical-card {{
        background: var(--panel-color);
        border: 1px solid var(--border-color);
        border-radius: 22px;
        padding: 1.5rem;
        box-shadow: 0 12px 28px rgba(31, 72, 109, 0.04);
        margin-bottom: 1.5rem;
    }}
    
    /* Hero section */
    .hero-container {{
        position: relative;
        background: linear-gradient(135deg, rgba(15, 82, 186, 0.92), rgba(13, 148, 136, 0.75)), url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=1200') center/cover no-repeat;
        border-radius: 26px;
        padding: 4.5rem 3rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 48px rgba(15, 82, 186, 0.15);
        overflow: hidden;
    }}
    
    .hero-container::before {{
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top right, rgba(239, 83, 80, 0.35), transparent 60%);
        z-index: 1;
    }}
    
    .hero-content {{
        position: relative;
        z-index: 2;
        max-width: 700px;
    }}
    
    .hero-pill {{
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.25);
        color: white;
        padding: 0.4rem 0.9rem;
        border-radius: 99px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 1.25rem;
    }}
    
    .hero-title-text {{
        font-size: clamp(2rem, 3.8vw, 3.25rem);
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 10px rgba(0,0,0,0.12);
    }}
    
    .hero-desc-text {{
        font-size: 1.1rem;
        line-height: 1.6;
        opacity: 0.95;
        margin-bottom: 2rem;
        text-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    
    /* Feature grids */
    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }}
    
    @media (max-width: 992px) {{
        .feature-grid {{
            grid-template-columns: 1fr 1fr;
        }}
    }}
    @media (max-width: 576px) {{
        .feature-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    
    .feature-tile {{
        background: #ffffff;
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(31, 72, 109, 0.02);
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.25s ease;
    }}
    
    .feature-tile:hover {{
        transform: translateY(-4px);
        box-shadow: 0 16px 36px rgba(15, 82, 186, 0.08);
        border-color: rgba(15, 82, 186, 0.2);
    }}
    
    .feature-icon {{
        font-size: 2.2rem;
        margin-bottom: 0.75rem;
        display: inline-block;
        padding: 0.5rem;
        border-radius: 16px;
        background: rgba(15, 82, 186, 0.07);
        color: var(--primary-color);
        width: 54px;
        height: 54px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }}
    
    .feature-title-text {{
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        color: var(--text-color);
    }}
    
    .feature-desc-text {{
        font-size: 0.85rem;
        color: var(--muted-color);
        line-height: 1.5;
    }}
    
    /* Custom button overrides — nav / page buttons */
    div[data-testid="column"]:not(:has([data-testid="stForm"])) button {{
        border-radius: 16px !important;
        font-size: 1rem !important;
        background-color: transparent !important;
        color: var(--primary-color) !important;
        border: 1.5px solid rgba(15, 82, 186, 0.25) !important;
        font-weight: 700 !important;
        transition: all 0.2s ease !important;
        height: 54px !important;
        min-width: 160px !important;
        padding: 0 1.5rem !important;
    }}
    
    div[data-testid="column"]:not(:has([data-testid="stForm"])) button:hover {{
        background-color: var(--primary-color) !important;
        color: white !important;
        border-color: var(--primary-color) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(15, 82, 186, 0.15) !important;
    }}
    
    /* Active button styling */
    .active-nav-btn button {{
        background-color: var(--primary-color) !important;
        color: white !important;
        border-color: var(--primary-color) !important;
    }}
    
    .stApp div[data-testid="stFormSubmitButton"] > button {{
        border-radius: 16px !important;
        font-size: 1rem !important;
        padding: 0.95rem 1.75rem !important;
        min-height: 54px !important;
        font-weight: 700 !important;
        background-color: var(--primary-color) !important;
        color: white !important;
        border: 1.5px solid var(--primary-color) !important;
    }}
    
    .stApp div[data-testid="stFormSubmitButton"] > button:hover {{
        background-color: #0c4298 !important;
        border-color: #0c4298 !important;
    }}
    
    /* Input and Form styling */
    .stTextInput input {{
        border-radius: 12px !important;
        border: 1.5px solid var(--border-color) !important;
        padding: 0.5rem 0.75rem !important;
        font-size: 0.9rem !important;
    }}
    
    .stTextInput input:focus {{
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 2px rgba(15, 82, 186, 0.1) !important;
    }}
    
    /* Custom styled data table */
    .clinical-table-wrapper {{
        overflow-x: auto;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 12px rgba(0,0,0,0.01);
    }}
    
    /* Fade-in Animations */
    .fade-in {{
        animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
    }}
    
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    /* Diagnosis warning/success banners */
    .diag-banner {{
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.25rem;
        border-left: 5px solid;
    }}
    
    .diag-banner.leukemia {{
        background-color: rgba(239, 83, 80, 0.08);
        border-left-color: var(--accent-color);
        color: #b71c1c;
    }}
    
    .diag-banner.normal {{
        background-color: rgba(16, 185, 129, 0.08);
        border-left-color: var(--success-color);
        color: #064e3b;
    }}
    
    .diag-title {{
        font-weight: 800;
        font-size: 1.2rem;
        margin: 0;
    }}
    
    .diag-desc {{
        font-size: 0.9rem;
        margin: 0.25rem 0 0 0;
        opacity: 0.9;
    }}
    
    .footer-section {{
        text-align: center;
        padding-top: 3rem;
        padding-bottom: 2rem;
        border-top: 1px solid var(--border-color);
        color: var(--muted-color);
        font-size: 0.85rem;
        margin-top: 4rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
