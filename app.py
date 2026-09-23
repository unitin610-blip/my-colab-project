import ipaddress
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopSafe AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).parent / "url_deploy_model.pkl"
FEATURES_PATH = Path(__file__).parent / "url_deploy_features.pkl"


# ─────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(99,102,241,.12), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(14,165,233,.10), transparent 24%),
            #070a12;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.4rem;
        padding-bottom: 3rem;
    }

    .hero {
        text-align: center;
        padding: 0.4rem 0 1.2rem;
    }

    .badge {
        display: inline-block;
        padding: .35rem .7rem;
        border: 1px solid rgba(148,163,184,.22);
        border-radius: 999px;
        background: rgba(15,23,42,.65);
        color: #cbd5e1;
        font-size: .78rem;
        font-weight: 600;
        letter-spacing: .04em;
    }

    .hero h1 {
        margin: .9rem 0 .35rem;
        font-size: clamp(2.2rem, 5vw, 4rem);
        line-height: 1;
        font-weight: 800;
        letter-spacing: -.045em;
    }

    .hero h1 span {
        background: linear-gradient(90deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #94a3b8;
        max-width: 720px;
        margin: 0 auto;
        line-height: 1.65;
    }

    .input-card, .result-card, .info-card {
        background: rgba(15,23,42,.72);
        border: 1px solid rgba(148,163,184,.13);
        border-radius: 22px;
        padding: 1.15rem;
        box-shadow: 0 18px 50px rgba(0,0,0,.18);
        backdrop-filter: blur(14px);
    }

    .section-label {
        color: #cbd5e1;
        font-weight: 700;
        font-size: .85rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: .55rem;
    }

    .url-preview {
        margin-top: .7rem;
        color: #94a3b8;
        font-size: .8rem;
        overflow-wrap: anywhere;
    }

    .risk-banner {
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin: .85rem 0 1rem;
        border: 1px solid;
    }

    .risk-title {
        font-weight: 800;
        font-size: 1.2rem;
        margin-bottom: .2rem;
    }

    .risk-copy {
        color: #cbd5e1;
        font-size: .9rem;
    }

    .score-card {
        background: rgba(2,6,23,.55);
        border: 1px solid rgba(148,163,184,.11);
        border-radius: 18px;
        padding: 1rem;
        min-height: 160px;
    }

    .score-kicker {
        color: #94a3b8;
        font-size: .78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .score-value {
        font-size: 2.3rem;
        font-weight: 800;
        margin: .35rem 0;
    }

    .score-caption {
        color: #94a3b8;
        font-size: .82rem;
    }

    .signal-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: .55rem;
    }

    [data-testid="stForm"] {
        border: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(148,163,184,.13) !important;
        border-radius: 20px !important;
        background: rgba(15,23,42,.62) !important;
    }

    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: .78rem;
        margin-top: 2rem;
    }

    div[data-testid="stTextInput"] input {
        background: rgba(2,6,23,.70) !important;
        border: 1px solid rgba(148,163,184,.20) !important;
        border-radius: 14px !important;
        color: #f8fafc !important;
        min-height: 52px !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: rgba(129,140,248,.75) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,.13) !important;
    }

    div.stButton > button {
        width: 100%;
        min-height: 50px;
        border: 0;
        border-radius: 14px;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #0ea5e9);
        color: white;
    }

    div.stButton > button:hover {
        filter: brightness(1.08);
        color: white;
        border: 0;
    }

    .tiny {
        color: #64748b;
        font-size: .74rem;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature list: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features


# ─────────────────────────────────────────────────────────────
# Same feature extraction used by the notebook
# ─────────────────────────────────────────────────────────────
def extract_url_features(url: str) -> dict:
    url = url.strip()

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)
    domain = parsed.netloc.split("@")[-1].split(":")[0]

    if not domain:
        raise ValueError("That doesn't look like a valid URL/domain.")

    url_length = len(url)
    domain_length = len(domain)

    try:
        ipaddress.ip_address(domain)
        is_domain_ip = 1
    except ValueError:
        is_domain_ip = 0

    domain_parts = domain.split(".")
    no_of_subdomain = max(len(domain_parts) - 2, 0)

    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)

    no_of_equals = url.count("=")
    no_of_qmark = url.count("?")
    no_of_ampersand = url.count("&")

    special_chars = sum(
        not c.isalnum() and c not in [".", "/", "=", "?", "&", "-", "_"]
        for c in url
    )

    obfuscated_chars = url.count("%")
    has_obfuscation = int(obfuscated_chars > 0)

    letter_ratio = letters / url_length if url_length else 0
    digit_ratio = digits / url_length if url_length else 0
    special_ratio = special_chars / url_length if url_length else 0
    obfuscation_ratio = obfuscated_chars / url_length if url_length else 0

    is_https = int(parsed.scheme.lower() == "https")

    return {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": obfuscated_chars,
        "ObfuscationRatio": obfuscation_ratio,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letter_ratio,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digit_ratio,
        "NoOfEqualsInURL": no_of_equals,
        "NoOfQMarkInURL": no_of_qmark,
        "NoOfAmpersandInURL": no_of_ampersand,
        "NoOfOtherSpecialCharsInURL": special_chars,
        "SpacialCharRatioInURL": special_ratio,
        "IsHTTPS": is_https,
    }


def get_risk_level(phishing_probability: float) -> str:
    if phishing_probability >= 0.70:
        return "HIGH RISK"
    if phishing_probability >= 0.40:
        return "MEDIUM RISK"
    return "LOW RISK"


def analyze_url(url: str, model, feature_order):
    features = extract_url_features(url)

    vector = pd.DataFrame(
        [[features[f] for f in feature_order]],
        columns=feature_order,
    )

    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    # The notebook trains with 0 = phishing and 1 = legitimate.
    class_probabilities = {
        int(cls): float(prob)
        for cls, prob in zip(model.classes_, probabilities)
    }

    phishing_probability = class_probabilities.get(0, 0.0)
    legitimate_probability = class_probabilities.get(1, 0.0)

    # Fall back safely if a differently encoded classifier is ever supplied.
    if len(class_probabilities) != 2:
        phishing_probability = float(probabilities[0])
        legitimate_probability = float(probabilities[-1])

    importances = getattr(model, "feature_importances_", None)
    top_features = pd.DataFrame()

    if importances is not None:
        top_features = pd.DataFrame(
            {
                "Feature": vector.columns,
                "Value": vector.iloc[0].values,
                "Importance": importances,
            }
        ).sort_values("Importance", ascending=False).head(5)

    return {
        "prediction": "LEGITIMATE" if int(prediction) == 1 else "PHISHING",
        "phishing_probability": phishing_probability,
        "legitimate_probability": legitimate_probability,
        "risk_level": get_risk_level(phishing_probability),
        "features": features,
        "top_features": top_features,
    }


def risk_style(risk: str):
    if risk == "HIGH RISK":
        return "#fb7185", "rgba(127,29,29,.24)", "This URL has a high phishing likelihood."
    if risk == "MEDIUM RISK":
        return "#fbbf24", "rgba(120,53,15,.24)", "This URL falls into the model's medium-risk band."
    return "#34d399", "rgba(6,78,59,.24)", "This URL falls into the model's low-risk band."


# ─────────────────────────────────────────────────────────────
# Clean UI
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="badge">AI-POWERED URL SECURITY</div>
        <h1>Shop<span>Safe</span> AI</h1>
        <p>Check a website URL for phishing risk.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Input card
with st.container(border=True):
    st.markdown("**Analyze a website**")

    with st.form("url_form", clear_on_submit=False):
        url = st.text_input(
            "Website URL",
            value="https://www.google.com",
            placeholder="example.com or https://example.com/login",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🔍 Analyze URL")


if submitted:
    if not url.strip():
        st.warning("Enter a URL first.")
        st.stop()

    try:
        model, feature_order = load_artifacts()
        result = analyze_url(url, model, feature_order)
    except FileNotFoundError as exc:
        st.error(
            f"{exc}\n\n"
            "Put `url_deploy_model.pkl` and `url_deploy_features.pkl` "
            "in the same folder as `app.py`."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Could not analyze this URL: {exc}")
        st.stop()

    risk = result["risk_level"]
    risk_color, risk_bg, _ = risk_style(risk)
    prediction = result["prediction"]
    prediction_color = "#34d399" if prediction == "LEGITIMATE" else "#fb7185"

    phishing_pct = result["phishing_probability"] * 100
    legitimate_pct = result["legitimate_probability"] * 100

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Main result card
    with st.container(border=True):
        st.markdown("### Analysis result")

        st.markdown(
            f"""
            <div class="risk-banner" style="border-color:{risk_color}55;background:{risk_bg};">
                <div class="risk-title" style="color:{risk_color};">{risk}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="text-align:center; padding:.3rem 0 .8rem;">
                <div class="score-kicker">Classification</div>
                <div class="score-value" style="color:{prediction_color};">
                    {prediction}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns(2, gap="medium")

        with left:
            with st.container(border=True):
                st.markdown('<div class="score-kicker">Phishing likelihood</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="score-value" style="color:#fb7185;">{phishing_pct:.2f}%</div>',
                    unsafe_allow_html=True,
                )
                st.progress(min(max(result["phishing_probability"], 0.0), 1.0))

        with right:
            with st.container(border=True):
                st.markdown('<div class="score-kicker">Legitimate likelihood</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="score-value" style="color:#34d399;">{legitimate_pct:.2f}%</div>',
                    unsafe_allow_html=True,
                )
                st.progress(min(max(result["legitimate_probability"], 0.0), 1.0))

    st.markdown(
        '<div class="tiny" style="text-align:center;margin-top:1rem;">'
        'This score is based on URL characteristics only. The website is not opened or fetched.'
        '</div>',
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="footer-note">ShopSafe AI · URL phishing classifier</div>',
    unsafe_allow_html=True,
)
