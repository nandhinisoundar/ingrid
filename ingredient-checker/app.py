"""Ingrid Streamlit interface."""

import tempfile

import streamlit as st
from PIL import Image

from barcode_detector import extract_text
from chain import analyse_label

st.set_page_config(page_title="Ingrid", page_icon="▣", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#15252b; --muted:#607177; --green:#059669; --teal:#0d9488; --line:#dce8e5; }
    .stApp { background:radial-gradient(circle at 8% 8%,#d9f8e8 0,transparent 26%),linear-gradient(135deg,#f5fffb,#f8fafc 52%,#effbf8); color:var(--ink); font-family:'DM Sans',sans-serif; }
    .block-container { max-width:950px; padding-top:1.2rem; }
    h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; color:var(--ink) !important; }
    .nav { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--line); padding:.2rem 0 1rem; }
    .brand { display:flex; align-items:center; gap:.75rem; }.mark { display:grid; place-items:center; width:42px; height:42px; border-radius:14px; color:white; font-size:22px; font-weight:800; background:linear-gradient(135deg,var(--green),var(--teal)); box-shadow:0 8px 20px #05966933; }
    .name { font:700 1.35rem 'Space Grotesk'; line-height:1; }.tag { color:#047857; font-size:.72rem; font-weight:700; margin-top:4px; }.pill { color:#065f46; background:#ecfdf5; border:1px solid #bbf7d0; border-radius:999px; padding:.45rem .8rem; font-size:.72rem; font-weight:700; }
    .hero { text-align:center; padding:3rem 1rem 2rem; }.kicker { display:inline-block; color:#065f46; background:#dcfce7; border:1px solid #bbf7d0; border-radius:999px; padding:.4rem .8rem; font-size:.72rem; font-weight:700; }.hero h1 { font-size:clamp(2.1rem,6vw,4rem) !important; line-height:1.05; margin:.9rem 0 .7rem; }.hero h1 span { color:var(--green); }.hero p { color:var(--muted); max-width:580px; margin:auto; line-height:1.6; }
    .card { background:#ffffffee; border:1px solid var(--line); border-radius:24px; padding:1.5rem; box-shadow:0 18px 45px #173d3214; margin-bottom:1.1rem; }.upload { text-align:center; }.label { color:#607177; font-size:.7rem; font-weight:800; text-transform:uppercase; letter-spacing:.12em; }.title { font:700 1.65rem 'Space Grotesk'; margin:.35rem 0; }.chip { display:inline-block; background:#f1f5f9; border:1px solid #e2e8f0; border-radius:9px; padding:.35rem .6rem; color:#475569; font:600 .72rem monospace; }.verdict { border-radius:22px; padding:1.25rem 1.4rem; border:1px solid #a7f3d0; background:linear-gradient(135deg,#ecfdf5,#f0fdfa); margin-bottom:1.1rem; }.verdict-title { color:#065f46; font:700 1.15rem 'Space Grotesk'; }.nutri { text-align:center; border-radius:22px; padding:1.25rem; background:#fff; border:1px solid var(--line); margin-bottom:1.1rem; }.grade { display:inline-grid; place-items:center; width:72px; height:72px; border-radius:20px; color:#fff; font:800 2.5rem 'Space Grotesk'; margin:.7rem; }.a{background:#059669}.b{background:#84cc16}.c{background:#f59e0b}.d{background:#f97316}.e{background:#dc2626}.unknown{background:#94a3b8}.ingredient { padding:.7rem 0; border-bottom:1px solid #edf3f1; }.stButton>button { border-radius:12px; font-weight:800; border:1px solid #b7e7d0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="nav"><div class="brand"><div class="mark">⌁</div><div><div class="name">Ingrid</div><div class="tag">Your ingredient, decoded</div></div></div><div class="pill">✓ Source traced</div></div>', unsafe_allow_html=True)
st.markdown('<section class="hero"><div class="kicker">Product intelligence, grounded in data</div><h1>Every ingredient,<br><span>decoded clearly.</span></h1><p>Scan a product barcode to reveal product details, ingredients, Nutri-Score, and an evidence-aware health verdict.</p></section>', unsafe_allow_html=True)

st.markdown('<div class="card upload"><div class="label">Scan a product</div><h3>Upload a barcode image</h3><p style="color:#607177">PNG, JPG, or JPEG. Make sure the barcode is visible and well lit.</p>', unsafe_allow_html=True)
photo = st.file_uploader("Choose barcode image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
label_text = ""
if photo:
    st.image(Image.open(photo), width=360)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(photo.getvalue())
        path = tmp.name
    with st.spinner("Reading barcode..."):
        scanned, confidence = extract_text(path)
    st.caption(f"Barcode confidence: {confidence:.0%}")
    label_text = st.text_input("Detected barcode", scanned)
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Decode product", type="primary", use_container_width=True):
    if not label_text.strip():
        st.error("Upload an image with a detectable barcode first.")
    else:
        with st.spinner("Retrieving product and assessing ingredients..."):
            result = analyse_label(label_text)
        product = result.get("product") or {}
        if product:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="label">{product.get("brands") or "Product record"}</div><div class="title">{product.get("product_name") or "Unknown product"}</div><span class="chip">▦ {product.get("barcode", label_text)}</span>', unsafe_allow_html=True)
            with st.expander("View full product details"):
                st.json(product)
            st.markdown('</div>', unsafe_allow_html=True)

        grade = str(result.get("nutriscore_grade") or "unknown").lower()
        grade_class = grade if grade in {"a", "b", "c", "d", "e"} else "unknown"
        display_grade = grade.upper() if grade != "unknown" else "?"
        st.markdown(f'<div class="nutri"><div class="label">Nutri-Score</div><div class="grade {grade_class}">{display_grade}</div><div style="color:#607177;font-size:.82rem">Product-level nutritional quality signal</div></div>', unsafe_allow_html=True)

        nutrition = result.get("nutrition", {})
        st.markdown('<div class="card"><div class="label">Nutrition per 100g</div>', unsafe_allow_html=True)
        if nutrition:
            rows = [
                {
                    "Nutrient": key.replace("-", " ").title(),
                    "Value": item.get("value", "Not available"),
                    "Unit": item.get("unit", ""),
                }
                for key, item in sorted(nutrition.items())
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.info("No nutrition values were stored for this product.")
        st.markdown('</div>', unsafe_allow_html=True)

        additives_highlights = result.get("additives_highlights", {})
        st.markdown('<div class="card"><div class="label">Additives & highlights</div>', unsafe_allow_html=True)
        additives = additives_highlights.get("additives", [])
        highlights = additives_highlights.get("highlights", [])
        if additives:
            st.caption("Additives")
            st.write(" · ".join(additives))
        else:
            st.caption("No additive E-numbers were stored for this product.")
        if highlights:
            st.caption("Ingredient highlights")
            st.write(" · ".join(highlights))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="label">Product ingredients</div>', unsafe_allow_html=True)
        if result["ingredients"]:
            st.write(", ".join(dict.fromkeys(result["ingredients"])))
        else:
            st.info("No ingredient list was stored for this product.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="verdict"><div class="label">Ingrid verdict</div><div class="verdict-title">LLM health assessment</div>', unsafe_allow_html=True)
        st.write(result.get("explanation") or "No health assessment was returned.")
        st.markdown('</div>', unsafe_allow_html=True)

