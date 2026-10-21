# -*- coding: utf-8 -*-
"""
SISTEM INTEGRASI BUSINESS INTELLIGENCE & DATA MINING
UTS Business Intelligence / Data Mining — Universitas Timor (UNIMOR)
Penyusun : Theresa Ivanna Oki (51230079) — Kelas VII A / Teknologi Informasi

Cara menjalankan:
    pip install streamlit pandas numpy plotly scikit-learn mlxtend openpyxl
    streamlit run UTS.py
"""
import io
import os
import re
import glob
import inspect
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="UTS BI & Data Mining - Theresa Ivanna Oki",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# ============================================================
# KONSTANTA
# ============================================================
MAX_QTY = 50            # batas kuantitas wajar per baris item (data normal: 1–6)
PRICE_TOL = 5           # harga dianggap tidak wajar jika > 5x atau < 1/5 harga referensi
PALETTE = ["#2563eb", "#06b6d4", "#8b5cf6", "#f59e0b", "#ec4899", "#10b981", "#f97316", "#64748b"]
SEG_COLORS = {
    "Loyal Customers": "#2563eb",
    "Potential Customers": "#06b6d4",
    "New Customers": "#10b981",
    "Churn Risk": "#f43f5e",
}
PAY_MAP = {"tunai": "Tunai", "cash": "Tunai", "debit": "Debit", "transfer": "Transfer", "qris": "QRIS"}
MONTH_ID = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun",
            7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"}

COL_SYNONYMS = {
    "Quantity": ["quantity", "qty", "jumlah", "jumlah_barang"],
    "Unit_Price": ["unit_price", "price", "harga", "harga_satuan", "unitprice"],
    "Transaction_ID": ["transaction_id", "id_transaksi", "notransaksi", "invoice"],
    "Customer_ID": ["customer_id", "id_pelanggan", "customer", "id_customer"],
    "Product_ID": ["product_id", "id_produk", "kode_produk"],
    "Product_Name": ["product_name", "nama_produk", "product", "produk"],
    "Transaction_Date": ["transaction_date", "tanggal", "date", "tgl_transaksi"],
    "Line_Total": ["line_total", "total", "total_harga", "total_pembelian"],
    "Category": ["category", "kategori"],
    "Discount_Rate": ["discount_rate", "diskon"],
    "Payment_Method": ["payment_method", "metode_pembayaran"],
    "Customer_Type": ["customer_type", "tipe_pelanggan"],
    "Region": ["region", "wilayah"],
}

# ============================================================
# CSS KUSTOM
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@600;700&display=swap');

html, body, .stApp, .stMarkdown, p, li, label, h1, h2, h3, h4, h5, h6, button, input, textarea, td, th {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at 8% 0%, #dbeafe 0%, transparent 38%),
        radial-gradient(circle at 95% 8%, #cffafe 0%, transparent 32%),
        radial-gradient(circle at 90% 100%, #ede9fe 0%, transparent 40%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    background-attachment: fixed;
}
.stApp .stMarkdown p, .stApp .stMarkdown li { color: #334155; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #1e3a8a; letter-spacing: -0.01em; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
footer { visibility: hidden; }

/* ---------- HERO LOGIN ---------- */
.hero {
    position: relative; overflow: hidden; text-align: center; color: #fff;
    background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 38%, #2563eb 72%, #06b6d4 100%);
    background-size: 220% 220%; animation: heroShift 14s ease infinite;
    padding: 54px 30px 42px; border-radius: 28px; margin-bottom: 30px;
    box-shadow: 0 30px 60px -20px rgba(30,58,138,.55);
}
@keyframes heroShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.hero::before, .hero::after { content:''; position:absolute; border-radius:50%; background: rgba(255,255,255,.08); }
.hero::before { width: 340px; height: 340px; top: -140px; left: -90px; }
.hero::after  { width: 260px; height: 260px; bottom: -120px; right: -60px; background: rgba(6,182,212,.22); }
.hero .eyebrow { text-transform: uppercase; letter-spacing: .22em; font-size: .78rem; font-weight: 700; color: #93c5fd !important; margin: 0; }
.hero h1 { color:#fff !important; font-size: 2.3rem; font-weight: 800; line-height: 1.2; margin: 10px 0 12px; position: relative; }
.hero p.sub { color:#dbeafe !important; font-size: 1.02rem; font-weight: 500; margin: 0 auto; max-width: 780px; position: relative; }
.chips { display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-top:22px; position:relative; }
.chip {
    background: rgba(255,255,255,.14); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,.35); color:#fff !important;
    padding: 7px 16px; border-radius: 999px; font-size: .82rem; font-weight: 600; transition: all .25s;
}
.chip:hover { background: rgba(255,255,255,.28); transform: translateY(-2px); }

/* ---------- KARTU UMUM ---------- */
.card {
    background: rgba(255,255,255,.96); padding: 28px 30px; border-radius: 22px;
    border: 1px solid #dbeafe; box-shadow: 0 18px 40px -16px rgba(30,58,138,.18);
}
.card-title {
    font-size: 1.12rem; font-weight: 800; color:#1e3a8a; margin-bottom: 18px; padding-bottom: 12px;
    border-bottom: 3px solid transparent;
    border-image: linear-gradient(90deg,#2563eb,#06b6d4,transparent) 1;
}
.info-table { width:100%; border-collapse: collapse; }
.info-table td { padding: 11px 0; border-bottom: 1px dashed #e2e8f0; font-size: .93rem; }
.info-table td.l { color:#64748b; font-weight:500; width:40%; }
.info-table td.v { color:#0f172a; font-weight:700; text-align:right; }

/* ---------- TOPBAR HALAMAN ---------- */
.topbar {
    position: relative; overflow: hidden; display:flex; align-items:center; justify-content:space-between; gap:20px;
    background: linear-gradient(115deg, #1e3a8a 0%, #2563eb 60%, #06b6d4 120%);
    border-radius: 22px; padding: 26px 32px; margin-bottom: 26px; color:#fff;
    box-shadow: 0 22px 44px -18px rgba(37,99,235,.55);
}
.topbar::after { content:''; position:absolute; width:260px; height:260px; border-radius:50%; right:-70px; top:-110px; background:rgba(255,255,255,.1); }
.topbar .tag { display:inline-block; background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35);
    padding:4px 14px; border-radius:999px; font-size:.75rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
.topbar h2 { color:#fff !important; margin: 10px 0 4px; font-size: 1.55rem; font-weight: 800; }
.topbar p { color:#dbeafe !important; margin:0; font-size:.95rem; }
.topbar .weight { position:relative; z-index:1; text-align:center; background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.3);
    padding: 14px 22px; border-radius: 18px; min-width: 110px; }
.topbar .weight b { display:block; font-size:1.9rem; font-weight:800; font-family:'JetBrains Mono',monospace; color:#fff; line-height:1; }
.topbar .weight span { font-size:.7rem; text-transform:uppercase; letter-spacing:.08em; color:#bfdbfe; font-weight:700; }

/* ---------- JUDUL SEKSI ---------- */
.sec { display:flex; align-items:center; gap:14px; margin: 30px 0 16px; }
.sec .no {
    width:42px; height:42px; border-radius:13px; display:flex; align-items:center; justify-content:center; flex-shrink:0;
    background: linear-gradient(135deg,#2563eb,#06b6d4); color:#fff; font-weight:800; font-size:1.05rem;
    box-shadow: 0 10px 20px -8px rgba(37,99,235,.6);
}
.sec .tt { font-size:1.28rem; font-weight:800; color:#1e3a8a; line-height:1.2; }
.sec .ss { font-size:.86rem; color:#64748b; font-weight:500; margin-top:2px; }

/* ---------- KPI ---------- */
.kpi {
    display:flex; align-items:center; gap:14px; background:#fff; border-radius:18px; padding:18px 18px; margin-bottom:14px;
    border:1px solid #e2e8f0; border-left: 5px solid var(--c, #2563eb);
    box-shadow: 0 10px 26px -14px rgba(30,58,138,.28); transition: transform .2s, box-shadow .2s; min-height: 92px;
}
.kpi:hover { transform: translateY(-4px); box-shadow: 0 18px 34px -14px rgba(30,58,138,.35); }
.kpi-ico {
    width:50px; height:50px; border-radius:15px; display:flex; align-items:center; justify-content:center; font-size:1.5rem; flex-shrink:0;
    background: color-mix(in srgb, var(--c, #2563eb) 15%, white);
}
.kpi-val { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1.32rem; color:#0f172a; line-height:1.15; word-break: break-word; }
.kpi-lab { font-size:.7rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:.07em; margin-top:4px; }
.kpi-sub { font-size:.72rem; color:#94a3b8; margin-top:2px; font-weight:500; }

/* ---------- CALLOUT ---------- */
.callout { border-radius:16px; padding:18px 22px; margin: 14px 0; border:1px solid; border-left-width:6px; }
.callout .ct { font-weight:800; font-size:1rem; margin-bottom:8px; }
.callout, .callout p, .callout li, .callout td, .callout th, .callout b { color:#1e293b !important; }
.callout ul, .callout ol { margin: 6px 0 0 0; padding-left: 20px; }
.callout li { margin-bottom: 5px; line-height: 1.55; }
.callout.info    { background:#eff6ff; border-color:#bfdbfe; border-left-color:#2563eb; }
.callout.info .ct { color:#1d4ed8 !important; }
.callout.ok      { background:#ecfdf5; border-color:#a7f3d0; border-left-color:#10b981; }
.callout.ok .ct  { color:#047857 !important; }
.callout.warn    { background:#fffbeb; border-color:#fde68a; border-left-color:#f59e0b; }
.callout.warn .ct{ color:#b45309 !important; }
.callout.risk    { background:#fff1f2; border-color:#fecdd3; border-left-color:#f43f5e; }
.callout.risk .ct{ color:#be123c !important; }
.callout.violet  { background:#f5f3ff; border-color:#ddd6fe; border-left-color:#8b5cf6; }
.callout.violet .ct { color:#6d28d9 !important; }

/* ---------- KARTU PERTANYAAN / REKOMENDASI ---------- */
.qcard { background:#fff; border:1px solid #e2e8f0; border-radius:18px; padding:20px; height:100%; box-shadow:0 10px 26px -16px rgba(30,58,138,.25); }
.qcard .qn { width:34px; height:34px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800; color:#fff;
    background:linear-gradient(135deg,#2563eb,#8b5cf6); margin-bottom:10px; }
.qcard, .qcard p, .qcard li { color:#334155 !important; font-size:.93rem; line-height:1.55; }
.rec { background:#fff; border:1px solid #e2e8f0; border-radius:20px; padding:22px 26px; margin: 14px 0; box-shadow:0 12px 30px -16px rgba(30,58,138,.28); }
.rec .rh { display:flex; align-items:center; gap:12px; margin-bottom:12px; }
.rec .rp { background:linear-gradient(135deg,#2563eb,#06b6d4); color:#fff; font-weight:800; padding:5px 14px; border-radius:999px; font-size:.78rem; letter-spacing:.05em; }
.rec .rt { font-weight:800; color:#1e3a8a; font-size:1.1rem; }
.rec table { width:100%; border-collapse:collapse; }
.rec td { padding:8px 4px; vertical-align:top; border-bottom:1px dashed #e2e8f0; font-size:.9rem; color:#334155; }
.rec td.k { width:22%; font-weight:700; color:#475569; }
.three { display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin: 10px 0 6px; }
.three > div { border-radius:18px; padding:18px 20px; border:1px solid; }
.three .t1 { background:#ecfdf5; border-color:#a7f3d0; }
.three .t2 { background:#fffbeb; border-color:#fde68a; }
.three .t3 { background:#f1f5f9; border-color:#cbd5e1; }
.three b, .three p, .three li { color:#1e293b !important; font-size:.9rem; }
@media (max-width: 900px) { .three { grid-template-columns: 1fr; } .topbar { flex-direction: column; align-items:flex-start; } }

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] { background: linear-gradient(185deg, #0b1220 0%, #172554 55%, #1e40af 130%); }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] li { color:#e2e8f0 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: rgba(255,255,255,.05); border:1px solid rgba(147,197,253,.18); padding: 11px 14px; border-radius: 13px;
    margin-bottom: 7px; width:100%; transition: all .2s; cursor:pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(59,130,246,.2); border-color:#60a5fa; transform: translateX(4px); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, rgba(37,99,235,.65), rgba(6,182,212,.22)); border-color:#60a5fa; border-left: 4px solid #22d3ee;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p { font-weight:600; font-size:.88rem; }
.profile {
    text-align:center; padding:18px 14px; border-radius:18px; margin-bottom: 14px;
    background: linear-gradient(145deg, rgba(255,255,255,.1), rgba(34,211,238,.1)); border:1px solid rgba(147,197,253,.28);
}
.avatar { width:84px; height:84px; margin:0 auto 10px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg,#2563eb,#22d3ee); color:#fff; font-size:2.2rem; font-weight:800; box-shadow:0 10px 24px rgba(34,211,238,.35); }
.profile .nm { font-weight:800; color:#fff !important; font-size:1rem; margin:0; }
.profile .mt { font-size:.8rem; color:#bfdbfe !important; margin:3px 0 0; }
.profile .un { font-size:.72rem; letter-spacing:.14em; color:#fde047 !important; font-weight:800; margin-top:8px; }
.prog { display:flex; align-items:center; gap:10px; font-size:.82rem; padding:5px 2px; color:#cbd5e1; }
.prog .dot { width:20px; height:20px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:.7rem; font-weight:800;
    background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.25); color:#94a3b8; flex-shrink:0; }
.prog.done { color:#fff; font-weight:600; }
.prog.done .dot { background:#10b981; border-color:#10b981; color:#fff; }

/* ---------- TOMBOL ---------- */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 60%, #0891b2 130%); color:#fff !important; font-weight:700;
    border:none; border-radius:12px; padding:.65rem 1.3rem; width:100%;
    box-shadow: 0 10px 22px -10px rgba(37,99,235,.7); transition: all .2s;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px); box-shadow: 0 16px 28px -10px rgba(37,99,235,.8); color:#fff !important; border:none;
}
.stButton > button p, .stFormSubmitButton > button p, .stDownloadButton > button p { color:#fff !important; font-weight:700; }

/* ---------- INPUT / TABS / DLL ---------- */
.stTextInput input { border-radius: 12px; border:1px solid #cbd5e1; background:#fff; color:#0f172a; }
.stTextInput input:focus { border-color:#2563eb; box-shadow:0 0 0 3px rgba(37,99,235,.16); }
div[data-testid="stFileUploader"] section { border:2px dashed #93c5fd; border-radius:18px; background:rgba(239,246,255,.8); }
.stTabs [data-baseweb="tab-list"] { gap:6px; background:#e0e7ff55; padding:6px; border-radius:16px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { border-radius:12px; padding:9px 18px; font-weight:700; color:#475569; height:auto; }
.stTabs [aria-selected="true"] { background:#fff; color:#1d4ed8; box-shadow:0 6px 14px -6px rgba(37,99,235,.4); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
div[data-testid="stPlotlyChart"] { background:#fff; border-radius:18px; border:1px solid #e2e8f0; padding:8px 6px; box-shadow:0 10px 26px -16px rgba(30,58,138,.28); }
div[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid #dbeafe; box-shadow:0 8px 22px -14px rgba(30,58,138,.3); }
div[data-testid="stExpander"] { background:#fff; border-radius:16px; border:1px solid #e2e8f0; }
div[data-testid="stSlider"] label p, div[data-testid="stSelectbox"] label p, div[data-testid="stMultiSelect"] label p,
div[data-testid="stDateInput"] label p, div[data-testid="stNumberInput"] label p { color:#1e293b !important; font-weight:700; }
.footer { text-align:center; margin-top:36px; color:#64748b; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER TAMPILAN
# ============================================================
def H(html: str):
    """Render HTML satu baris (menghindari markdown menganggap indentasi sebagai code block)."""
    st.markdown(" ".join(l.strip() for l in html.strip().splitlines()), unsafe_allow_html=True)


def fmt_id(x, d=0):
    """Format angka gaya Indonesia: 1.234.567,89"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def rp(x):
    return "Rp " + fmt_id(x)


def kpi(icon, label, value, color="#2563eb", sub=""):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (f'<div class="kpi" style="--c:{color}"><div class="kpi-ico">{icon}</div>'
            f'<div><div class="kpi-val">{value}</div><div class="kpi-lab">{label}</div>{sub_html}</div></div>')


def kpi_row(items, per_row=None):
    per_row = per_row or len(items)
    for start in range(0, len(items), per_row):
        chunk = items[start:start + per_row]
        cols = st.columns(len(chunk))
        for c, it in zip(cols, chunk):
            c.markdown(kpi(*it), unsafe_allow_html=True)


def topbar(tag, title, desc, weight):
    H(f'<div class="topbar"><div style="position:relative;z-index:1"><span class="tag">{tag}</span>'
      f'<h2>{title}</h2><p>{desc}</p></div><div class="weight"><b>{weight}</b><span>Bobot Nilai</span></div></div>')


def section(no, title, sub=""):
    sub_html = f'<div class="ss">{sub}</div>' if sub else ""
    H(f'<div class="sec"><div class="no">{no}</div><div><div class="tt">{title}</div>{sub_html}</div></div>')


def callout(title, body, kind="info", icon="💡"):
    H(f'<div class="callout {kind}"><div class="ct">{icon} {title}</div>{body}</div>')


def _ver():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (1, 0)


NEW_WIDTH_API = _ver() >= (1, 50)


def show_plot(fig, key=None):
    if NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def style_df(df):
    fmts = {}
    for col in df.columns:
        lc = str(col).lower()
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            fmts[col] = lambda v: "-" if pd.isna(v) else v.strftime("%Y-%m-%d %H:%M")
        elif pd.api.types.is_bool_dtype(s):
            continue
        elif pd.api.types.is_numeric_dtype(s):
            if any(k in lc for k in ["support", "confidence", "lift", "leverage"]):
                fmts[col] = lambda v: fmt_id(v, 3)
            elif lc.startswith("%") or "persen" in lc or "(%)" in lc or "rasio" in lc:
                fmts[col] = lambda v: fmt_id(v, 1)
            elif pd.api.types.is_integer_dtype(s):
                fmts[col] = lambda v: fmt_id(v, 0)
            else:
                vals = s.dropna()
                is_int_like = len(vals) > 0 and np.allclose(vals, np.round(vals))
                fmts[col] = (lambda v: fmt_id(v, 0)) if is_int_like else (lambda v: fmt_id(v, 2))
    return df.style.format(fmts, na_rep="-")


def show_df(df, height=None, fmt=True, max_rows=None):
    d = df.copy()
    if max_rows is not None and len(d) > max_rows:
        d = d.head(max_rows)
    d.index = range(1, len(d) + 1)
    obj = style_df(d) if fmt else d
    kw = {"height": height} if height else {}
    if NEW_WIDTH_API:
        st.dataframe(obj, width="stretch", **kw)
    else:
        st.dataframe(obj, use_container_width=True, **kw)


def style_fig(fig, title=None, height=380, legend=True):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>" if title else None, x=0.01, font=dict(size=15, color="#1e3a8a")),
        height=height, margin=dict(l=12, r=12, t=52 if title else 14, b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, Segoe UI, Arial", size=12, color="#334155"),
        separators=",.", showlegend=legend, colorway=PALETTE,
        legend=dict(orientation="h", y=-0.18, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0f172a", font_color="#fff", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor="#cbd5e1", zeroline=False)
    fig.update_yaxes(gridcolor="#e2e8f0", zeroline=False, linecolor="rgba(0,0,0,0)")
    return fig


BLUE_SCALE = [[0, "#bfdbfe"], [0.5, "#3b82f6"], [1, "#1e3a8a"]]


def hbar(series, title, xlabel="", height=380, money=True):
    s = series.sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=s.values, y=s.index.astype(str), orientation="h",
        marker=dict(color=s.values, colorscale=BLUE_SCALE, line=dict(width=0)),
        hovertemplate=("Rp %{x:,.0f}" if money else "%{x:,.0f}") + "<extra>%{y}</extra>",
    ))
    style_fig(fig, title, height, legend=False)
    fig.update_xaxes(title=xlabel, showgrid=True, gridcolor="#e2e8f0")
    fig.update_yaxes(showgrid=False)
    return fig


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# ============================================================
# FUNGSI DATA: MEMUAT & AUDIT
# ============================================================
def standardize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for col in df.columns:
        key = col.lower().replace(" ", "_")
        for canon, syns in COL_SYNONYMS.items():
            if key in syns and col != canon and canon not in df.columns:
                rename[col] = canon
    return df.rename(columns=rename)


def read_workbook(src):
    xls = pd.ExcelFile(src)
    names = xls.sheet_names
    target = "Data_Raw" if "Data_Raw" in names else next((n for n in names if "petunjuk" not in n.lower()), names[0])
    raw = standardize_columns(pd.read_excel(xls, sheet_name=target))
    catalog = None
    if "Data_Produk" in names:
        catalog = standardize_columns(pd.read_excel(xls, sheet_name="Data_Produk"))
    return raw, catalog, names, target


def _clean_text(v):
    if isinstance(v, str):
        v = re.sub(r"\s+", " ", v).strip()
        return v if v else np.nan
    return v


def _norm_category(v):
    v = _clean_text(v)
    if isinstance(v, str):
        return "ATK" if v.upper() == "ATK" else v.title()
    return v


def _norm_payment(v):
    v = _clean_text(v)
    if isinstance(v, str):
        return PAY_MAP.get(v.lower(), v)
    return v


def _mode(s):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else np.nan


def price_reference(df, catalog=None):
    """Harga acuan per baris: katalog Data_Produk bila ada, jika tidak median harga positif per produk."""
    name = df["Product_Name"]
    ref = pd.Series(np.nan, index=df.index)
    if catalog is not None and {"Product_Name", "Reference_Unit_Price_Rp"} <= set(catalog.columns):
        cmap = catalog.drop_duplicates("Product_Name").set_index("Product_Name")["Reference_Unit_Price_Rp"]
        ref = name.map(cmap).astype(float)
    pos = pd.to_numeric(df["Unit_Price"], errors="coerce")
    pos = pos.where(pos > 0)
    med = pos.groupby(name).transform("median")
    return ref.fillna(med)


def invalid_masks(df, catalog=None):
    q = pd.to_numeric(df["Quantity"], errors="coerce")
    p = pd.to_numeric(df["Unit_Price"], errors="coerce")
    ref = price_reference(df, catalog)
    q_bad = q.isna() | (q <= 0) | (q > MAX_QTY)
    p_bad = p.isna() | (p <= 0) | (p > PRICE_TOL * ref) | (p < ref / PRICE_TOL)
    return q_bad, p_bad


def quality_metrics(df, catalog=None):
    d = df.copy()
    for c in ["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    data_cols = [c for c in d.columns if c not in ("Catatan_Perubahan", "Basket_Size", "Bulan")]
    q_bad, p_bad = invalid_masks(d, catalog)
    calc = d["Quantity"] * d["Unit_Price"] * (1 - d["Discount_Rate"].fillna(0))
    mism = calc.notna() & d["Line_Total"].notna() & ((calc - d["Line_Total"]).abs() > 1)
    cat_bad = d["Category"].notna() & (d["Category"].map(_norm_category) != d["Category"])
    pay_bad = d["Payment_Method"].notna() & (d["Payment_Method"].map(_norm_payment) != d["Payment_Method"])
    return {
        "Jumlah baris": len(d),
        "Jumlah kolom (data asli)": len(data_cols),
        "Total sel kosong": int(d[data_cols].isna().sum().sum()),
        "Baris dengan nilai kosong": int(d[data_cols].isna().any(axis=1).sum()),
        "Baris duplikat penuh": int(d.duplicated(subset=data_cols).sum()),
        "Baris kategori tidak konsisten": int(cat_bad.sum()),
        "Baris metode bayar tidak konsisten": int(pay_bad.sum()),
        "Quantity tidak valid (kosong/≤0/>50)": int(q_bad.sum()),
        "Unit_Price tidak wajar (kosong/≤0/ekstrem)": int(p_bad.sum()),
        "Line_Total tidak konsisten dengan rumus": int(mism.sum()),
        "Transaksi unik": int(d["Transaction_ID"].nunique()),
        "Pelanggan unik": int(d["Customer_ID"].nunique()),
        "Total penjualan (Rp)": float(d["Line_Total"].sum()),
    }


# ============================================================
# FUNGSI DATA: PREPROCESSING LENGKAP
# ============================================================
def clean_data(raw, catalog=None):
    """Mengembalikan (data_bersih, log_keputusan). Data_Raw TIDAK ditimpa."""
    log = []

    def add(tahap, masalah, jumlah, keputusan, alasan):
        log.append({"Tahap": tahap, "Masalah": masalah, "Jumlah Ditemukan": int(jumlah),
                    "Keputusan / Tindakan": keputusan, "Alasan": alasan})

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    n0 = len(df)
    df["Catatan_Perubahan"] = ""

    def note(mask, text):
        df.loc[mask, "Catatan_Perubahan"] = df.loc[mask, "Catatan_Perubahan"] + text + "; "

    # 1. Standardisasi teks
    for col, fn, label in [("Category", _norm_category, "Kategori"), ("Payment_Method", _norm_payment, "Metode pembayaran")]:
        if col in df.columns:
            before = df[col].copy()
            df[col] = before.map(fn)
            ch = before.notna() & (before.fillna("§") != df[col].fillna("§"))
            note(ch, f"{label} distandarkan")
            add("1. Standardisasi teks", f"Inkonsistensi huruf/spasi pada {label}", ch.sum(),
                "Diseragamkan (hapus spasi ganda, Title Case; ATK & QRIS tetap kapital)",
                "Varian seperti 'sembako'/'SEMBAKO' akan dihitung sebagai kategori berbeda jika dibiarkan")
    for col in ["Transaction_ID", "Customer_ID", "Product_ID", "Product_Name", "Region", "Customer_Type"]:
        if col in df.columns:
            df[col] = df[col].map(_clean_text)

    # 2. Konversi tipe data
    for c in ["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]:
        b = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().sum() > b:
            add("2. Transformasi tipe", f"Nilai non-numerik pada {c}", df[c].isna().sum() - b,
                "Dikonversi menjadi NaN lalu ditangani pada langkah berikutnya", "Memastikan kolom numerik dapat dihitung")
    b = df["Transaction_Date"].isna().sum()
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    add("2. Transformasi tipe", "Format tanggal-waktu (Transaction_Date → datetime)", len(df),
        "Seluruh kolom dikonversi ke tipe datetime; nilai tak terbaca → NaT",
        "Dibutuhkan untuk tren bulanan dan perhitungan Recency")

    # 3. Duplikat penuh
    data_cols = [c for c in df.columns if c != "Catatan_Perubahan"]
    dup = df.duplicated(subset=data_cols, keep="first")
    add("3. Duplikasi", "Baris duplikat penuh (semua kolom identik)", dup.sum(), "Dihapus (simpan kemunculan pertama)",
        "Duplikat menggandakan penjualan & frekuensi; tidak ada informasi baru")
    df = df[~dup].copy().reset_index(drop=True)

    # 4. Tanggal tidak valid
    bad_date = df["Transaction_Date"].isna()
    add("4. Data tidak valid", "Tanggal transaksi kosong/tidak terbaca", bad_date.sum(), "Dihapus",
        "Tanpa tanggal, baris tidak dapat dipakai untuk tren maupun Recency")
    df = df[~bad_date].copy().reset_index(drop=True)

    # 5. Missing value atribut transaksi (dipulihkan dari baris lain pada Transaction_ID yang sama)
    for col in ["Customer_ID", "Region", "Customer_Type"]:
        b = df[col].isna()
        fill = df.groupby("Transaction_ID")[col].transform("first")
        df[col] = df[col].fillna(fill)
        pulih = b & df[col].notna()
        note(pulih, f"{col} dipulihkan dari transaksi yang sama")
        sisa = df[col].isna().sum()
        add("5. Missing values", f"{col} kosong", b.sum(),
            f"Dipulihkan dari baris lain pada Transaction_ID yang sama: {int(pulih.sum())} baris; sisa {int(sisa)} "
            + ("ditandai 'Tidak Diketahui'" if col != "Customer_ID" else "dipertahankan untuk analisis penjualan tetapi dikeluarkan dari RFM"),
            "Satu nota memiliki satu pelanggan, wilayah, dan tipe pelanggan; imputasi ini bersifat deterministik, bukan tebakan")
    for col in ["Region", "Customer_Type", "Payment_Method"]:
        m = df[col].isna()
        if m.sum():
            df.loc[m, "Catatan_Perubahan"] += f"{col} diisi 'Tidak Diketahui'; "
            df[col] = df[col].fillna("Tidak Diketahui")
            add("5. Missing values", f"{col} kosong tidak dapat dipulihkan", m.sum(), "Diisi 'Tidak Diketahui'",
                "Menjaga baris (nilai transaksi tetap valid) tanpa mengarang nilai")

    # 6. Product_ID / Product_Name / Category
    b = df["Product_ID"].isna()
    id_map = df.dropna(subset=["Product_ID", "Product_Name"]).groupby("Product_Name")["Product_ID"].agg(_mode)
    df["Product_ID"] = df["Product_ID"].fillna(df["Product_Name"].map(id_map))
    note(b & df["Product_ID"].notna(), "Product_ID dipulihkan dari Product_Name")
    gagal = df["Product_ID"].isna()
    add("5. Missing values", "Product_ID kosong", b.sum(),
        f"Dipulihkan lewat pemetaan Product_Name→Product_ID: {int((b & ~gagal).sum())} baris; sisa {int(gagal.sum())} dihapus",
        "Satu nama produk selalu berpasangan dengan satu ID; produk kunci basket tidak boleh kosong")
    df = df[~gagal].copy().reset_index(drop=True)
    b = df["Category"].isna()
    if b.any():
        cat_map = df.dropna(subset=["Category"]).groupby("Product_ID")["Category"].agg(_mode)
        df["Category"] = df["Category"].fillna(df["Product_ID"].map(cat_map))
        add("5. Missing values", "Category kosong", b.sum(), "Dipulihkan dari kategori produk yang sama", "Satu produk = satu kategori")

    # 7. Diskon
    dm = df["Discount_Rate"].isna() | (df["Discount_Rate"] < 0) | (df["Discount_Rate"] > 1)
    if dm.any():
        df.loc[dm, "Discount_Rate"] = 0
        note(dm, "Diskon tidak valid → 0")
    add("6. Nilai tidak valid", "Discount_Rate kosong / di luar 0–1", dm.sum(), "Diganti 0 (tanpa diskon)",
        "Pilihan paling konservatif agar tidak menggelembungkan/mengecilkan nilai transaksi")

    # 8. Harga satuan
    df["Product_Name"] = df["Product_Name"].astype(object)
    ref = price_reference(df, catalog)
    p_bad = df["Unit_Price"].isna() | (df["Unit_Price"] <= 0) | (df["Unit_Price"] > PRICE_TOL * ref) | (df["Unit_Price"] < ref / PRICE_TOL)
    fixable = p_bad & ref.notna()
    df.loc[fixable, "Unit_Price"] = ref[fixable]
    note(fixable, "Unit_Price diimputasi dari harga referensi produk")
    add("6. Nilai tidak valid", "Unit_Price kosong, negatif/nol (mis. -100), atau ekstrem (mis. 999.999)", p_bad.sum(),
        f"Diimputasi dengan harga referensi produk ({'katalog Data_Produk' if catalog is not None else 'median harga valid produk'}): {int(fixable.sum())} baris",
        "Harga produk stabil; imputasi referensi lebih akurat daripada menghapus baris atau memakai rata-rata global")
    gagal = df["Unit_Price"].isna()
    df = df[~gagal].copy().reset_index(drop=True)

    # 9. Quantity
    q = df["Quantity"]
    q_bad = q.isna() | (q <= 0) | (q > MAX_QTY)
    q_calc = df["Line_Total"] / (df["Unit_Price"] * (1 - df["Discount_Rate"]))
    q_round = q_calc.round()
    pulih = q_bad & q_calc.notna() & np.isfinite(q_calc) & ((q_calc - q_round).abs() < 0.05) & q_round.between(1, MAX_QTY)
    df.loc[pulih, "Quantity"] = q_round[pulih]
    note(pulih, "Quantity dipulihkan dari Line_Total÷(harga×(1−diskon))")
    add("6. Nilai tidak valid", "Quantity kosong, ≤ 0, atau ekstrem (> 50, mis. 999)", q_bad.sum(),
        f"Dipulihkan dari Line_Total ÷ (Unit_Price × (1−diskon)) bila hasil bilangan bulat 1–{MAX_QTY}: {int(pulih.sum())} baris; sisa {int((q_bad & ~pulih).sum())} dihapus",
        "Line_Total pada baris bermasalah umumnya masih benar sehingga kuantitas asli dapat direkonstruksi; konteks bisnis: belanja ritel wajar 1–6 unit/item")
    df = df[~(q_bad & ~pulih)].copy().reset_index(drop=True)

    # 10. Line_Total
    calc = (df["Quantity"] * df["Unit_Price"] * (1 - df["Discount_Rate"])).round(0)
    mism = df["Line_Total"].isna() | ((df["Line_Total"] - calc).abs() > 1)
    df.loc[mism, "Line_Total"] = calc[mism]
    note(mism, "Line_Total dihitung ulang")
    add("6. Nilai tidak valid", "Line_Total kosong / tidak sama dengan Qty × Harga × (1−Diskon)", mism.sum(),
        "Dihitung ulang dengan rumus Quantity × Unit_Price × (1 − Discount_Rate)", "Menjamin total penjualan konsisten secara aritmetika")

    # 10b. Duplikat baru yang muncul setelah nilai diperbaiki
    cols_now = [c for c in df.columns if c != "Catatan_Perubahan"]
    dup2 = df.duplicated(subset=cols_now, keep="first")
    add("3. Duplikasi", "Duplikat baru setelah perbaikan nilai (baris identik pasca-imputasi)", dup2.sum(), "Dihapus (simpan kemunculan pertama)",
        "Baris yang tadinya berbeda karena nilai rusak menjadi identik setelah diperbaiki, sehingga merupakan pencatatan ganda")
    df = df[~dup2].copy().reset_index(drop=True)

    # 11. Feature dasar
    df["Basket_Size"] = df.groupby("Transaction_ID")["Product_ID"].transform("nunique")
    df["Bulan"] = df["Transaction_Date"].dt.to_period("M").astype(str)
    add("7. Feature engineering", "Pembentukan basket_size", df["Transaction_ID"].nunique(),
        "Basket_Size = jumlah Product_ID unik per Transaction_ID", "Dasar analisis keranjang belanja & KPI dashboard")
    dup_item = df.duplicated(["Transaction_ID", "Product_ID"]).sum()
    add("7. Feature engineering", "Produk sama muncul >1 baris pada satu transaksi", dup_item,
        "Dipertahankan sebagai baris item (nilai penjualan tetap dihitung); pada matriks Apriori dihitung sebagai 1 (biner)",
        "Apriori hanya memerlukan ada/tidaknya produk pada satu basket")

    df["Catatan_Perubahan"] = df["Catatan_Perubahan"].str.rstrip("; ")
    log_df = pd.DataFrame(log)
    return df, log_df


# ============================================================
# FUNGSI DATA MINING
# ============================================================
@st.cache_data(show_spinner=False)
def build_basket(df):
    return pd.crosstab(df["Transaction_ID"], df["Product_Name"]).gt(0)


def _assoc(freq, n_tx, min_conf):
    kw = dict(metric="confidence", min_threshold=min_conf)
    if "num_itemsets" in inspect.signature(association_rules).parameters:
        kw["num_itemsets"] = n_tx
    return association_rules(freq, **kw)


@st.cache_data(show_spinner=False)
def run_apriori(df, min_sup, min_conf):
    basket = build_basket(df)
    n_tx = len(basket)
    if n_tx == 0:
        return pd.DataFrame(), pd.DataFrame(), n_tx
    freq = apriori(basket, min_support=min_sup, use_colnames=True)
    if freq.empty:
        return freq, pd.DataFrame(), n_tx
    rules = _assoc(freq, n_tx, min_conf) if (freq["itemsets"].apply(len) > 1).any() else pd.DataFrame()
    if len(rules):
        rules = rules[rules["lift"] > 1].sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    return freq, rules, n_tx


def rules_table(rules, top=10):
    if rules is None or len(rules) == 0:
        return pd.DataFrame()
    t = rules.head(top).copy()
    out = pd.DataFrame({
        "Jika membeli (A)": t["antecedents"].apply(lambda x: " + ".join(sorted(x))),
        "Maka juga membeli (B)": t["consequents"].apply(lambda x: " + ".join(sorted(x))),
        "Support": t["support"], "Confidence": t["confidence"], "Lift": t["lift"],
    })
    return out.reset_index(drop=True)


def freq_table(freq):
    if freq is None or len(freq) == 0:
        return pd.DataFrame()
    t = freq.copy()
    t["Itemset"] = t["itemsets"].apply(lambda x: " + ".join(sorted(x)))
    t["Ukuran"] = t["itemsets"].apply(len)
    return t[["Itemset", "Ukuran", "support"]].rename(columns={"support": "Support"}).sort_values("Support", ascending=False).reset_index(drop=True)


def network_figure(rules, top=15):
    t = rules.head(top)
    nodes = sorted({i for s in t["antecedents"] for i in s} | {i for s in t["consequents"] for i in s})
    n = len(nodes)
    ang = {nd: 2 * np.pi * i / n for i, nd in enumerate(nodes)}
    pos = {nd: (np.cos(a), np.sin(a)) for nd, a in ang.items()}
    fig = go.Figure()
    lmin, lmax = t["lift"].min(), t["lift"].max()
    for _, r in t.iterrows():
        w = 1.5 + 5 * (r["lift"] - lmin) / (lmax - lmin + 1e-9)
        for a in r["antecedents"]:
            for c in r["consequents"]:
                x0, y0 = pos[a]; x1, y1 = pos[c]
                fig.add_trace(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                                         line=dict(width=w, color="rgba(37,99,235,.35)"), hoverinfo="skip", showlegend=False))
                fig.add_annotation(x=x1 * 0.93, y=y1 * 0.93, ax=x0 * 0.93, ay=y0 * 0.93, xref="x", yref="y", axref="x", ayref="y",
                                   showarrow=True, arrowhead=3, arrowsize=1.1, arrowwidth=1.2, arrowcolor="rgba(37,99,235,.55)")
    deg = {nd: 0 for nd in nodes}
    for _, r in t.iterrows():
        for i in list(r["antecedents"]) + list(r["consequents"]):
            deg[i] += 1
    fig.add_trace(go.Scatter(
        x=[pos[nd][0] for nd in nodes], y=[pos[nd][1] for nd in nodes], mode="markers+text", text=nodes,
        textposition=["top center" if pos[nd][1] >= 0 else "bottom center" for nd in nodes],
        marker=dict(size=[18 + 7 * deg[nd] for nd in nodes], color=[deg[nd] for nd in nodes], colorscale=BLUE_SCALE,
                    line=dict(color="#fff", width=2)),
        hovertext=[f"{nd}: terhubung di {deg[nd]} aturan" for nd in nodes], hoverinfo="text", showlegend=False,
    ))
    style_fig(fig, f"Network Graph Top {len(t)} Aturan Asosiasi (tebal garis = Lift)", 520, legend=False)
    fig.update_xaxes(visible=False, range=[-1.5, 1.5])
    fig.update_yaxes(visible=False, range=[-1.4, 1.4], scaleanchor="x")
    return fig


def lift_heatmap(rules, top=25):
    t = rules.head(top)
    single = t[(t["antecedents"].apply(len) == 1) & (t["consequents"].apply(len) == 1)].copy()
    if single.empty:
        return None
    single["A"] = single["antecedents"].apply(lambda x: list(x)[0])
    single["B"] = single["consequents"].apply(lambda x: list(x)[0])
    pv = single.pivot_table(index="A", columns="B", values="lift", aggfunc="max")
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale=BLUE_SCALE,
                               hovertemplate="Jika %{y} → %{x}<br>Lift: %{z:.2f}<extra></extra>", colorbar=dict(title="Lift")))
    style_fig(fig, "Heatmap Lift antar Produk (A → B)", 460, legend=False)
    fig.update_xaxes(tickangle=-40)
    return fig


@st.cache_data(show_spinner=False)
def compute_rfm(df):
    d = df.dropna(subset=["Customer_ID"])
    ref_date = df["Transaction_Date"].max().normalize() + pd.Timedelta(days=1)
    rfm = d.groupby("Customer_ID").agg(
        Recency=("Transaction_Date", lambda x: (ref_date - x.max()).days),
        Frequency=("Transaction_ID", "nunique"),
        Monetary=("Line_Total", "sum"),
        Customer_Type=("Customer_Type", _mode),
    ).reset_index()
    return rfm, ref_date


@st.cache_data(show_spinner=False)
def kmeans_eval(rfm, kmin=2, kmax=8):
    X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    rows = []
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        rows.append({"k": k, "Inertia (WCSS)": km.inertia_, "Silhouette Score": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def elbow_k(ev):
    x = ev["k"].values.astype(float); y = ev["Inertia (WCSS)"].values.astype(float)
    xn = (x - x.min()) / (x.max() - x.min() + 1e-9); yn = (y - y.min()) / (y.max() - y.min() + 1e-9)
    p1, p2 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
    d = [abs(np.cross(p2 - p1, p1 - np.array([a, b]))) / np.linalg.norm(p2 - p1) for a, b in zip(xn, yn)]
    return int(x[int(np.argmax(d))])


def recommended_k(ev):
    cand = ev[ev["k"] >= 3]
    return int(cand.loc[cand["Silhouette Score"].idxmax(), "k"])


def seg_base(label):
    for b in SEG_COLORS:
        if str(label).startswith(b):
            return b
    return "Potential Customers"


@st.cache_data(show_spinner=False)
def kmeans_fit(rfm, k):
    X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    out = rfm.copy()
    out["Cluster"] = km.labels_
    prof = out.groupby("Cluster").agg(
        Jumlah=("Customer_ID", "count"),
        Recency_mean=("Recency", "mean"), Recency_med=("Recency", "median"),
        Frequency_mean=("Frequency", "mean"), Frequency_med=("Frequency", "median"),
        Monetary_mean=("Monetary", "mean"), Monetary_med=("Monetary", "median"),
        Monetary_sum=("Monetary", "sum"),
        Member=("Customer_Type", lambda s: (s == "Member").mean() * 100),
    )
    mu = out[["Recency", "Frequency", "Monetary"]].mean()
    sd = out[["Recency", "Frequency", "Monetary"]].std(ddof=0)
    labels, desc = {}, {}
    for cl, r in prof.iterrows():
        zR = (r["Recency_mean"] - mu["Recency"]) / sd["Recency"]
        zF = (r["Frequency_mean"] - mu["Frequency"]) / sd["Frequency"]
        zM = (r["Monetary_mean"] - mu["Monetary"]) / sd["Monetary"]
        if zF >= 0.5 and zM >= 0.5 and zR <= 0.5:
            lab = "Loyal Customers"
        elif zR >= 0.5:
            lab = "Churn Risk"
        elif zR <= 0 and zF < 0 and r["Frequency_mean"] < 2.5:
            lab = "New Customers"
        else:
            lab = "Potential Customers"
        labels[cl] = lab
        rec = "aktif baru-baru ini" if zR < -0.25 else ("sudah lama tidak bertransaksi" if zR > 0.25 else "keaktifan sedang")
        frq = "frekuensi tinggi" if zF > 0.25 else ("frekuensi rendah" if zF < -0.25 else "frekuensi sedang")
        mon = "nilai belanja tinggi" if zM > 0.25 else ("nilai belanja rendah" if zM < -0.25 else "nilai belanja sedang")
        desc[cl] = f"{rec}, {frq}, {mon}"
    # nama unik bila label dobel
    order = prof["Monetary_mean"].sort_values(ascending=False).index
    counts = pd.Series(list(labels.values())).value_counts().to_dict()
    seen = {}
    for cl in order:
        base = labels[cl]
        seen[base] = seen.get(base, 0) + 1
        if counts[base] > 1:
            labels[cl] = f"{base} {chr(64 + seen[base])}"
    prof.insert(0, "Segmen", [labels[c] for c in prof.index])
    prof["Karakteristik"] = [desc[c] for c in prof.index]
    prof["% Pelanggan"] = prof["Jumlah"] / prof["Jumlah"].sum() * 100
    prof["% Kontribusi Pendapatan"] = prof["Monetary_sum"] / prof["Monetary_sum"].sum() * 100
    out["Segmen"] = out["Cluster"].map(labels)
    return out, prof.reset_index(), X


STRATEGI = {
    "Loyal Customers": {
        "strategi": "Program loyalitas eksklusif (poin & reward bertingkat), diskon member, akses awal promo, layanan prioritas.",
        "kpi": "Retention rate, frekuensi belanja/bulan, rata-rata nilai keranjang",
    },
    "Potential Customers": {
        "strategi": "Cross-selling & up-selling memakai aturan Apriori, paket bundling, target frekuensi belanja 1 kali lebih sering.",
        "kpi": "Peningkatan frekuensi, attach rate produk bundling, perpindahan ke segmen Loyal",
    },
    "New Customers": {
        "strategi": "Onboarding: voucher belanja kedua, rekomendasi produk populer, ajakan bergabung menjadi member.",
        "kpi": "Repeat purchase rate 30–60 hari, konversi menjadi member",
    },
    "Churn Risk": {
        "strategi": "Kampanye win-back (voucher personal via WhatsApp/SMS), survei alasan berhenti, penawaran produk favorit terakhir.",
        "kpi": "Reactivation rate, penurunan rata-rata Recency, pendapatan hasil win-back",
    },
}


# ============================================================
# SESSION STATE
# ============================================================
for key, val in {"logged_in": False, "data_raw": None, "catalog": None, "sheets": None, "file_sig": None,
                 "data_clean": None, "clean_log": None, "k_final": None,
                 "min_sup": 0.01, "min_conf": 0.20, "apriori_seen": False, "kmeans_seen": False}.items():
    if key not in st.session_state:
        st.session_state[key] = val


def load_into_session(src, sig):
    raw, catalog, names, target = read_workbook(src)
    st.session_state.update(data_raw=raw, catalog=catalog, sheets=(names, target), file_sig=sig,
                            data_clean=None, clean_log=None, k_final=None,
                            apriori_seen=False, kmeans_seen=False)


def avatar(size=84):
    if os.path.exists("foto_theresa.jpg"):
        st.image("foto_theresa.jpg", width=size + 40)
    else:
        H(f'<div class="avatar" style="width:{size}px;height:{size}px">T</div>')


# ============================================================
# 1. HALAMAN LOGIN
# ============================================================
if not st.session_state.logged_in:
    H("""
    <div class="hero">
        <p class="eyebrow">Universitas Timor (UNIMOR)</p>
        <h1>Sistem Integrasi Business Intelligence &amp; Data Mining</h1>
        <p class="sub">Analisis Transaksi Toko Ritel · Market Basket Analysis (Apriori) · Segmentasi Pelanggan (K-Means RFM) · Executive Dashboard</p>
        <div class="chips">
            <span class="chip">📌 UAS Project BI</span><span class="chip">🛒 Apriori Association</span>
            <span class="chip">👥 K-Means RFM</span><span class="chip">📊 BI Executive Dashboard</span>
        </div>
    </div>
    """)

    col_info, col_login = st.columns([1.15, 1], gap="large")
    with col_info:
        H("""
        <div class="card"><div class="card-title">📌 Informasi Ujian & Penyusun</div>
        <table class="info-table">
            <tr><td class="l">Institusi</td><td class="v">Universitas Timor (UNIMOR)</td></tr>
            <tr><td class="l">Mata Kuliah</td><td class="v">Business Intelligence / Data Mining</td></tr>
            <tr><td class="l">Dosen Pengampu</td><td class="v">Krisantus Jumarto Tey Seran, S.T., M.T.</td></tr>
            <tr><td class="l">Bentuk Ujian</td><td class="v">Proyek Sistem Informasi Analisis Data</td></tr>
            <tr><td class="l">Nama Mahasiswa</td><td class="v">Theresa Ivanna Oki</td></tr>
            <tr><td class="l">NPM</td><td class="v">51230079</td></tr>
            <tr><td class="l">Kelas / Prodi</td><td class="v">VII A / Teknologi Informasi</td></tr>
            <tr><td class="l">Metode</td><td class="v">EDA · Preprocessing · Apriori · K-Means · BI</td></tr>
        </table></div>
        """)
    with col_login:
        H('<div class="card"><div class="card-title">🔐 Masuk ke Sistem Analisis</div>')
        _, cf, _ = st.columns([1, 2, 1])
        with cf:
            avatar(100)
        H('<p style="text-align:center;font-weight:700;color:#1e3a8a;margin:6px 0 14px">Theresa Ivanna Oki</p>')
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Masukkan username...")
            password = st.text_input("Password", type="password", placeholder="Masukkan password...")
            if st.form_submit_button("Masuk ke Sistem 🚀"):
                if username == "theresa" and password == "51230079":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Username atau Password salah! (Default: theresa / 51230079)")
        H("</div>")
    H('<div class="footer">Sistem Informasi Analisis Data · UAS Business Intelligence · Theresa Ivanna Oki (51230079) · Universitas Timor</div>')
    st.stop()

# ============================================================
# 2. SIDEBAR
# ============================================================
with st.sidebar:
    avatar(84)
    H('<div class="profile"><p class="nm">Theresa Ivanna Oki</p><p class="mt">NPM 51230079 · Kelas VII A</p>'
      '<div class="un">UNIVERSITAS TIMOR</div></div>')
    st.markdown("<h3 style='color:#fff !important;font-size:1.05rem'>🧭 Navigasi Tahapan UAS</h3>", unsafe_allow_html=True)
    menu = st.radio("Pilih Tahapan Analisis:", [
        "1. Business Understanding & EDA",
        "2. Preprocessing & Data Cleaning",
        "3. Market Basket (Apriori Analysis)",
        "4. Segmentasi (K-Means & RFM)",
        "5. Executive BI Dashboard",
        "6. Rekomendasi Preskriptif Bisnis",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<p style='font-weight:700;margin-bottom:4px'>Progres Analisis</p>", unsafe_allow_html=True)
    for ok, txt in [(st.session_state.data_raw is not None, "Dataset dimuat"),
                    (st.session_state.data_clean is not None, "Data dibersihkan"),
                    (st.session_state.apriori_seen, "Apriori dijalankan"),
                    (st.session_state.kmeans_seen, "K-Means dijalankan")]:
        H(f'<div class="prog {"done" if ok else ""}"><div class="dot">{"✓" if ok else "·"}</div>{txt}</div>')
    st.markdown("---")
    if st.button("🚪 Keluar (Logout)"):
        st.session_state.logged_in = False
        st.rerun()

raw_ready = st.session_state.data_raw is not None
clean_ready = st.session_state.data_clean is not None


def need_clean():
    if not clean_ready:
        callout("Data belum siap", "<p>Silakan muat dataset pada <b>Menu 1</b> lalu jalankan pembersihan data pada <b>Menu 2</b> terlebih dahulu.</p>", "warn", "⚠️")
        st.stop()


# ============================================================
# MENU 1 — BUSINESS UNDERSTANDING & EDA
# ============================================================
if menu.startswith("1."):
    topbar("Tahap 1 · Soal 1 & 2", "Business Understanding & Exploratory Data Analysis",
           "Memahami masalah bisnis, merumuskan pertanyaan analitis, dan mengeksplorasi kondisi awal dataset transaksi toko.", "10%")

    # ---------- SOAL 1 ----------
    section("1", "Memahami Permasalahan Bisnis", "Soal 1 · Identifikasi masalah, pertanyaan bisnis, dan manfaat analisis")
    callout("a. Permasalahan Utama Perusahaan Ritel", """
        <ul>
        <li>Data transaksi berjumlah besar <b>hanya dipakai untuk laporan penjualan rutin</b>, belum untuk menggali pola perilaku pelanggan.</li>
        <li>Manajemen belum mengetahui <b>kombinasi produk yang sering dibeli bersama</b> sehingga promosi, bundling, dan tata letak rak belum berbasis data.</li>
        <li>Pelanggan belum <b>tersegmentasi</b> berdasarkan perilaku belanja, sehingga strategi pemasaran bersifat seragam dan risiko pelanggan berhenti belanja (<i>churn</i>) tidak terpantau.</li>
        <li><b>Kualitas data rendah</b>: nilai kosong, duplikasi, inkonsistensi teks, serta kuantitas/harga tidak wajar yang bila tidak dibersihkan akan menyesatkan hasil algoritma.</li>
        </ul>""", "risk", "🎯")

    st.markdown("#### b. Pertanyaan Bisnis yang Dijawab Melalui Analisis Data")
    q1, q2, q3, q4 = st.columns(4)
    for col, n, t, m in [
        (q1, 1, "Produk apa yang sering dibeli bersamaan?", "Dijawab: <b>Apriori</b> (support, confidence, lift) → dasar bundling & penataan rak."),
        (q2, 2, "Segmen pelanggan mana yang paling bernilai?", "Dijawab: <b>K-Means + RFM</b> → profil dan label segmen."),
        (q3, 3, "Kapan penjualan meningkat & produk apa yang paling menyumbang pendapatan?", "Dijawab: <b>tren bulanan</b> dan <b>Top 10 produk/kategori</b> pada dashboard."),
        (q4, 4, "Segmen mana yang berisiko tidak kembali?", "Dijawab: segmen dengan <b>Recency tinggi</b> → target kampanye win-back."),
    ]:
        col.markdown(f'<div class="qcard"><div class="qn">{n}</div><b style="color:#1e3a8a">{t}</b><p>{m}</p></div>', unsafe_allow_html=True)

    callout("c. Peran Analisis Data bagi Manajemen", """
        <ul>
        <li><b>Meningkatkan penjualan:</b> bundling & cross-selling dari aturan Apriori menaikkan jumlah item per keranjang (basket size).</li>
        <li><b>Mempertahankan pelanggan:</b> segmentasi RFM memungkinkan program loyalitas untuk pelanggan bernilai tinggi dan <i>win-back</i> untuk pelanggan berisiko churn.</li>
        <li><b>Keputusan operasional:</b> tren bulanan membantu perencanaan stok dan jadwal promosi; dashboard memberi pemantauan KPI yang cepat dibaca.</li>
        </ul>""", "ok", "🚀")

    # ---------- SOAL 2 ----------
    section("2", "Eksplorasi Data (EDA)", "Soal 2 · Upload dataset dan periksa kondisi awal data mentah (Data_Raw tidak diubah)")
    up = st.file_uploader("Upload file Excel dataset UAS (contoh: P8. UTS_Business Intelligence.xlsx)", type=["xlsx", "xls"])
    local_files = [f for f in glob.glob("*.xlsx") if "intelligence" in f.lower()]
    if up is not None:
        sig = (up.name, up.size)
        if sig != st.session_state.file_sig:
            try:
                load_into_session(up, sig)
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")
    elif local_files and not raw_ready:
        if st.button(f"📂 Gunakan file dari folder aplikasi: {os.path.basename(local_files[0])}"):
            try:
                load_into_session(local_files[0], (local_files[0], os.path.getsize(local_files[0])))
                st.rerun()
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

    if st.session_state.data_raw is None:
        callout("Menunggu dataset", "<p>Unggah file Excel yang berisi sheet <b>Data_Raw</b> (dan opsional <b>Data_Produk</b> sebagai katalog harga acuan).</p>", "info", "📥")
        st.stop()

    raw = st.session_state.data_raw.copy()
    catalog = st.session_state.catalog
    names, target = st.session_state.sheets
    for need in ["Transaction_ID", "Transaction_Date", "Customer_ID", "Product_ID", "Product_Name", "Category", "Quantity", "Unit_Price", "Line_Total"]:
        if need not in raw.columns:
            st.error(f"Kolom wajib '{need}' tidak ditemukan pada sheet '{target}'. Periksa file Anda.")
            st.stop()
    for opt in ["Discount_Rate", "Payment_Method", "Customer_Type", "Region"]:
        if opt not in raw.columns:
            raw[opt] = 0.0 if opt == "Discount_Rate" else np.nan
    st.success(f"File berhasil dimuat dari sheet **{target}** — sheet tersedia: {', '.join(names)}.")

    eda = raw.copy()
    eda["Transaction_Date"] = pd.to_datetime(eda["Transaction_Date"], errors="coerce")
    for c in ["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]:
        eda[c] = pd.to_numeric(eda[c], errors="coerce")
    eda["Category_std"] = eda["Category"].map(_norm_category)
    eda["Payment_std"] = eda["Payment_Method"].map(_norm_payment)
    eda["Bulan"] = eda["Transaction_Date"].dt.to_period("M").astype(str)
    qm, pm = invalid_masks(eda, catalog)

    t1, t2, t3, t4, t5 = st.tabs(["📋 Ringkasan Dataset", "📐 Statistik Deskriptif", "🔎 Audit Kualitas Data", "📊 Visualisasi EDA", "💡 Temuan Awal"])

    with t1:
        st.markdown("##### a. Dimensi, rentang waktu, dan ukuran data")
        valid_q = eda.loc[~qm, "Quantity"]
        kpi_row([
            ("📄", "Jumlah Baris", fmt_id(len(eda)), "#2563eb"),
            ("🧱", "Jumlah Kolom", fmt_id(raw.shape[1]), "#8b5cf6"),
            ("🧾", "Transaksi Unik", fmt_id(eda["Transaction_ID"].nunique()), "#06b6d4", "bukan jumlah baris"),
            ("👥", "Pelanggan Unik", fmt_id(eda["Customer_ID"].nunique()), "#10b981"),
        ])
        kpi_row([
            ("📦", "Produk Unik", fmt_id(eda["Product_ID"].nunique()), "#f59e0b"),
            ("🛍️", "Rata-rata Qty (data valid)", fmt_id(valid_q.mean(), 2), "#ec4899"),
            ("💰", "Total Penjualan Sementara", rp(eda["Line_Total"].sum()), "#2563eb", "sebelum dibersihkan"),
            ("📅", "Rentang Waktu", f"{eda['Transaction_Date'].min():%d %b %Y} – {eda['Transaction_Date'].max():%d %b %Y}", "#06b6d4"),
        ])
        st.markdown("##### Tipe data tiap kolom")
        dt = pd.DataFrame({"Kolom": raw.columns, "Tipe Data (terbaca)": raw.dtypes.astype(str).values,
                           "Nilai Unik": [raw[c].nunique() for c in raw.columns],
                           "Nilai Kosong": raw.isna().sum().values, "Contoh Nilai": [str(raw[c].dropna().iloc[0]) if raw[c].notna().any() else "-" for c in raw.columns]})
        show_df(dt, fmt=False)
        st.markdown("##### Pratinjau data mentah (500 baris pertama)")
        show_df(raw, height=320, max_rows=500)

    with t2:
        st.markdown("##### b. Statistik deskriptif kolom numerik")
        desc = eda[["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]].describe().T.reset_index().rename(columns={"index": "Kolom"})
        show_df(desc)
        st.caption("Perhatikan nilai min/max Quantity dan Unit_Price yang tidak wajar — itu sebabnya data wajib dibersihkan sebelum algoritma dijalankan.")
        st.markdown("##### Statistik kolom kategorikal")
        cat_rows = []
        for c in ["Category", "Payment_Method", "Customer_Type", "Region"]:
            vc = eda[c].value_counts(dropna=False)
            cat_rows.append({"Kolom": c, "Nilai Unik (mentah)": eda[c].nunique(), "Paling Sering": str(vc.index[0]), "Frekuensi": int(vc.iloc[0])})
        show_df(pd.DataFrame(cat_rows))

    with t3:
        st.markdown("##### c. Identifikasi missing values, duplikasi, inkonsistensi, dan outlier")
        m = quality_metrics(eda.drop(columns=["Category_std", "Payment_std", "Bulan"]), catalog)
        n = m["Jumlah baris"]
        issues = pd.DataFrame([
            ("Baris dengan ≥ 1 nilai kosong", m["Baris dengan nilai kosong"], "Missing values pada Customer_ID, Region, Quantity, dst."),
            ("Baris duplikat penuh", m["Baris duplikat penuh"], "Semua kolom identik dengan baris lain"),
            ("Kategori tidak konsisten", m["Baris kategori tidak konsisten"], "Variasi huruf besar/kecil (mis. 'sembako', 'SEMBAKO')"),
            ("Metode pembayaran tidak konsisten", m["Baris metode bayar tidak konsisten"], "Variasi ejaan (mis. 'qris', 'Qris')"),
            ("Quantity tidak valid / outlier", m["Quantity tidak valid (kosong/≤0/>50)"], "Kosong, negatif/nol, atau ekstrem (mis. 999)"),
            ("Unit_Price tidak wajar / outlier", m["Unit_Price tidak wajar (kosong/≤0/ekstrem)"], "Kosong, negatif (-100), atau ekstrem (999.999)"),
            ("Line_Total tidak konsisten", m["Line_Total tidak konsisten dengan rumus"], "≠ Qty × Harga × (1 − Diskon)"),
        ], columns=["Jenis Masalah", "Jumlah Baris", "Keterangan"])
        issues["% dari Data"] = issues["Jumlah Baris"] / n * 100
        show_df(issues[["Jenis Masalah", "Jumlah Baris", "% dari Data", "Keterangan"]])
        c1, c2 = st.columns(2)
        miss = eda.drop(columns=["Category_std", "Payment_std", "Bulan"]).isna().sum()
        miss = miss[miss > 0]
        with c1:
            if len(miss):
                show_plot(hbar(miss, "Nilai Kosong per Kolom", "Jumlah sel kosong", 340, money=False), key="eda_miss")
        with c2:
            bx = px.box(eda[eda["Quantity"].between(-5, 1000)], y="Quantity", points="outliers", color_discrete_sequence=["#2563eb"])
            style_fig(bx, "Sebaran Quantity (outlier 999 & nilai ≤ 0 terlihat jelas)", 340, legend=False)
            show_plot(bx, key="eda_box")

    with t4:
        st.markdown("##### d. Visualisasi awal kondisi data")
        v1, v2 = st.columns(2)
        monthly = eda.groupby("Bulan")["Line_Total"].sum().reset_index()
        with v1:
            f = go.Figure(go.Scatter(x=monthly["Bulan"], y=monthly["Line_Total"], mode="lines+markers", fill="tozeroy",
                                     line=dict(color="#2563eb", width=3), marker=dict(size=8, color="#fff", line=dict(color="#2563eb", width=2.5)),
                                     fillcolor="rgba(37,99,235,.12)", hovertemplate="%{x}<br>Rp %{y:,.0f}<extra></extra>"))
            style_fig(f, "1. Tren Penjualan Bulanan (data mentah)", 360, legend=False)
            show_plot(f, key="eda_trend")
        with v2:
            top5 = eda.groupby("Product_Name")["Line_Total"].sum().sort_values(ascending=False).head(10)
            show_plot(hbar(top5, "2. Top 10 Produk berdasarkan Penjualan", "Total penjualan (Rp)", 360), key="eda_top")
        v3, v4 = st.columns(2)
        with v3:
            cs = eda.groupby("Category_std")["Line_Total"].sum().sort_values(ascending=False)
            d = go.Figure(go.Pie(labels=cs.index, values=cs.values, hole=.55, marker=dict(colors=PALETTE + ["#94a3b8"], line=dict(color="#fff", width=2)),
                                 hovertemplate="%{label}<br>Rp %{value:,.0f} (%{percent})<extra></extra>"))
            style_fig(d, "3. Distribusi Penjualan per Kategori (setelah penyeragaman huruf)", 360)
            show_plot(d, key="eda_cat")
        with v4:
            ps = eda["Payment_std"].fillna("Kosong").value_counts()
            b = go.Figure(go.Bar(x=ps.index, y=ps.values, marker=dict(color=PALETTE[:len(ps)], line=dict(width=0)),
                                 hovertemplate="%{x}: %{y:,.0f} baris<extra></extra>"))
            style_fig(b, "4. Metode Pembayaran", 360, legend=False)
            show_plot(b, key="eda_pay")

    with t5:
        st.markdown("##### e. Temuan awal dari eksplorasi data")
        cs = eda.groupby("Category_std")["Line_Total"].sum().sort_values(ascending=False)
        top_p = eda.groupby("Product_Name")["Line_Total"].sum().sort_values(ascending=False)
        mon = eda.groupby("Bulan")["Line_Total"].sum()
        bad_rows = int((eda.drop(columns=["Category_std", "Payment_std", "Bulan"]).isna().any(axis=1) | qm | pm).sum())
        member = eda["Customer_Type"].value_counts(normalize=True)
        callout("Temuan 1 — Kualitas data belum layak langsung dianalisis",
                f"<p><b>{fmt_id(bad_rows)} baris ({bad_rows / len(eda) * 100:.1f}%)</b> mengandung nilai kosong atau kuantitas/harga tidak wajar, ditambah <b>{fmt_id(m['Baris duplikat penuh'])} baris duplikat</b>. Tanpa preprocessing, total penjualan dan frekuensi pelanggan akan bias.</p>", "risk", "🧹")
        callout("Temuan 2 — Konsentrasi pendapatan",
                f"<p>Kategori <b>{cs.index[0]}</b> menyumbang {cs.iloc[0] / cs.sum() * 100:.1f}% penjualan, disusul <b>{cs.index[1]}</b> ({cs.iloc[1] / cs.sum() * 100:.1f}%). Produk dengan penjualan tertinggi adalah <b>{top_p.index[0]}</b> ({rp(top_p.iloc[0])}).</p>", "info", "🏆")
        callout("Temuan 3 — Pola musiman",
                f"<p>Penjualan bulanan tertinggi terjadi pada <b>{mon.idxmax()}</b> ({rp(mon.max())}) dan terendah pada <b>{mon.idxmin()}</b> ({rp(mon.min())}); selisihnya {((mon.max() - mon.min()) / mon.min() * 100):.1f}% — dapat dipakai untuk perencanaan stok dan jadwal promosi.</p>", "ok", "📈")
        callout("Temuan 4 — Struktur data transaksi",
                f"<p>Dataset berisi <b>{fmt_id(len(eda))} baris item</b> tetapi hanya <b>{fmt_id(eda['Transaction_ID'].nunique())} transaksi unik</b> (rata-rata {len(eda) / eda['Transaction_ID'].nunique():.2f} item/transaksi). Karena itu analisis harus memakai Transaction_ID, bukan jumlah baris. Pelanggan Member: {member.get('Member', 0) * 100:.1f}%, Non-Member: {member.get('Non-Member', 0) * 100:.1f}%.</p>", "violet", "🧾")
        st.info("➡️ Lanjut ke **Menu 2** untuk membersihkan data berdasarkan temuan di atas.")

# ============================================================
# MENU 2 — PREPROCESSING
# ============================================================
elif menu.startswith("2."):
    topbar("Tahap 2 · Soal 3 & 4", "Data Preprocessing & Feature Engineering",
           "Membersihkan data secara transparan, membentuk fitur RFM & basket_size, serta menyiapkan matriks untuk Apriori dan K-Means.", "25%")
    if not raw_ready:
        callout("Dataset belum dimuat", "<p>Silakan upload dataset pada <b>Menu 1</b> terlebih dahulu.</p>", "warn", "⚠️")
        st.stop()
    raw = st.session_state.data_raw
    catalog = st.session_state.catalog
    for opt in ["Discount_Rate", "Payment_Method", "Customer_Type", "Region"]:
        if opt not in raw.columns:
            raw = raw.copy(); raw[opt] = 0.0 if opt == "Discount_Rate" else np.nan

    section("3", "Pembersihan & Transformasi Data", "Data_Raw tetap utuh; seluruh perubahan dilakukan pada salinan kerja")
    callout("Rencana pembersihan (berdasarkan audit pada Menu 1)", """
        <ol>
        <li>Standardisasi teks (kategori, metode pembayaran) → 2. Konversi tipe data (tanggal, harga, jumlah) → 3. Hapus duplikat penuh</li>
        <li>Pulihkan nilai kosong dari <b>Transaction_ID yang sama</b> (pelanggan/wilayah/tipe) dan dari <b>Product_Name</b> (Product_ID)</li>
        <li>Perbaiki harga tidak wajar dengan <b>harga referensi produk</b>; pulihkan kuantitas dari <b>Line_Total</b></li>
        <li>Hitung ulang Line_Total, lalu bentuk <b>Basket_Size</b>; hasil akhir diaudit ulang untuk membuktikan masalah telah tertangani</li>
        </ol>""", "info", "🗺️")

    if st.button("🧼 Jalankan Pembersihan & Feature Engineering Lengkap"):
        with st.spinner("Membersihkan data, menangani outlier, dan membentuk fitur..."):
            df_c, log_df = clean_data(raw, catalog)
            st.session_state.data_clean = df_c
            st.session_state.clean_log = log_df
            st.session_state.k_final = None
            st.session_state.apriori_seen = False
            st.session_state.kmeans_seen = False
        st.rerun()

    if st.session_state.data_clean is None:
        st.stop()

    df_c = st.session_state.data_clean
    log_df = st.session_state.clean_log
    m_before = quality_metrics(raw.copy().assign(Catatan_Perubahan=""), catalog)
    m_after = quality_metrics(df_c, catalog)
    st.success(f"Preprocessing selesai: {fmt_id(len(raw))} baris mentah → {fmt_id(len(df_c))} baris bersih.")

    t1, t2, t3, t4, t5 = st.tabs(["🧾 Log Keputusan", "⚖️ Sebelum vs Sesudah", "🧬 Feature Engineering", "💾 Data Bersih & Unduh", "🗄️ Strategi Data Besar"])

    with t1:
        st.markdown("##### Keputusan penanganan tiap masalah beserta alasannya (Soal 3a, 3b, 3c, 3d)")
        show_df(log_df, fmt=True)

    with t2:
        st.markdown("##### Perbandingan kondisi dataset (Soal 3e)")
        comp = pd.DataFrame({"Metrik": list(m_before.keys()), "Sebelum": list(m_before.values()), "Sesudah": list(m_after.values())})
        comp["Perubahan"] = comp["Sesudah"] - comp["Sebelum"]
        show_df(comp)
        st.markdown("##### g. Dokumentasi jumlah data dipertahankan, dihapus, dan diubah")
        removed = len(raw) - len(df_c)
        mod_rows = int((df_c["Catatan_Perubahan"] != "").sum())
        kpi_row([
            ("📥", "Baris Mentah", fmt_id(len(raw)), "#64748b"),
            ("✅", "Baris Dipertahankan", fmt_id(len(df_c)), "#10b981", f"{len(df_c) / len(raw) * 100:.1f}% dari data mentah"),
            ("🗑️", "Baris Dihapus", fmt_id(removed), "#f43f5e", "duplikat & tidak dapat dipulihkan"),
            ("✏️", "Baris Diubah/Diperbaiki", fmt_id(mod_rows), "#f59e0b", "lihat kolom Catatan_Perubahan"),
        ])
        fig = go.Figure(go.Bar(x=["Mentah", "Dipertahankan", "Dihapus", "Diperbaiki"], y=[len(raw), len(df_c), removed, mod_rows],
                               marker=dict(color=["#64748b", "#10b981", "#f43f5e", "#f59e0b"]), text=[fmt_id(v) for v in [len(raw), len(df_c), removed, mod_rows]], textposition="outside"))
        style_fig(fig, "Ringkasan Perubahan Jumlah Baris", 340, legend=False)
        show_plot(fig, key="prep_bar")

    rfm, ref_date = compute_rfm(df_c)
    basket = build_basket(df_c)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    with t3:
        st.markdown("##### Soal 4 — Feature engineering & persiapan analisis")
        callout("a. Fitur RFM per pelanggan", f"""
            <ul><li><b>Tanggal acuan (dokumentasi):</b> {ref_date:%d %B %Y} = satu hari setelah transaksi terakhir pada data bersih.</li>
            <li><b>Recency</b> = hari sejak transaksi terakhir terhadap tanggal acuan · <b>Frequency</b> = jumlah Transaction_ID unik · <b>Monetary</b> = total penjualan bersih (Line_Total setelah diskon).</li>
            <li>Baris tanpa Customer_ID dikeluarkan dari RFM (jumlah: {int(df_c['Customer_ID'].isna().sum())}). Satu baris = satu pelanggan ({fmt_id(len(rfm))} pelanggan).</li></ul>""", "info", "🧮")
        show_df(rfm.drop(columns=["Customer_Type"]).head(10))
        c1, c2 = st.columns(2)
        with c1:
            bs = df_c.drop_duplicates("Transaction_ID")["Basket_Size"].value_counts().sort_index()
            f = go.Figure(go.Bar(x=bs.index.astype(str), y=bs.values, marker=dict(color=bs.values, colorscale=BLUE_SCALE),
                                 hovertemplate="%{x} produk: %{y:,.0f} transaksi<extra></extra>"))
            style_fig(f, "b. Distribusi Basket_Size (produk unik per transaksi)", 340, legend=False)
            f.update_xaxes(title="Basket size"); f.update_yaxes(title="Jumlah transaksi")
            show_plot(f, key="prep_basket")
        with c2:
            callout("c. Matriks transaksi × produk (Apriori)", f"<p>Bentuk matriks: <b>{fmt_id(basket.shape[0])} transaksi × {basket.shape[1]} produk</b>. Nilai 1 = produk ada pada nota, 0 = tidak ada (biner, bukan jumlah unit). Satu baris mewakili satu transaksi, bukan satu baris item.</p>", "violet", "🛒")
            show_df(basket.astype(int).head(6), fmt=False)
        callout("d & e. Matriks fitur K-Means & standardisasi", f"""
            <p><b>Matriks fitur:</b> {fmt_id(Xs.shape[0])} pelanggan × 3 fitur (Recency, Frequency, Monetary); Customer_ID <u>tidak</u> dimasukkan sebagai fitur.</p>
            <p><b>Metode:</b> <i>StandardScaler</i> (z-score) → setiap fitur bermean ≈ 0 dan std = 1 (hasil: mean {np.round(Xs.mean(axis=0), 3).tolist()}, std {np.round(Xs.std(axis=0), 3).tolist()}).</p>
            <p><b>Alasan:</b> K-Means memakai jarak Euclidean. Monetary berskala jutaan rupiah sedangkan Frequency hanya 1–14; tanpa standardisasi Monetary akan mendominasi jarak dan membuat klaster bias terhadap satu fitur.</p>""", "ok", "⚖️")
        sc = pd.DataFrame({"Fitur": ["Recency", "Frequency", "Monetary"],
                           "Mean (asli)": rfm[["Recency", "Frequency", "Monetary"]].mean().values, "Std (asli)": rfm[["Recency", "Frequency", "Monetary"]].std().values,
                           "Mean (terstandar)": Xs.mean(axis=0), "Std (terstandar)": Xs.std(axis=0)})
        show_df(sc)

    with t4:
        st.markdown("##### Dataset hasil preprocessing (500 baris pertama)")
        show_df(df_c, height=340, max_rows=500)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            df_c.to_excel(xw, sheet_name="Data_Clean", index=False)
            log_df.to_excel(xw, sheet_name="Log_Preprocessing", index=False)
            comp.to_excel(xw, sheet_name="Sebelum_vs_Sesudah", index=False)
            rfm.to_excel(xw, sheet_name="RFM", index=False)
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Data_Clean (.csv)", csv_bytes(df_c), "Data_Clean.csv", "text/csv")
        d2.download_button("⬇️ Log Preprocessing (.csv)", csv_bytes(log_df), "Log_Preprocessing.csv", "text/csv")
        d3.download_button("⬇️ Paket Lengkap (.xlsx)", buf.getvalue(), "Hasil_Preprocessing_UAS.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with t5:
        callout("f. Strategi mengolah dataset berukuran besar", """
            <ul>
            <li><b>Sampling:</b> ambil sampel acak/stratifikasi (mis. per bulan atau kategori) untuk eksplorasi cepat; validasi hasil akhir pada data penuh.</li>
            <li><b>Chunking:</b> baca file bertahap (<code>pd.read_csv(..., chunksize=100000)</code>), bersihkan per potongan, lalu gabungkan hasil agregat.</li>
            <li><b>Agregasi:</b> ubah data item (18 ribu baris) menjadi tabel <i>transaksi-basket</i> (untuk Apriori) dan tabel <i>pelanggan-RFM</i> (untuk K-Means) — ukuran jauh lebih kecil namun informasi analisis tetap utuh.</li>
            <li><b>Efisiensi:</b> tipe data hemat memori (<code>category</code>, <code>int32</code>), caching hasil (<code>st.cache_data</code> dipakai pada aplikasi ini), dan menaikkan <i>min_support</i> agar Apriori tidak meledak.</li>
            </ul>""", "violet", "🗄️")

# ============================================================
# MENU 3 — APRIORI
# ============================================================
elif menu.startswith("3."):
    topbar("Tahap 3 · Soal 5 & 6", "Market Basket Analysis dengan Algoritma Apriori",
           "Menemukan kombinasi produk yang sering dibeli bersamaan dan menerjemahkannya menjadi strategi cross-selling & penataan rak.", "20%")
    need_clean()
    df_c = st.session_state.data_clean
    basket = build_basket(df_c)
    n_tx = len(basket)
    item_sup = basket.mean()

    section("5", "Matriks Transaksi & Parameter Apriori", "Soal 5 a–b · Tentukan minimum support dan confidence beserta alasannya")
    kpi_row([
        ("🧾", "Transaksi (basket)", fmt_id(n_tx), "#2563eb"),
        ("📦", "Produk Unik", fmt_id(basket.shape[1]), "#8b5cf6"),
        ("🛒", "Rata-rata Item/Basket", fmt_id(basket.sum(axis=1).mean(), 2), "#06b6d4"),
        ("📊", "Rata-rata Support Produk", f"{item_sup.mean() * 100:.1f}%".replace(".", ","), "#f59e0b"),
    ])
    c1, c2 = st.columns(2)
    with c1:
        min_sup = st.slider("Minimum Support", 0.005, 0.10, float(st.session_state.min_sup), 0.005, format="%.3f")
    with c2:
        min_conf = st.slider("Minimum Confidence", 0.05, 0.90, float(st.session_state.min_conf), 0.05)
    st.session_state.min_sup, st.session_state.min_conf = min_sup, min_conf

    callout("Alasan pemilihan nilai (berdasarkan karakteristik data & tujuan bisnis)", f"""
        <ul>
        <li><b>Min. support = {min_sup * 100:.1f}%</b> ≈ kombinasi muncul pada minimal <b>{int(round(min_sup * n_tx))} transaksi</b>. Dataset berisi {basket.shape[1]} produk dengan rata-rata {basket.sum(axis=1).mean():.1f} item/basket,
            sehingga support satu produk rata-rata ±{item_sup.mean() * 100:.0f}% dan pasangan produk wajar bernilai sekitar 1%. Nilai yang lebih tinggi akan membuang hampir semua pasangan; nilai terlalu rendah memunculkan pola kebetulan.</li>
        <li><b>Min. confidence = {min_conf * 100:.0f}%</b> ≈ minimal {min_conf / max(item_sup.mean(), 1e-9):.1f}× peluang dasar pembelian satu produk, sehingga aturan benar-benar lebih kuat daripada membeli acak.</li>
        <li><b>Filter lift &gt; 1</b> dan peringkat berdasarkan <b>Lift → Confidence → Support</b>: lift menunjukkan kekuatan hubungan di atas kebetulan, confidence kepastian aturan, support seberapa luas aturan berlaku.</li>
        </ul>""", "info", "🎚️")

    with st.spinner("Menghitung frequent itemsets dan association rules..."):
        freq, rules, _ = run_apriori(df_c, min_sup, min_conf)
    st.session_state.apriori_seen = True

    section("6", "Hasil Frequent Itemsets & Association Rules", "Soal 5 c–g · Itemset, aturan, support/confidence/lift, dan visualisasi")
    if len(freq) == 0:
        callout("Tidak ada frequent itemset", "<p>Turunkan nilai minimum support.</p>", "warn", "⚠️"); st.stop()
    ft = freq_table(freq)
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi("🧩", "Frequent Itemsets", fmt_id(len(ft)), "#2563eb"), unsafe_allow_html=True)
    k2.markdown(kpi("🔗", "Itemset ≥ 2 Produk", fmt_id(int((ft["Ukuran"] >= 2).sum())), "#8b5cf6"), unsafe_allow_html=True)
    k3.markdown(kpi("📜", "Aturan (lift > 1)", fmt_id(len(rules)), "#10b981"), unsafe_allow_html=True)

    tabA, tabB, tabC = st.tabs(["🏅 Top 10 Aturan", "🧩 Frequent Itemsets", "🕸️ Visualisasi"])
    with tabA:
        if len(rules) == 0:
            callout("Tidak ada aturan", "<p>Turunkan minimum confidence atau minimum support.</p>", "warn", "⚠️")
        else:
            st.markdown("**Maksimal 10 aturan terbaik** — diurutkan berdasarkan Lift, kemudian Confidence, kemudian Support")
            top_tbl = rules_table(rules, 10)
            show_df(top_tbl)
            st.download_button("⬇️ Unduh seluruh aturan (.csv)", csv_bytes(rules_table(rules, len(rules))), "Aturan_Asosiasi.csv", "text/csv")
    with tabB:
        st.markdown("**Frequent itemsets** yang memenuhi minimum support (diurutkan dari support tertinggi)")
        show_df(ft.head(30))
    with tabC:
        if len(rules) == 0:
            st.info("Belum ada aturan untuk divisualisasikan.")
        else:
            show_plot(network_figure(rules, 15), key="apr_net")
            hm = lift_heatmap(rules, 30)
            if hm is not None:
                show_plot(hm, key="apr_hm")
            sc = go.Figure(go.Scatter(x=rules["support"].head(30), y=rules["confidence"].head(30), mode="markers",
                                      marker=dict(size=rules["lift"].head(30) * 12, color=rules["lift"].head(30), colorscale=BLUE_SCALE, showscale=True,
                                                  colorbar=dict(title="Lift"), line=dict(color="#fff", width=1.5)),
                                      text=[f"{' + '.join(sorted(a))} → {' + '.join(sorted(c))}" for a, c in zip(rules['antecedents'].head(30), rules['consequents'].head(30))],
                                      hovertemplate="%{text}<br>Support %{x:.3f} · Confidence %{y:.3f}<extra></extra>"))
            style_fig(sc, "Support vs Confidence (ukuran & warna = Lift)", 400, legend=False)
            sc.update_xaxes(title="Support"); sc.update_yaxes(title="Confidence")
            show_plot(sc, key="apr_sc")

    if len(rules):
        pairs, seen = [], set()
        for _, r in rules.iterrows():
            key = frozenset(set(r["antecedents"]) | set(r["consequents"]))
            if key in seen:
                continue
            seen.add(key); pairs.append(r)
            if len(pairs) == 3:
                break
        A = lambda r: " + ".join(sorted(r["antecedents"])); B = lambda r: " + ".join(sorted(r["consequents"]))
        section("7", "Interpretasi Bisnis & Rekomendasi", "Soal 6 · Makna metrik, rekomendasi cross-selling/bundling, dan penempatan rak")
        best = pairs[0]
        callout("a & b. Aturan paling menarik dan arti nilainya", f"""
            <p>Aturan terkuat: <b>{A(best)} → {B(best)}</b></p>
            <ul>
            <li><b>Support {best['support']:.3f}</b> ({best['support'] * 100:.1f}%): kombinasi ini muncul pada ±{int(round(best['support'] * n_tx))} dari {fmt_id(n_tx)} transaksi.</li>
            <li><b>Confidence {best['confidence']:.3f}</b> ({best['confidence'] * 100:.1f}%): dari semua transaksi yang memuat {A(best)}, {best['confidence'] * 100:.1f}% juga memuat {B(best)}.</li>
            <li><b>Lift {best['lift']:.2f}</b>: pembelian {A(best)} membuat peluang membeli {B(best)} <b>{best['lift']:.2f}× lebih besar</b> dibanding peluang dasarnya — hubungan nyata, bukan kebetulan. <i>Catatan: aturan asosiasi bukan bukti sebab-akibat.</i></li>
            </ul>""", "ok", "🏆")
        recs = "".join(
            f"<li><b>Bundling #{i + 1} — {A(r)} + {B(r)}:</b> paket hemat/diskon silang. Alasan: lift {r['lift']:.2f}, confidence {r['confidence'] * 100:.0f}%, support {r['support'] * 100:.1f}% (±{int(round(r['support'] * n_tx))} transaksi).</li>"
            for i, r in enumerate(pairs))
        callout("c. Tiga rekomendasi cross-selling / bundling", f"<ul>{recs}</ul>", "violet", "🎁")
        shelf = "".join(f"<li>Letakkan <b>{A(r)}</b> berdekatan dengan <b>{B(r)}</b> (lift {r['lift']:.2f}); pasang rak promosi di antara keduanya.</li>" for r in pairs)
        callout("d & e. Penempatan produk di rak/etalase (berdasarkan angka lift, bukan perkiraan)", f"<ul>{shelf}</ul>", "info", "🗄️")

    section("🛍", "Simulator Keranjang Belanja", "Pilih produk di keranjang untuk melihat rekomendasi tambahan secara real-time")
    all_products = sorted(df_c["Product_Name"].astype(str).unique())
    top_ant = [x for x in (sorted(rules.iloc[0]["antecedents"]) if len(rules) else []) if x in all_products]
    user_basket = st.multiselect("Produk dalam keranjang:", all_products, default=top_ant[:1] or all_products[:1])
    if user_basket and len(rules):
        rec = []
        for _, r in rules.iterrows():
            if set(r["antecedents"]).issubset(user_basket) and not set(r["consequents"]).issubset(user_basket):
                rec.append({"Produk di keranjang": " + ".join(sorted(r["antecedents"])), "Rekomendasi tambahan": " + ".join(sorted(r["consequents"])),
                            "Support": r["support"], "Confidence": r["confidence"], "Lift": r["lift"]})
        if rec:
            st.success(f"Ditemukan {len(rec)} rekomendasi produk pelengkap.")
            show_df(pd.DataFrame(rec).sort_values("Lift", ascending=False))
        else:
            st.info("Belum ada rekomendasi untuk kombinasi keranjang ini pada parameter saat ini.")

# ============================================================
# MENU 4 — K-MEANS
# ============================================================
elif menu.startswith("4."):
    topbar("Tahap 4 · Soal 7 & 8", "Segmentasi Pelanggan dengan K-Means & RFM",
           "Menentukan jumlah klaster optimal (Elbow + Silhouette), memprofilkan segmen, dan menyusun strategi pemasaran.", "20%")
    need_clean()
    df_c = st.session_state.data_clean
    rfm, ref_date = compute_rfm(df_c)

    section("7", "Pemodelan Segmentasi Pelanggan", "Soal 7 · Fitur, pemilihan k, penerapan K-Means, dan visualisasi")
    callout("a. Fitur yang dipakai", f"""
        <p>Segmentasi memakai <b>RFM</b> pada {fmt_id(len(rfm))} pelanggan (satu baris = satu pelanggan; tanggal acuan {ref_date:%d %b %Y}):
        <b>Recency</b> (hari sejak belanja terakhir), <b>Frequency</b> (jumlah transaksi unik), <b>Monetary</b> (total belanja bersih).
        Fitur distandardisasi dengan <i>StandardScaler</i> agar tidak ada fitur yang mendominasi jarak Euclidean.</p>""", "info", "🧮")

    ev = kmeans_eval(rfm, 2, 8)
    k_elbow, k_rec = elbow_k(ev), recommended_k(ev)
    b = st.columns([1.1, 1])
    with b[0]:
        f = go.Figure(go.Scatter(x=ev["k"], y=ev["Inertia (WCSS)"], mode="lines+markers", line=dict(color="#2563eb", width=3),
                                 marker=dict(size=10, color="#fff", line=dict(color="#2563eb", width=2.5)),
                                 hovertemplate="k=%{x}<br>WCSS=%{y:,.0f}<extra></extra>"))
        f.add_vline(x=k_elbow, line_dash="dash", line_color="#f59e0b", annotation_text=f"Siku (elbow) k={k_elbow}")
        style_fig(f, "b. Elbow Method (Inertia vs k)", 360, legend=False)
        f.update_xaxes(title="Jumlah klaster (k)", dtick=1); f.update_yaxes(title="Inertia (WCSS)")
        show_plot(f, key="km_elbow")
    with b[1]:
        colors = ["#10b981" if k == k_rec else "#93c5fd" for k in ev["k"]]
        g = go.Figure(go.Bar(x=ev["k"], y=ev["Silhouette Score"], marker=dict(color=colors), text=[f"{v:.3f}".replace(".", ",") for v in ev["Silhouette Score"]],
                             textposition="outside", hovertemplate="k=%{x}<br>Silhouette=%{y:.3f}<extra></extra>"))
        style_fig(g, "c. Silhouette Score per k (hijau = terpilih)", 360, legend=False)
        g.update_xaxes(title="Jumlah klaster (k)", dtick=1); g.update_yaxes(title="Silhouette", range=[0, ev["Silhouette Score"].max() * 1.2])
        show_plot(g, key="km_sil")
    show_df(ev)

    best_sil = ev.loc[ev["k"] == k_rec, "Silhouette Score"].iloc[0]
    callout("d. Alasan pemilihan jumlah klaster", f"""
        <ul>
        <li><b>Metrik:</b> Silhouette tertinggi untuk k ≥ 3 ada pada <b>k = {k_rec}</b> (skor {best_sil:.3f}); titik siku Elbow berada di sekitar k = {k_elbow}.</li>
        <li><b>Interpretasi bisnis:</b> k = 2 hanya memisahkan "bernilai tinggi vs rendah", terlalu kasar untuk strategi pemasaran berbeda. Minimal 3 segmen diperlukan (mis. loyal, potensial, berisiko), sedangkan k terlalu besar menyulitkan eksekusi dan menurunkan silhouette.</li>
        <li>Anda dapat mengganti nilai k di bawah ini; seluruh profil, dashboard, dan rekomendasi akan mengikuti pilihan Anda.</li>
        </ul>""", "ok", "🎯")

    k_default = st.session_state.k_final or k_rec
    k_sel = st.slider("Jumlah klaster (k) yang digunakan:", 2, 8, int(k_default))
    st.session_state.k_final = k_sel
    seg, prof, Xs = kmeans_fit(rfm, k_sel)
    st.session_state.kmeans_seen = True

    section("8", "Hasil Klaster, Profil & Strategi", "Soal 7 e–f dan Soal 8 · Profil, label, strategi, dan prioritas segmen")
    sil = silhouette_score(Xs, seg["Cluster"])
    st.success(f"K-Means dengan k = {k_sel} selesai · Silhouette Score = {fmt_id(sil, 3)}")

    tabA, tabB, tabC = st.tabs(["👥 Profil Segmen", "📈 Visualisasi Klaster", "🧭 Strategi Pemasaran"])
    with tabA:
        ptab = prof[["Cluster", "Segmen", "Jumlah", "% Pelanggan", "Recency_mean", "Recency_med", "Frequency_mean", "Frequency_med",
                     "Monetary_mean", "Monetary_med", "% Kontribusi Pendapatan", "Member", "Karakteristik"]].rename(columns={
            "Jumlah": "Jumlah Pelanggan", "Recency_mean": "Recency Rata-rata (hari)", "Recency_med": "Recency Median",
            "Frequency_mean": "Frequency Rata-rata", "Frequency_med": "Frequency Median", "Monetary_mean": "Monetary Rata-rata (Rp)",
            "Monetary_med": "Monetary Median (Rp)", "Member": "% Member"})
        show_df(ptab)
        cc = st.columns(2)
        with cc[0]:
            cnt = go.Figure(go.Bar(x=prof["Segmen"], y=prof["Jumlah"], marker=dict(color=[SEG_COLORS[seg_base(s)] for s in prof["Segmen"]]),
                                   text=prof["Jumlah"], textposition="outside"))
            style_fig(cnt, "Jumlah Pelanggan per Segmen", 340, legend=False)
            show_plot(cnt, key="km_cnt")
        with cc[1]:
            pie = go.Figure(go.Pie(labels=prof["Segmen"], values=prof["Monetary_sum"], hole=.55,
                                   marker=dict(colors=[SEG_COLORS[seg_base(s)] for s in prof["Segmen"]], line=dict(color="#fff", width=2)),
                                   hovertemplate="%{label}<br>Rp %{value:,.0f} (%{percent})<extra></extra>"))
            style_fig(pie, "Kontribusi Pendapatan per Segmen", 340)
            show_plot(pie, key="km_pie")
        callout("a–c. Karakteristik tiap segmen (berdasarkan perhitungan)", "<ul>" + "".join(
            f"<li><b>{r['Segmen']}</b> — {fmt_id(r['Jumlah'])} pelanggan ({r['% Pelanggan']:.1f}%): {r['Karakteristik']}. Rata-rata Recency {r['Recency_mean']:.0f} hari, Frequency {r['Frequency_mean']:.1f} transaksi, Monetary {rp(r['Monetary_mean'])}; menyumbang {r['% Kontribusi Pendapatan']:.1f}% pendapatan.</li>"
            for _, r in prof.sort_values("Monetary_mean", ascending=False).iterrows()) + "</ul>"
            + "<p><i>Label ditentukan otomatis dari posisi rata-rata RFM klaster terhadap rata-rata seluruh pelanggan (z-score), bukan ditetapkan manual.</i></p>", "info", "🏷️")
    with tabB:
        sc = px.scatter(seg, x="Frequency", y="Monetary", color="Segmen", color_discrete_map={s: SEG_COLORS[seg_base(s)] for s in prof["Segmen"]},
                        hover_data={"Customer_ID": True, "Recency": True}, opacity=.75)
        sc.update_traces(marker=dict(size=9, line=dict(width=.5, color="#fff")))
        style_fig(sc, f"f. Scatter Plot Klaster K-Means (k = {k_sel}) — Frequency vs Monetary", 460)
        show_plot(sc, key="km_sc")
        s3 = px.scatter_3d(seg, x="Recency", y="Frequency", z="Monetary", color="Segmen", opacity=.8,
                           color_discrete_map={s: SEG_COLORS[seg_base(s)] for s in prof["Segmen"]})
        s3.update_traces(marker=dict(size=3.5))
        style_fig(s3, "Visualisasi 3D RFM", 520)
        show_plot(s3, key="km_3d")
    with tabC:
        ranked = prof.sort_values("Monetary_sum", ascending=False)
        top_seg = ranked.iloc[0]
        churn = prof[prof["Segmen"].str.startswith("Churn Risk")]
        st.markdown("##### d. Strategi pemasaran untuk tiap segmen")
        rows = []
        for _, r in prof.sort_values("Monetary_mean", ascending=False).iterrows():
            s = STRATEGI[seg_base(r["Segmen"])]
            rows.append({"Segmen": r["Segmen"], "Jumlah Pelanggan": r["Jumlah"], "Strategi Pemasaran": s["strategi"], "KPI Pantau": s["kpi"]})
        show_df(pd.DataFrame(rows))
        extra = ""
        if len(churn):
            c0 = churn.sort_values("Monetary_sum", ascending=False).iloc[0]
            extra = f"<li><b>Prioritas penyelamatan:</b> {c0['Segmen']} — {fmt_id(c0['Jumlah'])} pelanggan dengan Recency rata-rata {c0['Recency_mean']:.0f} hari yang masih memegang {rp(c0['Monetary_sum'])} nilai belanja historis; reaktivasi lebih murah daripada akuisisi pelanggan baru.</li>"
        callout("e. Segmen prioritas perusahaan", f"""
            <ul>
            <li><b>Prioritas utama (nilai):</b> {top_seg['Segmen']} — hanya {top_seg['% Pelanggan']:.1f}% pelanggan tetapi menyumbang <b>{top_seg['% Kontribusi Pendapatan']:.1f}%</b> pendapatan; kehilangan segmen ini berdampak paling besar.</li>
            {extra}
            </ul>""", "ok", "⭐")

# ============================================================
# MENU 5 — DASHBOARD
# ============================================================
elif menu.startswith("5."):
    topbar("Tahap 5 · Soal 9", "Executive Business Intelligence Dashboard",
           "KPI, tren penjualan, Top 10, aturan Apriori, dan profil K-Means dalam satu tampilan interaktif untuk manajemen.", "15%")
    need_clean()
    df_all = st.session_state.data_clean.copy()
    rfm, ref_date = compute_rfm(df_all)
    ev = kmeans_eval(rfm, 2, 8)
    k_use = st.session_state.k_final or recommended_k(ev)
    seg, prof, _ = kmeans_fit(rfm, k_use)
    df_all["Segmen"] = df_all["Customer_ID"].map(seg.set_index("Customer_ID")["Segmen"]).fillna("Tanpa ID")
    st.session_state.kmeans_seen = True

    section("🎛", "Filter Dashboard", "Periode transaksi, kategori produk, dan segmen pelanggan")
    fc1, fc2, fc3 = st.columns([1.1, 1.2, 1.2])
    dmin, dmax = df_all["Transaction_Date"].min().date(), df_all["Transaction_Date"].max().date()
    with fc1:
        dr = st.date_input("Periode transaksi", (dmin, dmax), min_value=dmin, max_value=dmax)
    cats = sorted(df_all["Category"].dropna().unique())
    segs = sorted(df_all["Segmen"].unique())
    with fc2:
        sel_cat = st.multiselect("Kategori produk", cats, default=cats)
    with fc3:
        sel_seg = st.multiselect("Segmen pelanggan", segs, default=segs)
    d0, d1 = (dr[0], dr[1]) if isinstance(dr, (tuple, list)) and len(dr) == 2 else (dmin, dmax)
    df_f = df_all[(df_all["Transaction_Date"].dt.date >= d0) & (df_all["Transaction_Date"].dt.date <= d1)
                  & df_all["Category"].isin(sel_cat) & df_all["Segmen"].isin(sel_seg)].copy()
    if df_f.empty:
        callout("Tidak ada data", "<p>Filter yang dipilih tidak menghasilkan data. Ubah filter.</p>", "warn", "⚠️"); st.stop()

    # KPI
    rev = df_f["Line_Total"].sum()
    trx = df_f["Transaction_ID"].nunique()
    cust = df_f["Customer_ID"].nunique()
    bsk = df_f.groupby("Transaction_ID")["Product_ID"].nunique().mean()
    section("a", "Key Performance Indicators", f"{len(df_f):,} baris item pada filter aktif".replace(",", "."))
    kpi_row([
        ("💰", "Total Pendapatan", rp(rev), "#2563eb"),
        ("🧾", "Jumlah Transaksi", fmt_id(trx), "#06b6d4"),
        ("👥", "Pelanggan Unik", fmt_id(cust), "#10b981"),
        ("🛒", "Rata-rata Basket Size", fmt_id(bsk, 2), "#f59e0b", "produk unik / transaksi"),
    ])
    kpi_row([
        ("🧮", "Rata-rata Nilai Transaksi", rp(rev / trx), "#8b5cf6"),
        ("📦", "Item Terjual (unit)", fmt_id(df_f["Quantity"].sum()), "#ec4899"),
        ("🏷️", "Produk Terjual", fmt_id(df_f["Product_ID"].nunique()), "#64748b"),
        ("🎯", "Segmen Aktif", fmt_id(df_f["Segmen"].nunique()), "#2563eb", f"k = {k_use}"),
    ])

    tabs = st.tabs(["📈 Analisis Penjualan", "🛒 Analisis Apriori", "👥 Analisis K-Means", "🧩 Bonus: Produk per Segmen", "🗂️ Data"])

    with tabs[0]:
        mo = df_f.groupby("Bulan").agg(Pendapatan=("Line_Total", "sum"), Transaksi=("Transaction_ID", "nunique")).reset_index()
        f = go.Figure()
        f.add_trace(go.Bar(x=mo["Bulan"], y=mo["Transaksi"], name="Transaksi", marker=dict(color="rgba(147,197,253,.55)"), yaxis="y2",
                           hovertemplate="%{x}: %{y:,.0f} transaksi<extra></extra>"))
        f.add_trace(go.Scatter(x=mo["Bulan"], y=mo["Pendapatan"], name="Pendapatan", mode="lines+markers", line=dict(color="#2563eb", width=3.5),
                               marker=dict(size=9, color="#fff", line=dict(color="#2563eb", width=2.5)), hovertemplate="%{x}<br>Rp %{y:,.0f}<extra></extra>"))
        style_fig(f, "Tren Penjualan Bulanan (garis = pendapatan, batang = jumlah transaksi)", 400)
        f.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Transaksi"), yaxis=dict(title="Pendapatan (Rp)"))
        show_plot(f, key="db_trend")
        c1, c2 = st.columns(2)
        with c1:
            mode_top = st.radio("Top 10 berdasarkan:", ["Produk", "Kategori"], horizontal=True)
            colx = "Product_Name" if mode_top == "Produk" else "Category"
            tp = df_f.groupby(colx)["Line_Total"].sum().sort_values(ascending=False).head(10)
            show_plot(hbar(tp, f"Top 10 {mode_top} berdasarkan Penjualan", "Total penjualan (Rp)", 400), key="db_top")
        with c2:
            st.markdown("<div style='height:38px'></div>", unsafe_allow_html=True)
            cs = df_f.groupby("Category")["Line_Total"].sum().sort_values(ascending=False)
            pie = go.Figure(go.Pie(labels=cs.index, values=cs.values, hole=.55, marker=dict(colors=PALETTE + ["#94a3b8", "#cbd5e1"], line=dict(color="#fff", width=2)),
                                   hovertemplate="%{label}<br>Rp %{value:,.0f} (%{percent})<extra></extra>"))
            style_fig(pie, "Distribusi Penjualan per Kategori", 400)
            show_plot(pie, key="db_cat")
        rg = df_f.groupby("Region")["Line_Total"].sum().sort_values(ascending=False)
        show_plot(hbar(rg, "Penjualan per Wilayah", "Total penjualan (Rp)", 340), key="db_reg")
        top_cat = cs.index[0]; pk = mo.loc[mo["Pendapatan"].idxmax()]
        callout("Insight untuk manajemen", f"<ul><li>Pendapatan puncak pada <b>{pk['Bulan']}</b> ({rp(pk['Pendapatan'])}) — siapkan stok dan promosi sebelum bulan tersebut.</li>"
                f"<li>Kategori <b>{top_cat}</b> menyumbang {cs.iloc[0] / cs.sum() * 100:.1f}% penjualan pada filter aktif.</li></ul>", "info", "💡")

    with tabs[1]:
        if df_f["Transaction_ID"].nunique() < 100:
            st.info("Transaksi pada filter terlalu sedikit untuk Apriori yang stabil. Perlebar filter.")
        else:
            freq, rules, n_tx = run_apriori(df_f, st.session_state.min_sup, st.session_state.min_conf)
            st.caption(f"Parameter: min support {st.session_state.min_sup:.3f} · min confidence {st.session_state.min_conf:.2f} (ubah di Menu 3) · {fmt_id(n_tx)} transaksi pada filter")
            if len(rules) == 0:
                st.warning("Tidak ada aturan pada filter ini. Turunkan min support/confidence di Menu 3.")
            else:
                rt = rules_table(rules, 10)
                show_df(rt)
                lab = [f"{a} → {b}" for a, b in zip(rt["Jika membeli (A)"], rt["Maka juga membeli (B)"])]
                bar = go.Figure(go.Bar(x=rt["Lift"][::-1], y=lab[::-1], orientation="h", marker=dict(color=rt["Lift"][::-1], colorscale=BLUE_SCALE),
                                       hovertemplate="Lift %{x:.2f}<extra>%{y}</extra>"))
                style_fig(bar, "Visualisasi Kombinasi Produk — Lift Top 10 Aturan", 420, legend=False)
                show_plot(bar, key="db_rules")
                show_plot(network_figure(rules, 12), key="db_net")

    with tabs[2]:
        fseg = seg[seg["Segmen"].isin(sel_seg)]
        pf = prof[prof["Segmen"].isin(sel_seg)]
        c1, c2 = st.columns(2)
        with c1:
            cnt = go.Figure(go.Bar(x=pf["Segmen"], y=pf["Jumlah"], marker=dict(color=[SEG_COLORS[seg_base(s)] for s in pf["Segmen"]]), text=pf["Jumlah"], textposition="outside"))
            style_fig(cnt, "Jumlah Pelanggan per Klaster", 360, legend=False)
            show_plot(cnt, key="db_km_cnt")
        with c2:
            sc = px.scatter(fseg, x="Frequency", y="Monetary", color="Segmen", color_discrete_map={s: SEG_COLORS[seg_base(s)] for s in prof["Segmen"]}, opacity=.75)
            sc.update_traces(marker=dict(size=8))
            style_fig(sc, "Visualisasi Segmentasi Pelanggan", 360)
            show_plot(sc, key="db_km_sc")
        show_df(pf[["Segmen", "Jumlah", "% Pelanggan", "Recency_mean", "Frequency_mean", "Monetary_mean", "% Kontribusi Pendapatan", "Karakteristik"]].rename(columns={
            "Jumlah": "Jumlah Pelanggan", "Recency_mean": "Recency Rata-rata", "Frequency_mean": "Frequency Rata-rata", "Monetary_mean": "Monetary Rata-rata (Rp)"}))
        st.caption("Profil klaster dihitung dari seluruh pelanggan (periode penuh); filter segmen menyaring tampilan dan data transaksi.")

    with tabs[3]:
        st.markdown("**Bonus — integrasi Apriori & K-Means:** produk dan kombinasi yang relevan bagi tiap segmen")
        dseg = df_f[df_f["Segmen"] != "Tanpa ID"]
        if dseg.empty:
            st.info("Tidak ada data segmen pada filter ini.")
        else:
            tx_seg = dseg.drop_duplicates("Transaction_ID").groupby("Segmen").size()
            pen = (dseg.drop_duplicates(["Transaction_ID", "Product_Name"]).groupby(["Segmen", "Product_Name"]).size().unstack(fill_value=0))
            pen = pen.div(tx_seg, axis=0) * 100
            top_items = dseg.groupby("Product_Name")["Line_Total"].sum().sort_values(ascending=False).head(12).index
            z = pen[top_items]
            hm = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, colorscale=BLUE_SCALE, colorbar=dict(title="% transaksi"),
                                      hovertemplate="%{y}<br>%{x}: %{z:.1f}% transaksi<extra></extra>"))
            style_fig(hm, "Penetrasi Produk per Segmen (% transaksi segmen yang memuat produk)", 400, legend=False)
            hm.update_xaxes(tickangle=-35)
            show_plot(hm, key="db_bonus_hm")
            idx = pen.div(pen.mean(axis=0), axis=1)
            rows = []
            for s in idx.index:
                tops = idx.loc[s].sort_values(ascending=False).head(3)
                row = {"Segmen": s, "Produk Paling Khas (indeks vs rata-rata)": ", ".join(f"{p} ({v:.2f}×)" for p, v in tops.items())}
                sub = dseg[dseg["Segmen"] == s]
                if sub["Transaction_ID"].nunique() >= 200:
                    _, rl, _ = run_apriori(sub, max(st.session_state.min_sup, 0.01), st.session_state.min_conf)
                    if len(rl):
                        r0 = rl.iloc[0]
                        row["Aturan Apriori Terkuat di Segmen"] = f"{' + '.join(sorted(r0['antecedents']))} → {' + '.join(sorted(r0['consequents']))} (lift {r0['lift']:.2f})"
                    else:
                        row["Aturan Apriori Terkuat di Segmen"] = "-"
                else:
                    row["Aturan Apriori Terkuat di Segmen"] = "Transaksi segmen terlalu sedikit"
                rows.append(row)
            show_df(pd.DataFrame(rows), fmt=False)
            st.caption("Indeks > 1 berarti produk dibeli lebih sering oleh segmen tersebut dibanding rata-rata segmen lain → bahan penawaran yang dipersonalisasi.")

    with tabs[4]:
        st.write(f"**Data transaksi terfilter ({fmt_id(len(df_f))} baris) — 500 baris pertama:**")
        show_df(df_f.drop(columns=["Catatan_Perubahan"], errors="ignore"), height=360, max_rows=500)
        st.download_button("⬇️ Unduh data terfilter (.csv)", csv_bytes(df_f), "Data_Dashboard_Terfilter.csv", "text/csv")
    st.caption("📸 Tips: gunakan fitur tangkapan layar pada tiap tab sebagai dokumentasi dashboard untuk laporan.")

# ============================================================
# MENU 6 — REKOMENDASI
# ============================================================
elif menu.startswith("6."):
    topbar("Tahap 6 · Soal 10", "Rekomendasi Keputusan Bisnis Preskriptif",
           "Strategi penempatan produk, retensi, prioritas tindakan, estimasi dampak, rencana implementasi, dan KPI evaluasi.", "10%")
    need_clean()
    df_c = st.session_state.data_clean
    rfm, ref_date = compute_rfm(df_c)
    ev = kmeans_eval(rfm, 2, 8)
    k_use = st.session_state.k_final or recommended_k(ev)
    seg, prof, _ = kmeans_fit(rfm, k_use)
    freq, rules, n_tx = run_apriori(df_c, st.session_state.min_sup, st.session_state.min_conf)

    A = lambda r: " + ".join(sorted(r["antecedents"])); B = lambda r: " + ".join(sorted(r["consequents"]))
    pairs, seen = [], set()
    for _, r in rules.iterrows():
        key = frozenset(set(r["antecedents"]) | set(r["consequents"]))
        if key not in seen:
            seen.add(key); pairs.append(r)
        if len(pairs) == 3:
            break
    total_rev = df_c["Line_Total"].sum()
    loyal = prof[prof["Segmen"].str.startswith("Loyal")].sort_values("Monetary_sum", ascending=False)
    churn = prof[prof["Segmen"].str.startswith("Churn")].sort_values("Monetary_sum", ascending=False)
    pot = prof[prof["Segmen"].str.startswith("Potential")].sort_values("Monetary_sum", ascending=False)
    L = loyal.iloc[0] if len(loyal) else prof.sort_values("Monetary_mean", ascending=False).iloc[0]
    C = churn.iloc[0] if len(churn) else prof.sort_values("Recency_mean", ascending=False).iloc[0]

    section("🧮", "Asumsi Estimasi Dampak", "Ubah asumsi di bawah ini — hasil estimasi akan menyesuaikan. Semua angka pada bagian ini adalah ESTIMASI, bukan fakta.")
    a1, a2, a3 = st.columns(3)
    with a1:
        adopt = st.slider("Adopsi bundling (% pembeli A yang belum beli B)", 1, 30, 8) / 100
        disc = st.slider("Diskon bundling pada produk B (%)", 0, 20, 5) / 100
    with a2:
        vip_gain = st.slider("Penurunan churn segmen Loyal (poin %)", 1, 20, 5) / 100
        reward = st.slider("Biaya reward loyalitas (% belanja segmen Loyal)", 0, 10, 2) / 100
    with a3:
        win = st.slider("Tingkat reaktivasi segmen Churn Risk (%)", 1, 30, 10) / 100
        vouch = st.slider("Biaya voucher (% dari belanja rata-rata pelanggan)", 1, 30, 10) / 100

    # --- hitung estimasi ---
    ben1 = cost1 = 0.0
    ev1 = []
    for r in pairs:
        n_a = r["antecedent support"] * n_tx
        b_items = list(r["consequents"])
        avg_b = df_c[df_c["Product_Name"].isin(b_items)]["Line_Total"].mean()
        new_adopters = n_a * (1 - r["confidence"]) * adopt
        ben1 += new_adopters * avg_b
        cost1 += (new_adopters + n_a * r["confidence"]) * avg_b * disc
        ev1.append(f"{A(r)} → {B(r)}")
    ben2 = L["Monetary_sum"] * vip_gain
    cost2 = L["Monetary_sum"] * reward
    ben3 = C["Jumlah"] * win * C["Monetary_mean"]
    cost3 = C["Jumlah"] * win * C["Monetary_mean"] * vouch + C["Jumlah"] * 2000
    est = pd.DataFrame([
        {"Rekomendasi": "R1 Bundling & penempatan rak", "Estimasi Manfaat (Rp/tahun)": ben1, "Estimasi Biaya (Rp/tahun)": cost1},
        {"Rekomendasi": "R2 Retensi pelanggan Loyal", "Estimasi Manfaat (Rp/tahun)": ben2, "Estimasi Biaya (Rp/tahun)": cost2},
        {"Rekomendasi": "R3 Win-back Churn Risk", "Estimasi Manfaat (Rp/tahun)": ben3, "Estimasi Biaya (Rp/tahun)": cost3},
    ])
    est["Estimasi Net (Rp/tahun)"] = est["Estimasi Manfaat (Rp/tahun)"] - est["Estimasi Biaya (Rp/tahun)"]
    est["% terhadap Pendapatan Tahun Data"] = est["Estimasi Net (Rp/tahun)"] / total_rev * 100

    section("10", "Tiga Rekomendasi Prioritas", "Soal 10 a–d · Bukti data, tindakan, KPI, biaya/manfaat, asumsi, dan cara evaluasi")

    def rec_card(pr, title, rows):
        body = "".join(f"<tr><td class='k'>{k}</td><td>{v}</td></tr>" for k, v in rows)
        H(f'<div class="rec"><div class="rh"><span class="rp">{pr}</span><span class="rt">{title}</span></div><table>{body}</table></div>')

    bund = "; ".join(f"{A(r)} → {B(r)} (lift {r['lift']:.2f}, conf {r['confidence'] * 100:.0f}%)" for r in pairs) or "Belum ada aturan — turunkan parameter Apriori di Menu 3"
    rec_card("R1", "Bundling & penempatan produk berdekatan (a)", [
        ("Bukti data (fakta)", f"Aturan Apriori terkuat: {bund}."),
        ("Tindakan", "Buat paket bundling dari ketiga pasangan produk dan letakkan produk pasangan berdampingan di rak/etalase; tampilkan rekomendasi kasir 'pelanggan yang membeli A biasanya juga membeli B'."),
        ("KPI keberhasilan", "Attach rate pasangan produk, rata-rata basket size, nilai rata-rata transaksi."),
        ("Estimasi biaya / manfaat", f"Manfaat ±{rp(ben1)}, biaya diskon ±{rp(cost1)}, net ±{rp(ben1 - cost1)} per tahun (estimasi)."),
        ("Asumsi", f"{adopt * 100:.0f}% pembeli A yang belum membeli B mengadopsi bundling; diskon {disc * 100:.0f}% diberikan pada produk B untuk semua pembeli paket."),
        ("Cara evaluasi", "Uji A/B: toko/rak dengan bundling vs tanpa bundling selama 4–8 minggu; bandingkan attach rate dan basket size."),
    ])
    rec_card("R2", f"Retensi pelanggan bernilai tinggi — {L['Segmen']} (b)", [
        ("Bukti data (fakta)", f"{fmt_id(L['Jumlah'])} pelanggan ({L['% Pelanggan']:.1f}%) menyumbang {L['% Kontribusi Pendapatan']:.1f}% pendapatan ({rp(L['Monetary_sum'])}); Recency rata-rata {L['Recency_mean']:.0f} hari, Frequency {L['Frequency_mean']:.1f}."),
        ("Tindakan", "Program loyalitas bertingkat: poin, diskon member, akses awal promo, layanan prioritas."),
        ("KPI keberhasilan", "Retention rate segmen, frekuensi belanja/bulan, share of wallet."),
        ("Estimasi biaya / manfaat", f"Manfaat ±{rp(ben2)}, biaya reward ±{rp(cost2)}, net ±{rp(ben2 - cost2)} per tahun (estimasi)."),
        ("Asumsi", f"Program menurunkan kehilangan pelanggan segmen ini sebesar {vip_gain * 100:.0f} poin; biaya reward {reward * 100:.0f}% dari belanja segmen."),
        ("Cara evaluasi", "Bandingkan kohort peserta program vs non-peserta (retention 3 & 6 bulan) pada segmen yang sama."),
    ])
    rec_card("R3", f"Kampanye win-back — {C['Segmen']} (b)", [
        ("Bukti data (fakta)", f"{fmt_id(C['Jumlah'])} pelanggan dengan Recency rata-rata {C['Recency_mean']:.0f} hari (jauh di atas rata-rata seluruh pelanggan {rfm['Recency'].mean():.0f} hari) dan nilai belanja historis {rp(C['Monetary_sum'])}."),
        ("Tindakan", "Voucher personal via WhatsApp/SMS berisi produk favorit terakhir, batas waktu 14 hari, plus survei singkat alasan berhenti."),
        ("KPI keberhasilan", "Reactivation rate, pendapatan hasil kampanye, ROI kampanye."),
        ("Estimasi biaya / manfaat", f"Manfaat ±{rp(ben3)}, biaya ±{rp(cost3)}, net ±{rp(ben3 - cost3)} (estimasi)."),
        ("Asumsi", f"{win * 100:.0f}% pelanggan segmen kembali bertransaksi senilai rata-rata historis; voucher {vouch * 100:.0f}% dari belanja rata-rata + biaya kirim pesan Rp 2.000/pelanggan."),
        ("Cara evaluasi", "Kelompok kontrol (hold-out 20% pelanggan tanpa voucher); ukur selisih reaktivasi dalam 30–60 hari."),
    ])

    section("⚖", "Prioritas Tindakan", "Soal 10 c · Urutan berdasarkan urgensi, potensi manfaat, biaya, dan kemudahan penerapan (skala 1–5, penilaian analis)")
    manfaat_rank = est["Estimasi Net (Rp/tahun)"].rank(method="first").astype(int)  # 1 = terkecil
    pr = pd.DataFrame({
        "Rekomendasi": est["Rekomendasi"],
        "Urgensi": [3, 4, 5],
        "Potensi Manfaat": [min(5, int(round(1 + 4 * (v - 1) / max(len(est) - 1, 1)))) for v in manfaat_rank],
        "Biaya (5 = murah)": [4, 3, 3],
        "Kemudahan": [4, 3, 4],
    })
    pr["Skor"] = 0.30 * pr["Urgensi"] + 0.35 * pr["Potensi Manfaat"] + 0.15 * pr["Biaya (5 = murah)"] + 0.20 * pr["Kemudahan"]
    pr = pr.sort_values("Skor", ascending=False).reset_index(drop=True)
    pr.insert(0, "Peringkat", range(1, len(pr) + 1))
    show_df(pr, fmt=False)
    st.caption("Bobot skor: urgensi 30% · potensi manfaat 35% · biaya 15% · kemudahan 20%. Potensi manfaat mengikuti peringkat estimasi net; skor lainnya adalah penilaian analis dan dapat disesuaikan.")

    section("💸", "Estimasi Dampak Bisnis", "Soal 10 d · Seluruh angka = estimasi berbasis asumsi di atas")
    show_df(est)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Manfaat", x=est["Rekomendasi"], y=est["Estimasi Manfaat (Rp/tahun)"], marker_color="#10b981"))
    fig.add_trace(go.Bar(name="Biaya", x=est["Rekomendasi"], y=est["Estimasi Biaya (Rp/tahun)"], marker_color="#f43f5e"))
    fig.add_trace(go.Bar(name="Net", x=est["Rekomendasi"], y=est["Estimasi Net (Rp/tahun)"], marker_color="#2563eb"))
    style_fig(fig, "Estimasi Manfaat vs Biaya per Rekomendasi (Rp)", 380)
    fig.update_layout(barmode="group")
    show_plot(fig, key="rec_est")

    H("""
    <div class="three">
        <div class="t1"><b>✅ Temuan faktual (dari data)</b><ul>
            <li>Aturan asosiasi & nilai support/confidence/lift</li><li>Profil RFM & kontribusi pendapatan tiap segmen</li><li>Tren dan komposisi penjualan tahun data</li></ul></div>
        <div class="t2"><b>🧪 Estimasi berbasis asumsi</b><ul>
            <li>Tingkat adopsi bundling, penurunan churn, reaktivasi</li><li>Biaya diskon, reward, dan voucher</li><li>Seluruh angka manfaat/biaya Rupiah di atas</li></ul></div>
        <div class="t3"><b>📏 Dampak aktual (belum diketahui)</b><ul>
            <li>Baru dapat diukur setelah implementasi</li><li>Wajib diuji dengan A/B test atau kelompok kontrol</li><li>Kenaikan penjualan tidak boleh diklaim sebagai fakta sebelum terukur</li></ul></div>
    </div>
    """)

    section("🗓", "Rencana Implementasi & KPI Evaluasi", "Soal 10 e–f")
    plan = pd.DataFrame([
        {"Horizon": "Jangka pendek (0–1 bulan)", "Tindakan": "Susun paket bundling dari top aturan; atur ulang rak pasangan produk; siapkan daftar pelanggan Churn Risk; tentukan kelompok kontrol.", "PIC": "Manajer Toko & Tim Data"},
        {"Horizon": "Jangka pendek (1–2 bulan)", "Tindakan": "Luncurkan voucher win-back (hold-out 20%) dan uji A/B bundling pada sebagian rak/toko.", "PIC": "Pemasaran"},
        {"Horizon": "Jangka menengah (2–6 bulan)", "Tindakan": "Luncurkan program loyalitas bertingkat; evaluasi hasil uji; perluas bundling yang berhasil.", "PIC": "Pemasaran & Operasional"},
        {"Horizon": "Jangka menengah (6 bulan+)", "Tindakan": "Jalankan ulang RFM & Apriori tiap kuartal untuk memperbarui segmen dan aturan; otomatisasi dashboard.", "PIC": "Tim Data / BI"},
    ])
    show_df(plan, fmt=False)
    kp = pd.DataFrame([
        {"KPI": "Total pendapatan & tren bulanan", "Sumber": "Dashboard (Menu 5)", "Frekuensi": "Bulanan"},
        {"KPI": "Rata-rata basket size & attach rate bundling", "Sumber": "Dashboard + Apriori", "Frekuensi": "Mingguan"},
        {"KPI": "Retention rate segmen Loyal", "Sumber": "RFM kuartalan", "Frekuensi": "Kuartalan"},
        {"KPI": "Reactivation rate & ROI voucher", "Sumber": "Kampanye vs kontrol", "Frekuensi": "30–60 hari pasca-kampanye"},
        {"KPI": "Jumlah pelanggan berpindah ke segmen lebih bernilai", "Sumber": "K-Means periodik", "Frekuensi": "Kuartalan"},
    ])
    show_df(kp, fmt=False)
    st.success("✅ Analisis UAS Business Intelligence & Data Mining selesai dan siap dipresentasikan.")

H('<div class="footer">Sistem Informasi Analisis Data · UAS Business Intelligence · Theresa Ivanna Oki (51230079) · Universitas Timor</div>')
