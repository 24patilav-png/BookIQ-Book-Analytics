"""
BookIQ - Book Price & Rating Analysis Dashboard  (UI build 2.0)

Run:  python -m streamlit run app.py
Data: BookIQ_Updated_Dataset_With_Availability.csv (same folder as this file)
"""
from __future__ import annotations

import html
import math
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BUILD = "UI build 2.0"
APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "BookIQ_Updated_Dataset_With_Availability.csv"
REQUIRED_COLS = ["Title", "Price", "Rating", "Availability"]
TECH_COLS = ["Image URL", "Image Name", "Image Extension"]
DEFAULT_GBP_INR = 128.00  # manually configured reference (mid-Oct 2026), NOT a live rate
AVAIL_VALUES = ["In stock", "Out of stock"]

ACCENT = "#7c8cff"
TEAL = "#3ccfb0"
RED = "#ef6b7b"
MUTED = "#8a93ad"

PAGES = [
    ("Overview", "◧  Overview"),
    ("Discovery", "⌕  Book Discovery"),
    ("Compare", "⇄  Compare Books"),
    ("Analytics", "▤  Analytics"),
    ("Lab", "∆  Hypothesis Lab"),
    ("Data", "▦  Data & Quality"),
]

st.set_page_config(page_title="BookIQ · Analytics", page_icon="📚", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family:'Inter',system-ui,sans-serif; }
.stApp { background: radial-gradient(1000px 500px at 85% -10%, rgba(124,140,255,.10), transparent 60%), #0b0d14; color:#e8eaf2; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"] { right:1rem; }
.block-container { max-width:1400px; padding:1.4rem 2.2rem 3rem; }
h1,h2,h3,h4 { font-family:'Manrope',sans-serif!important; letter-spacing:-.02em; color:#f4f5fb; }
p, li, label, span { color:#c4c9da; }
hr { border-color:rgba(255,255,255,.07); }

/* sidebar = app rail */
[data-testid="stSidebar"] { background:#0f1220; border-right:1px solid rgba(255,255,255,.06); min-width:290px; }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top:1rem; }
.brand { display:flex; gap:12px; align-items:center; padding:4px 4px 18px; }
.brand-logo { width:40px;height:40px;border-radius:12px;display:grid;place-items:center;font-size:20px;
  background:linear-gradient(135deg,#7c8cff,#4b5df0); box-shadow:0 8px 22px rgba(92,108,255,.35); }
.brand-name { font-family:'Manrope',sans-serif; font-weight:800; font-size:20px; color:#fff; line-height:1; }
.brand-sub { font-size:11px; color:#7f89a6; margin-top:3px; }
.rail-label { font-size:10px; letter-spacing:.16em; text-transform:uppercase; color:#68718c; font-weight:700; margin:16px 0 8px 4px; }
[data-testid="stSidebar"] div[role="radiogroup"] { gap:4px; }
[data-testid="stSidebar"] div[role="radiogroup"] label { width:100%; padding:10px 12px; border-radius:10px; border:1px solid transparent; cursor:pointer; transition:background .15s; }
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display:none; }
[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size:14px; font-weight:500; color:#9aa3bd; }
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background:rgba(255,255,255,.04); }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background:rgba(124,140,255,.14); border-color:rgba(124,140,255,.35); }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color:#fff; font-weight:600; }
.rail-foot { font-size:11px; color:#6c7592; line-height:1.55; margin-top:14px; padding:12px; border-radius:12px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.05); }

/* page header */
.page-head { display:flex; justify-content:space-between; align-items:flex-end; gap:20px; margin:2px 0 20px; flex-wrap:wrap; }
.page-title { font-family:'Manrope',sans-serif; font-size:30px; font-weight:800; color:#fff; letter-spacing:-.03em; line-height:1.1; }
.page-sub { color:#8791ad; font-size:14px; margin-top:6px; max-width:720px; }
.chips { display:flex; gap:8px; flex-wrap:wrap; }
.chip { font-size:11px; font-weight:600; padding:6px 11px; border-radius:999px; background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); color:#aab2cb; white-space:nowrap; }
.chip.warn { background:rgba(245,184,75,.09); border-color:rgba(245,184,75,.28); color:#f2c46d; }

/* panels (st.container(key="panel_*")) */
[class*="st-key-panel"] { background:linear-gradient(160deg,#141829,#101320); border:1px solid rgba(255,255,255,.07); border-radius:18px; padding:20px 22px; box-shadow:0 10px 30px rgba(0,0,0,.18); }
.ptitle { font-family:'Manrope',sans-serif; font-weight:700; font-size:15px; color:#f1f3fb; margin:0; }
.psub { font-size:12px; color:#7f89a6; margin:2px 0 10px; }

/* KPI strip */
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:18px; }
@media (max-width:1000px){ .kpis{grid-template-columns:repeat(2,1fr);} }
.kpi { position:relative; padding:18px 20px; border-radius:16px; background:linear-gradient(160deg,#161a2c,#10131f); border:1px solid rgba(255,255,255,.07); overflow:hidden; }
.kpi:after { content:""; position:absolute; right:-30px; top:-30px; width:100px; height:100px; border-radius:50%; background:radial-gradient(circle,rgba(124,140,255,.18),transparent 70%); }
.kpi-l { font-size:11px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:#7f89a6; }
.kpi-v { font-family:'Manrope',sans-serif; font-size:30px; font-weight:800; color:#fff; margin:8px 0 2px; letter-spacing:-.03em; }
.kpi-n { font-size:12px; color:#7f89a6; }

/* insights */
.ins { display:flex; gap:12px; padding:11px 0; border-bottom:1px solid rgba(255,255,255,.05); }
.ins:last-child { border-bottom:none; }
.ins-dot { flex:0 0 8px; height:8px; border-radius:50%; margin-top:7px; background:#7c8cff; }
.ins-t { font-size:13.5px; color:#d6dbea; line-height:1.55; }
.ins-t b { color:#fff; }

/* book cards */
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:14px; margin:10px 0; }
.bcard { display:flex; gap:12px; padding:14px; border-radius:16px; background:linear-gradient(160deg,#151929,#10131f); border:1px solid rgba(255,255,255,.07); min-height:132px; }
.bcard:hover { border-color:rgba(124,140,255,.4); }
.bcover { flex:0 0 64px; height:96px; border-radius:8px; background:linear-gradient(135deg,#1c2136,#252b46); display:grid; place-items:center; font-size:22px; overflow:hidden; }
.bcover img { width:100%; height:100%; object-fit:cover; }
.bbody { display:flex; flex-direction:column; min-width:0; flex:1; }
.btitle { font-weight:600; font-size:13.5px; color:#f1f3fb; line-height:1.35; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.bmeta { margin-top:auto; display:flex; justify-content:space-between; align-items:flex-end; gap:6px; }
.bprice { font-family:'Manrope',sans-serif; font-size:19px; font-weight:800; color:#fff; }
.stars { color:#f5b84b; font-size:12px; letter-spacing:1px; }
.stars .off { color:#3a4058; }
.badge { font-size:10.5px; font-weight:600; padding:3px 8px; border-radius:999px; }
.badge.in { background:rgba(60,207,176,.12); color:#52dcc0; border:1px solid rgba(60,207,176,.3); }
.badge.out { background:rgba(239,107,123,.1); color:#f48a97; border:1px solid rgba(239,107,123,.3); }
.bid { font-size:10.5px; color:#667090; }

/* compare */
.cmp { padding:22px; border-radius:18px; background:linear-gradient(160deg,#161a2c,#10131f); border:1px solid rgba(255,255,255,.08); height:100%; }
.cmp.win { border-color:rgba(60,207,176,.55); box-shadow:0 0 0 1px rgba(60,207,176,.18), 0 12px 34px rgba(60,207,176,.07); }
.cmp-tag { font-size:10px; letter-spacing:.16em; font-weight:700; color:#7c8cff; text-transform:uppercase; }
.cmp-name { font-family:'Manrope',sans-serif; font-size:19px; font-weight:700; color:#fff; margin:6px 0 14px; line-height:1.3; }
.cmp-row { display:flex; justify-content:space-between; padding:9px 0; border-top:1px solid rgba(255,255,255,.06); font-size:13.5px; }
.cmp-row span:first-child { color:#7f89a6; }
.cmp-row span:last-child { color:#f1f3fb; font-weight:600; text-align:right; }
.win-flag { display:inline-block; margin-left:8px; font-size:10px; padding:3px 8px; border-radius:999px; background:rgba(60,207,176,.14); color:#52dcc0; border:1px solid rgba(60,207,176,.35); letter-spacing:.06em; vertical-align:middle; }
.delta { padding:18px 14px; text-align:center; border-radius:18px; background:rgba(255,255,255,.025); border:1px dashed rgba(255,255,255,.1); }
.delta-l { font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:#7f89a6; font-weight:700; margin-top:10px; }
.delta-l:first-child { margin-top:0; }
.delta-v { font-family:'Manrope',sans-serif; font-weight:800; font-size:20px; color:#fff; }
.verdict { padding:18px 20px; border-radius:16px; border:1px solid rgba(124,140,255,.3); background:rgba(124,140,255,.07); font-size:14px; color:#dfe3f5; line-height:1.65; }
.verdict.good { border-color:rgba(60,207,176,.4); background:rgba(60,207,176,.07); }
.verdict.neutral { border-color:rgba(245,184,75,.35); background:rgba(245,184,75,.06); }
.verdict b { color:#fff; }

/* stats lab */
.hyp { font-size:13.5px; color:#cfd4e6; line-height:1.7; }
.hyp code { background:rgba(255,255,255,.06); padding:2px 7px; border-radius:6px; color:#dfe3f5; }
.result-big { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:8px 0 6px; }
.rb { padding:14px 16px; border-radius:14px; background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.07); }
.rb-l { font-size:10px; letter-spacing:.12em; text-transform:uppercase; color:#7f89a6; font-weight:700; }
.rb-v { font-family:'Manrope',sans-serif; font-weight:800; font-size:22px; color:#fff; margin-top:4px; }
.decision { padding:14px 18px; border-radius:14px; font-size:14px; line-height:1.65; color:#e4e7f4; }
.decision.sig { background:rgba(60,207,176,.08); border:1px solid rgba(60,207,176,.35); }
.decision.ns { background:rgba(245,184,75,.07); border:1px solid rgba(245,184,75,.3); }

/* inputs */
.stButton>button, .stDownloadButton>button { border-radius:10px; font-weight:600; border:1px solid rgba(124,140,255,.4); background:rgba(124,140,255,.12); color:#e6e9ff; }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:#9aa6ff; background:rgba(124,140,255,.22); color:#fff; }
[data-testid="stDataFrame"] { border:1px solid rgba(255,255,255,.07); border-radius:12px; overflow:hidden; }
[data-testid="stMetric"] { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.06); padding:12px 14px; border-radius:12px; }
button[data-baseweb="tab"] { font-weight:600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def esc(x) -> str:
    return html.escape(str(x), quote=True)


def inr(x: float, decimals: int = 2) -> str:
    """Format a number as ₹ with Indian-style digit grouping (12,34,567.89)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    neg = x < 0
    s = f"{abs(x):.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    out = "₹" + whole + (("." + frac) if decimals else "")
    return ("−" if neg else "") + out


def stars_html(r: float) -> str:
    n = int(round(r))
    return '<span class="stars">' + "★" * n + '<span class="off">' + "★" * (5 - n) + "</span></span>"


def badge_html(a: str) -> str:
    cls = "in" if a == "In stock" else "out"
    return f'<span class="badge {cls}">{esc(a)}</span>'


def call_stretch(fn, *args, **kwargs):
    """Use the new width='stretch' API when available, else the legacy flag."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except (TypeError, st.errors.StreamlitAPIException):
        return fn(*args, use_container_width=True, **kwargs)


def page_header(title: str, sub: str, chips: list[tuple[str, bool]] | None = None):
    chips_html = "".join(f'<span class="chip{" warn" if w else ""}">{esc(t)}</span>' for t, w in (chips or []))
    st.markdown(
        f'<div class="page-head"><div><div class="page-title">{esc(title)}</div>'
        f'<div class="page-sub">{esc(sub)}</div></div><div class="chips">{chips_html}</div></div>',
        unsafe_allow_html=True,
    )


def panel_title(t: str, s: str = ""):
    st.markdown(f'<p class="ptitle">{esc(t)}</p><p class="psub">{esc(s)}</p>', unsafe_allow_html=True)


def effect_label(v: float, kind: str) -> str:
    v = abs(v)
    cuts = {"d": (0.2, 0.5, 0.8), "eta": (0.01, 0.06, 0.14), "r": (0.1, 0.3, 0.5)}[kind]
    return ["negligible", "small", "medium", "large"][sum(v >= c for c in cuts)]


# ----------------------------------------------------------------------------
# Data layer  (single source of truth: every page reads the same frame)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str, mtime: float, source_currency: str, gbp_inr: float) -> pd.DataFrame:
    """mtime, currency and rate are part of the cache key, so editing the CSV or the
    exchange rate always produces freshly computed data (no stale cache)."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df["Title"] = df["Title"].astype(str).str.strip()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Availability"] = df["Availability"].astype(str).str.strip()
    df.insert(0, "ID", np.arange(1, len(df) + 1))
    factor = gbp_inr if source_currency == "GBP" else 1.0
    df["Price (₹)"] = (df["Price"] * factor).round(2)  # converted exactly once, here
    return df


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Quartile price bands (labels show the actual ₹ ranges)."""
    df = df.copy()
    valid = df["Price (₹)"].dropna()
    try:
        q = pd.qcut(valid, 4, duplicates="drop")
        labels = [f"{inr(iv.left, 0)}–{inr(iv.right, 0)}" for iv in q.cat.categories]
        df["Price Band"] = pd.qcut(df["Price (₹)"], 4, labels=labels, duplicates="drop").astype(str)
    except ValueError:
        df["Price Band"] = "All"
    return df


# ----------------------------------------------------------------------------
# Charts (Altair, transparent, dark)
# ----------------------------------------------------------------------------
def bar_chart(data: pd.DataFrame, x: str, y: str, *, money: bool = False, fmt: str = ",.0f",
              color: str = ACCENT, height: int = 250, sort=None, x_title: str = "", y_title: str = "",
              color_field: str | None = None, domain=None):
    d = data.copy()
    d["_lbl"] = d[y].map(lambda v: inr(v, 0) if money else f"{v:,.0f}")
    axis_y = alt.Axis(labelExpr="'₹' + format(datum.value, ',.0f')" if money else "format(datum.value, ',.0f')",
                      grid=True, gridColor="rgba(255,255,255,.06)", domain=False, title=y_title or None,
                      labelColor=MUTED, titleColor=MUTED, tickColor="transparent")
    axis_x = alt.Axis(labelAngle=0, labelColor="#c4c9da", title=x_title or None, titleColor=MUTED,
                      domain=False, tickColor="transparent", labelLimit=140)
    enc_color = (alt.Color(f"{color_field}:N", legend=None,
                           scale=alt.Scale(domain=domain, range=[TEAL, RED])) if color_field
                 else alt.value(color))
    base = alt.Chart(d).encode(x=alt.X(f"{x}:N", sort=sort, axis=axis_x),
                               y=alt.Y(f"{y}:Q", axis=axis_y))
    bars = base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=38).encode(color=enc_color)
    labels = base.mark_text(dy=-8, color="#e8eaf2", fontSize=11, fontWeight=600).encode(text="_lbl:N")
    chart = (bars + labels).properties(height=height).configure(background="transparent") \
        .configure_view(strokeWidth=0)
    call_stretch(st.altair_chart, chart, theme=None)


# ----------------------------------------------------------------------------
# Sidebar: brand, navigation, currency settings, data status
# ----------------------------------------------------------------------------
if "nav" not in st.session_state:
    st.session_state["nav"] = PAGES[0][1]

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-logo">📚</div><div><div class="brand-name">BookIQ</div>'
        '<div class="brand-sub">Price &amp; Rating Analytics</div></div></div>'
        '<div class="rail-label">Navigate</div>',
        unsafe_allow_html=True,
    )
    st.radio("Navigation", [p[1] for p in PAGES], key="nav", label_visibility="collapsed")
    st.markdown('<div class="rail-label">Currency settings</div>', unsafe_allow_html=True)
    src_cur_label = st.selectbox(
        "Currency of the CSV price column",
        ["GBP (£) → convert to INR", "INR (₹) → already in rupees"],
        help="books.toscrape.com lists prices in £, so GBP is the default. Choose INR only if you "
             "replace the CSV with one whose prices are already in rupees (prevents double conversion).",
    )
    SRC_CUR = "GBP" if src_cur_label.startswith("GBP") else "INR"
    RATE = st.number_input("INR per £1 (manually configured)", min_value=1.0, max_value=1000.0,
                           value=DEFAULT_GBP_INR, step=0.5, disabled=(SRC_CUR == "INR"),
                           help="Manual reference rate, not a live feed. Update it to the rate you want to cite.")
    if SRC_CUR == "GBP":
        st.caption(f"Prices = CSV £ × {RATE:g}. Manual rate, not live.")
    else:
        st.caption("No conversion applied.")

if not DATA_FILE.exists():
    st.error(f"Dataset not found at: {DATA_FILE}\n\nPut `{DATA_FILE.name}` in the same folder as app.py.")
    st.stop()

try:
    RAW = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime, SRC_CUR, float(RATE))
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load dataset: {exc}")
    st.stop()

# Clean analysis frame: rows with valid price & rating, bands attached. All pages use DF.
DF = add_bands(RAW.dropna(subset=["Price (₹)", "Rating"]))
DF["Rating"] = DF["Rating"].astype(int)
N_DROPPED = len(RAW) - len(DF)
PRICE = "Price (₹)"
PMIN, PMAX = float(DF[PRICE].min()), float(DF[PRICE].max())
MEDIAN_PRICE = float(DF[PRICE].median())
INSTOCK_PCT = float((DF["Availability"] == "In stock").mean() * 100)

with st.sidebar:
    st.markdown(
        f'<div class="rail-foot"><b style="color:#aab2cb">Dataset</b><br>{esc(DATA_FILE.name)}<br>'
        f'{len(RAW):,} rows loaded · {len(DF):,} analysed<br>'
        f'Availability is <b>simulated</b> in this dataset.<br><span style="opacity:.7">{BUILD}</span></div>',
        unsafe_allow_html=True,
    )
    if st.button("↻ Reload data & clear cache"):
        st.cache_data.clear()
        st.rerun()

CHIP_CUR = ([(f"£ → ₹ @ {RATE:g} (manual)", True)] if SRC_CUR == "GBP" else [("Prices in ₹", False)])


# ----------------------------------------------------------------------------
# PAGE: Overview
# ----------------------------------------------------------------------------
def page_overview():
    page_header("Overview", "A quick read on how the catalogue is priced and rated.",
                CHIP_CUR + [("Availability: simulated", True)])
    kpis = [
        ("Total books", f"{len(DF):,}", "rows analysed"),
        ("Average price", inr(DF[PRICE].mean()), f"median {inr(MEDIAN_PRICE)}"),
        ("Average rating", f"{DF['Rating'].mean():.2f} / 5", "source star rating"),
        ("Availability", f"{INSTOCK_PCT:.0f}% in stock", "simulated, not real stock"),
    ]
    st.markdown('<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="kpi-l">{esc(l)}</div><div class="kpi-v">{esc(v)}</div>'
        f'<div class="kpi-n">{esc(n)}</div></div>' for l, v, n in kpis) + "</div>", unsafe_allow_html=True)

    by_r = DF.groupby("Rating")[PRICE].agg(["mean", "median", "count"]).reset_index()
    by_r["Rating label"] = by_r["Rating"].astype(str) + "★"
    order = by_r["Rating label"].tolist()
    band_order = list(dict.fromkeys(DF.sort_values(PRICE)["Price Band"]))
    bands = DF["Price Band"].value_counts().reindex(band_order).reset_index()
    bands.columns = ["Price Band", "Books"]

    c1, c2 = st.columns([1, 1])
    with c1, st.container(key="panel_ov_1"):
        panel_title("Average price by rating", "Mean price (₹) of books at each star rating")
        bar_chart(by_r.rename(columns={"mean": "Avg price"}), "Rating label", "Avg price", money=True,
                  sort=order, height=230)
    with c2, st.container(key="panel_ov_2"):
        panel_title("Books per price band", "Quartile bands: each holds roughly 25% of books")
        bar_chart(bands, "Price Band", "Books", sort=band_order, color=TEAL, height=230)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    hi = by_r.loc[by_r["mean"].idxmax()]
    lo = by_r.loc[by_r["mean"].idxmin()]
    hi_grp = DF[DF.Rating >= 4][PRICE].mean()
    lo_grp = DF[DF.Rating <= 2][PRICE].mean()
    top5 = DF[DF.Rating == 5].nsmallest(1, PRICE)
    r5 = DF[DF.Rating == 5]
    ins = [
        f"<b>{int(hi['Rating'])}★ books have the highest average price</b> ({inr(hi['mean'])}); "
        f"<b>{int(lo['Rating'])}★</b> the lowest ({inr(lo['mean'])}). The spread between rating levels is "
        f"{inr(hi['mean'] - lo['mean'])}.",
        f"Books rated 4–5★ average <b>{inr(hi_grp)}</b> versus <b>{inr(lo_grp)}</b> for 1–2★ "
        f"({'higher' if hi_grp > lo_grp else 'lower or equal'} by {inr(abs(hi_grp - lo_grp))}). "
        "This is descriptive only. Use the Hypothesis Lab to check whether the difference is statistically significant.",
        f"<b>{len(r5):,} books are rated 5★</b>; the cheapest costs {inr(float(top5[PRICE].iloc[0]))} "
        f"(“{esc(top5['Title'].iloc[0])}”)." if len(top5) else "No 5★ books in the data.",
        f"<b>{INSTOCK_PCT:.0f}% of titles are marked In stock</b>, but availability in this dataset is simulated "
        "for demonstration, so treat it as a model input and not real inventory.",
    ]
    with st.container(key="panel_ov_3"):
        panel_title("Business insights", "Generated live from the data, so the numbers always match the charts")
        st.markdown("".join(f'<div class="ins"><div class="ins-dot"></div><div class="ins-t">{t}</div></div>'
                            for t in ins), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PAGE: Book Discovery
# ----------------------------------------------------------------------------
SORTS = {
    "Price: low → high": ([PRICE, "Title"], [True, True]),
    "Price: high → low": ([PRICE, "Title"], [False, True]),
    "Rating: high → low": (["Rating", PRICE], [False, True]),
    "Rating: low → high": (["Rating", PRICE], [True, True]),
    "Title: A → Z": (["Title"], [True]),
}


def _discovery_defaults():
    lo, hi = math.floor(PMIN), math.ceil(PMAX)
    if st.session_state.get("_bounds") != (lo, hi, SRC_CUR):  # currency/rate changed → reset slider
        st.session_state["_bounds"] = (lo, hi, SRC_CUR)
        st.session_state["d_price"] = (float(lo), float(hi))
    st.session_state.setdefault("d_q", "")
    st.session_state.setdefault("d_rating", (1, 5))
    st.session_state.setdefault("d_avail", list(AVAIL_VALUES))
    st.session_state.setdefault("d_sort", "Price: low → high")
    st.session_state.setdefault("d_view", "Cards")
    st.session_state.setdefault("d_psize", 12)
    st.session_state.setdefault("d_page", 1)
    st.session_state.setdefault("d_covers", False)


def _clear_filters():
    lo, hi, _ = st.session_state["_bounds"]
    st.session_state.update(d_q="", d_price=(float(lo), float(hi)), d_rating=(1, 5),
                            d_avail=list(AVAIL_VALUES), d_sort="Price: low → high", d_page=1)


def page_discovery():
    _discovery_defaults()
    page_header("Book Discovery", "Search the catalogue and narrow it down by price, rating and availability.",
                CHIP_CUR + [("Availability: simulated", True)])
    lo, hi, _ = st.session_state["_bounds"]

    with st.container(key="panel_dc_filters"):
        s1, s2, s3 = st.columns([3, 1.4, 1])
        s1.text_input("Search by title", key="d_q", placeholder="Type a title or keyword…")
        s2.selectbox("Sort by", list(SORTS), key="d_sort")
        s3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        s3.button("✕ Clear filters", on_click=_clear_filters, key="d_clear")
        f1, f2, f3 = st.columns([2, 1.4, 1.4])
        f1.slider("Price range (₹)", float(lo), float(hi), step=1.0, key="d_price")
        f2.slider("Rating (★)", 1, 5, key="d_rating")
        f3.multiselect("Availability (simulated)", AVAIL_VALUES, key="d_avail")

    q = st.session_state["d_q"].strip()
    pmin, pmax = st.session_state["d_price"]
    rmin, rmax = st.session_state["d_rating"]
    m = DF[PRICE].between(pmin, pmax) & DF["Rating"].between(rmin, rmax) & DF["Availability"].isin(st.session_state["d_avail"])
    if q:
        m &= DF["Title"].str.contains(q, case=False, regex=False, na=False)
    res = DF[m]
    by, asc = SORTS[st.session_state["d_sort"]]
    res = res.sort_values(by, ascending=asc, kind="mergesort")

    h1, h2, h3, h4 = st.columns([2.2, 1, 1, 1])
    h1.markdown(f"<div style='padding-top:10px;font-size:14px'><b style='color:#fff;font-size:18px'>{len(res):,}</b> "
                f"of {len(DF):,} books match</div>", unsafe_allow_html=True)
    h2.radio("View", ["Cards", "Table"], key="d_view", horizontal=True, label_visibility="collapsed")
    h3.toggle("Cover images", key="d_covers", help="Loads thumbnails from books.toscrape.com (needs internet).")
    h4.download_button("⇩ Export results", res[["Title", PRICE, "Rating", "Availability"]].to_csv(index=False).encode("utf-8-sig"),
                       "bookiq_filtered_results.csv", "text/csv")

    if res.empty:
        st.info("No books match these filters. Try widening the price range or clicking **Clear filters**.")
        return

    if st.session_state["d_view"] == "Table":
        show = res[["ID", "Title", PRICE, "Rating", "Availability"]].rename(columns={"Availability": "Availability (simulated)"})
        call_stretch(st.dataframe, show, hide_index=True, height=520, column_config={
            "Price (₹)": st.column_config.NumberColumn("Price (₹)", format="₹%.2f"),
            "Rating": st.column_config.NumberColumn("Rating", format="%d ★"),
        })
        return

    st.selectbox("Cards per page", [12, 24, 48], key="d_psize")
    size = st.session_state["d_psize"]
    pages = max(1, math.ceil(len(res) / size))
    st.session_state["d_page"] = min(max(1, st.session_state["d_page"]), pages)
    pc1, pc2 = st.columns([1, 5])
    pc1.number_input("Page", 1, pages, key="d_page")
    pc2.markdown(f"<div style='padding-top:32px;color:#7f89a6;font-size:13px'>of {pages} pages</div>", unsafe_allow_html=True)
    page_df = res.iloc[(st.session_state["d_page"] - 1) * size: st.session_state["d_page"] * size]
    cards = []
    for _, r in page_df.iterrows():
        cover = "📖"
        if st.session_state["d_covers"] and isinstance(r.get("Image URL"), str) and r["Image URL"].startswith("http"):
            cover = f'<img src="{esc(r["Image URL"])}" loading="lazy" referrerpolicy="no-referrer" onerror="this.style.display=\'none\'">'
        cards.append(
            f'<div class="bcard"><div class="bcover">{cover}</div><div class="bbody">'
            f'<div class="btitle" title="{esc(r["Title"])}">{esc(r["Title"])}</div>'
            f'<div style="margin:6px 0">{stars_html(r["Rating"])}</div>'
            f'<div class="bmeta"><div><div class="bprice">{inr(r[PRICE])}</div><div class="bid">ID #{int(r["ID"])}</div></div>'
            f'{badge_html(r["Availability"])}</div></div></div>')
    st.markdown('<div class="grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PAGE: Compare Books
# ----------------------------------------------------------------------------
def _book_label(i: int) -> str:
    r = DF.loc[DF["ID"] == i].iloc[0]
    return f"#{i} · {r['Title'][:60]} · {inr(r[PRICE], 0)} · {r['Rating']}★"


def book_input(side: str, default_idx: int) -> dict | None:
    """Returns a normalised book dict, or None if the input is invalid."""
    mode = st.radio(f"Book {side} source", ["From dataset", "Custom book"], key=f"c_mode_{side}", horizontal=True,
                    label_visibility="collapsed")
    if mode == "From dataset":
        ids = DF["ID"].tolist()
        bid = st.selectbox(f"Choose book {side}", ids, index=default_idx, key=f"c_id_{side}", format_func=_book_label)
        r = DF.loc[DF["ID"] == bid].iloc[0]
        return dict(side=side, name=r["Title"], price=float(r[PRICE]), rating=int(r["Rating"]),
                    avail=r["Availability"], avail_note="simulated", source="Dataset", id=int(bid))
    name = st.text_input("Book title", value=f"Custom book {side}", key=f"c_name_{side}").strip()
    price = st.number_input("Price (₹)", min_value=0.0, max_value=1_000_000.0, value=round(MEDIAN_PRICE, 0),
                            step=10.0, key=f"c_price_{side}")
    rating = st.slider("Rating (★)", 1, 5, 3, key=f"c_rating_{side}")
    avail = st.selectbox("Availability", AVAIL_VALUES, key=f"c_avail_{side}")
    if not name:
        st.warning("Enter a title for the custom book.")
        return None
    if price <= 0:
        st.warning("Enter a price greater than ₹0.")
        return None
    return dict(side=side, name=name, price=float(price), rating=int(rating), avail=avail,
                avail_note="user-entered", source="Custom", id=None)


def peer_context(b: dict) -> dict | None:
    peers = DF[(DF["Rating"] == b["rating"])]
    if b["id"] is not None:
        peers = peers[peers["ID"] != b["id"]]
    if len(peers) < 10:
        return None
    return dict(n=len(peers), cheaper=float((peers[PRICE] > b["price"]).mean() * 100), median=float(peers[PRICE].median()))


def book_card(b: dict, ctx, win: bool) -> str:
    flag = '<span class="win-flag">BETTER VALUE</span>' if win else ""
    peers = (f"cheaper than {ctx['cheaper']:.0f}% of {ctx['n']} other {b['rating']}★ books" if ctx else "n/a (too few peers)")
    return (f'<div class="cmp{" win" if win else ""}"><div class="cmp-tag">Book {b["side"]} · {b["source"]}</div>'
            f'<div class="cmp-name">{esc(b["name"])}{flag}</div>'
            f'<div class="cmp-row"><span>Price</span><span>{inr(b["price"])}</span></div>'
            f'<div class="cmp-row"><span>Rating</span><span>{stars_html(b["rating"])} {b["rating"]}/5</span></div>'
            f'<div class="cmp-row"><span>Availability ({b["avail_note"]})</span><span>{badge_html(b["avail"])}</span></div>'
            f'<div class="cmp-row"><span>Rating points per ₹1,000</span><span>{b["rating"] / b["price"] * 1000:.2f}</span></div>'
            f'<div class="cmp-row"><span>Price vs same-rating books</span><span style="font-size:12px">{esc(peers)}</span></div></div>')


def value_verdict(a: dict, b: dict) -> tuple[str, str, str | None]:
    """Transparent rule: a book is 'better value' only if it is at least as cheap AND at least
    as highly rated, and strictly better on one of them. Otherwise it is a trade-off."""
    dp = b["price"] - a["price"]
    dr = b["rating"] - a["rating"]
    same_p = abs(dp) < 0.005
    if same_p and dr == 0:
        return "neutral", "Identical on price and rating, so there is no value difference on the data available.", None
    a_dom = (a["price"] <= b["price"] + 0.005) and (a["rating"] >= b["rating"]) and not (same_p and dr == 0)
    b_dom = (b["price"] <= a["price"] + 0.005) and (b["rating"] >= a["rating"])
    if a_dom or b_dom:
        w, l = (a, b) if a_dom else (b, a)
        parts = []
        if abs(w["price"] - l["price"]) >= 0.005:
            parts.append(f"{inr(abs(w['price'] - l['price']))} cheaper")
        if w["rating"] != l["rating"]:
            parts.append(f"rated {abs(w['rating'] - l['rating'])} star(s) higher")
        txt = (f"<b>Book {w['side']} is the better value</b> on the data available: it is " + " and ".join(parts) +
               f" than Book {l['side']}.")
        if w["avail"] == "Out of stock":
            txt += (f" <i>Note: Book {w['side']} is marked Out of stock "
                    f"({w['avail_note']}), which is not part of the value rule.</i>")
        return "good", txt, w["side"]
    hi, lo = (a, b) if a["rating"] > b["rating"] else (b, a)
    extra = hi["price"] - lo["price"]
    per = extra / (hi["rating"] - lo["rating"])
    txt = (f"<b>No clear winner. This is a trade-off.</b> Book {hi['side']} is rated {hi['rating'] - lo['rating']} "
           f"star(s) higher but costs {inr(extra)} more ({inr(per)} per extra rating point). "
           f"Whether that is worth it depends on how much you value the rating difference.")
    return "neutral", txt, None


def page_compare():
    page_header("Compare Books", "Compare two dataset books, a custom book, or any mix. The verdict follows a transparent rule.",
                CHIP_CUR)
    ca, cb = st.columns(2)
    with ca, st.container(key="panel_cmp_a"):
        panel_title("Book A", "Pick from the dataset or enter your own")
        A = book_input("A", 0)
    with cb, st.container(key="panel_cmp_b"):
        panel_title("Book B", "Pick from the dataset or enter your own")
        B = book_input("B", 1)
    if A is None or B is None:
        st.info("Complete both books to see the comparison.")
        return
    if A["source"] == "Dataset" and B["source"] == "Dataset" and A["id"] == B["id"]:
        st.warning("Both selections are the same book. Choose two different books to compare.")

    st.markdown("&nbsp;", unsafe_allow_html=True)
    kind, text, winner = value_verdict(A, B)
    dp, dr = B["price"] - A["price"], B["rating"] - A["rating"]
    cA, cM, cB = st.columns([5, 2.6, 5])
    cA.markdown(book_card(A, peer_context(A), winner == "A"), unsafe_allow_html=True)
    sign = lambda v, f: ("+" if v > 0 else "") + f(v)  # noqa: E731
    cM.markdown(
        '<div class="delta"><div class="delta-l">Price difference (B − A)</div>'
        f'<div class="delta-v">{"+" if dp > 0 else ("−" if dp < 0 else "")}{inr(abs(dp))}</div>'
        '<div class="delta-l">Rating difference (B − A)</div>'
        f'<div class="delta-v">{sign(dr, lambda v: f"{v}★")}</div>'
        '<div class="delta-l">Availability</div>'
        f'<div style="font-size:12.5px;color:#c4c9da;margin-top:4px">A: {esc(A["avail"])}<br>B: {esc(B["avail"])}</div></div>',
        unsafe_allow_html=True)
    cB.markdown(book_card(B, peer_context(B), winner == "B"), unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown(f'<div class="verdict {kind}">{text}</div>', unsafe_allow_html=True)
    with st.expander("How is “value for money” assessed?"):
        st.markdown(
            "- **Rule:** a book is called *better value* only when it is **at least as cheap and at least as highly rated** "
            "as the other, and strictly better on one of the two. Otherwise the app reports a **trade-off** and never invents a winner.\n"
            "- **Rating points per ₹1,000** and the **same-rating price percentile** are shown for context but do not decide the verdict.\n"
            "- Ratings are 1–5 star values from the source site (not an average of many reviews), so small differences carry limited weight.\n"
            "- **Availability** in the dataset is simulated for demonstration; for custom books it is whatever you enter. It is displayed but not part of the value rule.")


# ----------------------------------------------------------------------------
# PAGE: Analytics
# ----------------------------------------------------------------------------
def page_analytics():
    page_header("Analytics", "Compact views of how price, rating and availability relate.",
                CHIP_CUR + [("Availability: simulated", True)])
    g = DF.groupby("Rating")
    summ = g[PRICE].agg(Books="count", Mean="mean", Median="median", Std="std", Min="min", Max="max").reset_index()
    summ["Rating label"] = summ["Rating"].astype(str) + "★"
    order = summ["Rating label"].tolist()
    avail = (DF.assign(**{"Rating label": DF["Rating"].astype(str) + "★"})
             .groupby("Rating label")["Availability"].apply(lambda s: (s == "In stock").mean() * 100)
             .reindex(order).reset_index(name="In stock %"))
    band_order = list(dict.fromkeys(DF.sort_values(PRICE)["Price Band"]))
    band_rating = DF.groupby("Price Band")["Rating"].mean().reindex(band_order).reset_index(name="Avg rating")
    band_rating["Avg rating"] = band_rating["Avg rating"].round(2)

    c1, c2 = st.columns(2)
    with c1, st.container(key="panel_an_1"):
        panel_title("Median price by rating", "Median is less sensitive to outliers than the mean")
        bar_chart(summ.rename(columns={"Median": "Median price"}), "Rating label", "Median price", money=True,
                  sort=order, height=230)
    with c2, st.container(key="panel_an_2"):
        panel_title("Books per rating", "Distribution of star ratings")
        bar_chart(summ, "Rating label", "Books", sort=order, color=TEAL, height=230)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3, st.container(key="panel_an_3"):
        panel_title("Average rating by price band", "Do pricier books score higher? Quartile bands in ₹")
        d = band_rating.copy()
        d["_lbl"] = d["Avg rating"].map(lambda v: f"{v:.2f}")
        base = alt.Chart(d).encode(
            x=alt.X("Price Band:N", sort=band_order, axis=alt.Axis(labelAngle=0, labelColor="#c4c9da", title=None, domain=False, tickColor="transparent")),
            y=alt.Y("Avg rating:Q", scale=alt.Scale(domain=[0, 5]), axis=alt.Axis(grid=True, gridColor="rgba(255,255,255,.06)", domain=False, labelColor=MUTED, title="Average rating (★)", titleColor=MUTED, tickColor="transparent")))
        ch = (base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=38, color=ACCENT)
              + base.mark_text(dy=-8, color="#e8eaf2", fontSize=11, fontWeight=600).encode(text="_lbl:N")
              ).properties(height=230).configure(background="transparent").configure_view(strokeWidth=0)
        call_stretch(st.altair_chart, ch, theme=None)
    with c4, st.container(key="panel_an_4"):
        panel_title("In-stock share by rating (simulated)", "Availability is simulated, so this chart shows the dataset's design, not real stock")
        d = avail.copy()
        d["In stock %"] = d["In stock %"].round(1)
        d["_lbl"] = d["In stock %"].map(lambda v: f"{v:.0f}%")
        base = alt.Chart(d).encode(
            x=alt.X("Rating label:N", sort=order, axis=alt.Axis(labelAngle=0, labelColor="#c4c9da", title=None, domain=False, tickColor="transparent")),
            y=alt.Y("In stock %:Q", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(grid=True, gridColor="rgba(255,255,255,.06)", domain=False, labelColor=MUTED, title="% In stock", titleColor=MUTED, tickColor="transparent")))
        ch = (base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=38, color="#5b6a9a")
              + base.mark_text(dy=-8, color="#e8eaf2", fontSize=11, fontWeight=600).encode(text="_lbl:N")
              ).properties(height=230).configure(background="transparent").configure_view(strokeWidth=0)
        call_stretch(st.altair_chart, ch, theme=None)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    with st.container(key="panel_an_5"):
        panel_title("Price summary by rating (₹)", "Count and descriptive statistics")
        show = summ[["Rating", "Books", "Mean", "Median", "Std", "Min", "Max"]].copy()
        cfg = {c: st.column_config.NumberColumn(c, format="₹%.2f") for c in ["Mean", "Median", "Std", "Min", "Max"]}
        cfg["Rating"] = st.column_config.NumberColumn("Rating", format="%d ★")
        call_stretch(st.dataframe, show, hide_index=True, column_config=cfg)
        e1, e2 = st.columns(2)
        e1.dataframe(DF.nlargest(5, PRICE)[["Title", PRICE, "Rating"]], hide_index=True,
                     column_config={PRICE: st.column_config.NumberColumn(PRICE, format="₹%.2f")})
        e1.caption("Five most expensive books")
        e2.dataframe(DF.nsmallest(5, PRICE)[["Title", PRICE, "Rating"]], hide_index=True,
                     column_config={PRICE: st.column_config.NumberColumn(PRICE, format="₹%.2f")})
        e2.caption("Five cheapest books")


# ----------------------------------------------------------------------------
# PAGE: Hypothesis Testing Lab
# ----------------------------------------------------------------------------
GROUP_VARS = {"Rating (★)": "Rating", "Availability (simulated)": "Availability"}
CAT_VARS = {"Rating (★)": "Rating", "Availability (simulated)": "Availability", "Price band (quartiles)": "Price Band"}
NUM_VARS = {"Price (₹)": PRICE, "Rating (★)": "Rating"}
SIM_NOTE = ("Availability in this dataset is **simulated**, so any result involving it demonstrates the method and does not "
            "describe real stock behaviour.")


def _result(title, variables, h0, h1, stat_name, stat, p, alpha, extras, warns, tables, interp, note_sim=False):
    return dict(title=title, variables=variables, h0=h0, h1=h1, stat_name=stat_name, stat=stat, p=p, alpha=alpha,
                extras=extras, warns=warns, tables=tables, interp=interp, sim=note_sim)


def run_welch(df, gcol, ga, gb, alpha):
    a = df.loc[df[gcol].astype(str) == str(ga), PRICE].dropna()
    b = df.loc[df[gcol].astype(str) == str(gb), PRICE].dropna()
    if ga == gb:
        return "Choose two different groups."
    if len(a) < 2 or len(b) < 2:
        return f"Not enough data: each group needs at least 2 observations (found {len(a)} and {len(b)})."
    if a.std(ddof=1) == 0 and b.std(ddof=1) == 0:
        return "Both groups have zero variance, so the t-statistic is undefined."
    r = stats.ttest_ind(a, b, equal_var=False)
    diff = a.mean() - b.mean()
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = diff / sp if sp > 0 else float("nan")
    try:
        ci = r.confidence_interval(1 - alpha)
        ci_txt = f"{inr(ci.low)} to {inr(ci.high)}"
    except Exception:  # noqa: BLE001
        ci_txt = "n/a"
    warns = []
    if min(len(a), len(b)) < 30:
        warns.append("At least one group has fewer than 30 observations, so the normality assumption matters more.")
    if abs(stats.skew(a)) > 1 or abs(stats.skew(b)) > 1:
        warns.append("Price is noticeably skewed in at least one group. Interpret the p-value with caution.")
    gl = {"Rating": lambda x: f"{x}★"}.get(gcol, lambda x: str(x))
    tbl = pd.DataFrame({"Group": [gl(ga), gl(gb)], "n": [len(a), len(b)], "Mean (₹)": [a.mean(), b.mean()],
                        "Std (₹)": [a.std(ddof=1), b.std(ddof=1)]})
    sig = r.pvalue < alpha
    interp = (f"Mean price is {inr(a.mean())} for {gl(ga)} and {inr(b.mean())} for {gl(gb)} (difference {inr(diff)}, "
              f"{int((1 - alpha) * 100)}% CI {ci_txt}). " +
              (f"Since p = {r.pvalue:.4f} < α = {alpha}, we reject H₀: the mean prices differ significantly. "
               if sig else
               f"Since p = {r.pvalue:.4f} ≥ α = {alpha}, we fail to reject H₀: there is not enough evidence of a difference in mean price "
               "(this does not prove the means are equal). ") +
              f"Effect size (Cohen's d) = {d:.2f}, which is {effect_label(d, 'd')}.")
    return _result("Welch's t-test (two independent groups, unequal variances)",
                   f"Price (₹) grouped by {gcol}: {gl(ga)} vs {gl(gb)}",
                   f"mean price({gl(ga)}) = mean price({gl(gb)})", f"mean price({gl(ga)}) ≠ mean price({gl(gb)})",
                   "t", r.statistic, r.pvalue, alpha,
                   [("Degrees of freedom", f"{r.df:.1f}"), ("Mean difference", inr(diff)), ("Cohen's d", f"{d:.2f}")],
                   warns, [("Group summary", tbl)], interp, note_sim=(gcol == "Availability"))


def run_anova(df, gcol, alpha):
    groups = {k: v[PRICE].dropna().values for k, v in df.groupby(gcol)}
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(groups) < 2:
        return "Not enough data: ANOVA needs at least 2 groups with 2+ observations each."
    arrs = list(groups.values())
    if all(np.var(a, ddof=1) == 0 for a in arrs):
        return "All groups have zero variance, so F is undefined."
    r = stats.f_oneway(*arrs)
    allv = np.concatenate(arrs)
    ssb = sum(len(a) * (a.mean() - allv.mean()) ** 2 for a in arrs)
    sst = ((allv - allv.mean()) ** 2).sum()
    eta = ssb / sst if sst > 0 else float("nan")
    dfb, dfw = len(arrs) - 1, len(allv) - len(arrs)
    warns = []
    try:
        lev = stats.levene(*arrs)
        if lev.pvalue < 0.05:
            warns.append(f"Levene's test suggests unequal variances (p = {lev.pvalue:.4f}); classic ANOVA assumes equal variances, so treat the result cautiously.")
    except Exception:  # noqa: BLE001
        pass
    if min(len(a) for a in arrs) < 30:
        warns.append("At least one group has fewer than 30 observations.")
    lab = (lambda k: f"{k}★") if gcol == "Rating" else (lambda k: str(k))
    tbl = pd.DataFrame({"Group": [lab(k) for k in groups], "n": [len(v) for v in groups.values()],
                        "Mean (₹)": [v.mean() for v in groups.values()], "Std (₹)": [v.std(ddof=1) for v in groups.values()]})
    tables = [("Group summary", tbl)]
    sig = r.pvalue < alpha
    if sig and len(arrs) > 2:
        try:
            tk = stats.tukey_hsd(*arrs)
            keys = list(groups)
            rows = [dict(Pair=f"{lab(keys[i])} vs {lab(keys[j])}", **{"Mean diff (₹)": arrs[i].mean() - arrs[j].mean(),
                    "p (Tukey)": tk.pvalue[i, j]}) for i in range(len(keys)) for j in range(i + 1, len(keys))]
            tables.append(("Post-hoc: Tukey HSD pairwise comparisons", pd.DataFrame(rows)))
        except Exception:  # noqa: BLE001
            pass
    interp = ((f"Since p = {r.pvalue:.4f} < α = {alpha}, we reject H₀: at least one group's mean price differs. "
               if sig else f"Since p = {r.pvalue:.4f} ≥ α = {alpha}, we fail to reject H₀: no significant difference in mean price across groups. ") +
              f"Eta-squared = {eta:.3f}, meaning {gcol} explains about {eta * 100:.1f}% of the price variance ({effect_label(eta, 'eta')} effect)."
              + (" The post-hoc table shows which pairs differ." if sig and len(arrs) > 2 else ""))
    return _result("One-way ANOVA", f"Price (₹) across {gcol} groups ({len(arrs)} groups)",
                   "all group mean prices are equal", "at least one group mean price differs",
                   "F", r.statistic, r.pvalue, alpha,
                   [("Degrees of freedom", f"{dfb}, {dfw}"), ("Eta²", f"{eta:.3f}"), ("Total n", f"{len(allv):,}")],
                   warns, tables, interp, note_sim=(gcol == "Availability"))


def run_pearson(df, xcol, ycol, alpha):
    if xcol == ycol:
        return "Choose two different variables."
    d = df[[xcol, ycol]].dropna()
    if len(d) < 3:
        return "Not enough data: Pearson correlation needs at least 3 paired observations."
    if d[xcol].nunique() < 2 or d[ycol].nunique() < 2:
        return "One variable is constant, so correlation is undefined."
    r, p = stats.pearsonr(d[xcol], d[ycol])
    rho, p_s = stats.spearmanr(d[xcol], d[ycol])
    warns = []
    if "Rating" in (xcol, ycol):
        warns.append("Rating is an ordinal 1–5 score, not a continuous measure; Pearson is shown as requested and Spearman is a more suitable cross-check.")
    sig = p < alpha
    direction = "positive" if r > 0 else "negative"
    interp = ((f"Since p = {p:.4f} < α = {alpha}, we reject H₀: there is a statistically significant {direction} linear relationship "
               if sig else f"Since p = {p:.4f} ≥ α = {alpha}, we fail to reject H₀: no significant linear relationship detected ") +
              f"(r = {r:.3f}, {effect_label(r, 'r')} strength; r² = {r * r:.3f}, so it accounts for {r * r * 100:.1f}% of the variance). "
              f"Spearman rank check: ρ = {rho:.3f} (p = {p_s:.4f}). Correlation does not imply causation.")
    xl = {PRICE: "Price (₹)", "Rating": "Rating"}
    return _result("Pearson correlation", f"{xl[xcol]} vs {xl[ycol]} (n = {len(d):,})",
                   "ρ = 0 (no linear correlation)", "ρ ≠ 0", "r", r, p, alpha,
                   [("n", f"{len(d):,}"), ("r²", f"{r * r:.3f}"), ("Spearman ρ", f"{rho:.3f}")], warns, [], interp)


def run_chi2(df, c1, c2, alpha):
    if c1 == c2:
        return "Choose two different categorical variables."
    ct = pd.crosstab(df[c1], df[c2])
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return "Not enough data: each variable needs at least 2 categories with observations."
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    n = ct.values.sum()
    v = math.sqrt(chi2 / (n * (min(ct.shape) - 1))) if n else float("nan")
    warns = []
    low = (exp < 5).mean() * 100
    if exp.min() < 1 or low > 20:
        warns.append(f"Chi-square assumption violated: {low:.0f}% of cells have an expected count below 5 (minimum {exp.min():.2f}). The p-value may be unreliable; consider combining categories.")
    elif exp.min() < 5:
        warns.append(f"Minimum expected cell count is {exp.min():.2f} (<5) but under 20% of cells are affected; result is acceptable with care.")
    if ct.shape == (2, 2):
        warns.append("2×2 table: Yates' continuity correction is applied.")
    sig = p < alpha
    tbl = ct.copy()
    tbl.index = tbl.index.astype(str)
    tbl.columns = tbl.columns.astype(str)
    interp = ((f"Since p = {p:.4f} < α = {alpha}, we reject H₀: {c1} and {c2} are statistically associated. "
               if sig else f"Since p = {p:.4f} ≥ α = {alpha}, we fail to reject H₀: no significant association between {c1} and {c2}. ") +
              f"Cramér's V = {v:.3f} ({effect_label(v, 'r')} association). Association does not imply causation.")
    return _result("Chi-square test of independence", f"{c1} × {c2} (n = {int(n):,})",
                   f"{c1} and {c2} are independent", f"{c1} and {c2} are associated",
                   "χ²", chi2, p, alpha, [("Degrees of freedom", f"{dof}"), ("Cramér's V", f"{v:.3f}"), ("n", f"{int(n):,}")],
                   warns, [("Observed counts", tbl.reset_index())], interp, note_sim=("Availability" in (c1, c2)))


def render_result(res: dict):
    sig = res["p"] < res["alpha"]
    st.markdown(f'<div class="ptitle" style="font-size:17px">{esc(res["title"])}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hyp"><b>Variables:</b> {esc(res["variables"])}<br>'
        f'<b>H₀:</b> <code>{esc(res["h0"])}</code><br><b>H₁:</b> <code>{esc(res["h1"])}</code></div>',
        unsafe_allow_html=True)
    ptxt = "< 0.0001" if res["p"] < 1e-4 else f"{res['p']:.4f}"
    st.markdown(
        '<div class="result-big">'
        f'<div class="rb"><div class="rb-l">Test statistic ({esc(res["stat_name"])})</div><div class="rb-v">{res["stat"]:.4f}</div></div>'
        f'<div class="rb"><div class="rb-l">p-value</div><div class="rb-v">{ptxt}</div></div>'
        f'<div class="rb"><div class="rb-l">Significance level α</div><div class="rb-v">{res["alpha"]}</div></div>'
        f'<div class="rb"><div class="rb-l">Decision</div><div class="rb-v" style="color:{TEAL if sig else "#f2c46d"}">{"Reject H₀" if sig else "Fail to reject H₀"}</div></div></div>',
        unsafe_allow_html=True)
    cols = st.columns(len(res["extras"]))
    for c, (k, v) in zip(cols, res["extras"]):
        c.metric(k, v)
    st.markdown(f'<div class="decision {"sig" if sig else "ns"}"><b>Plain-English interpretation.</b> {res["interp"]}</div>',
                unsafe_allow_html=True)
    for w in res["warns"]:
        st.warning(w)
    if res["sim"]:
        st.info(SIM_NOTE)
    for name, tbl in res["tables"]:
        st.caption(name)
        cfg = {c: st.column_config.NumberColumn(c, format="₹%.2f") for c in tbl.columns if "(₹)" in c}
        call_stretch(st.dataframe, tbl, hide_index=True, column_config=cfg)


def page_lab():
    page_header("Hypothesis Testing Lab", "Run classical tests on the live dataset. Every result shows its hypotheses, statistic, p-value and meaning.",
                CHIP_CUR + [("Availability: simulated", True)])
    with st.container(key="panel_lab_ctl"):
        c1, c2 = st.columns([2, 1])
        test = c1.selectbox("Statistical test", ["Welch's t-test (compare two groups)", "One-way ANOVA (compare 2+ groups)",
                                                  "Pearson correlation (two numeric variables)", "Chi-square (two categorical variables)"],
                            key="lab_test")
        alpha = c2.selectbox("Significance level (α)", [0.05, 0.01, 0.10], key="lab_alpha")
        res = None
        if test.startswith("Welch"):
            g1, g2, g3 = st.columns(3)
            glabel = g1.selectbox("Grouping variable", list(GROUP_VARS), key="lab_w_g")
            gcol = GROUP_VARS[glabel]
            levels = sorted(DF[gcol].unique().tolist())
            fmt = (lambda x: f"{x}★") if gcol == "Rating" else str
            ga = g2.selectbox("Group 1", levels, index=0, format_func=fmt, key=f"lab_w_a_{gcol}")
            gb = g3.selectbox("Group 2", levels, index=min(1, len(levels) - 1), format_func=fmt, key=f"lab_w_b_{gcol}")
            st.caption("Outcome variable: Price (₹). The t-statistic and p-value are unaffected by the £→₹ conversion because it is a constant multiplier.")
            res = run_welch(DF, gcol, ga, gb, alpha)
        elif test.startswith("One-way"):
            glabel = st.selectbox("Grouping variable", list(GROUP_VARS), key="lab_a_g")
            st.caption("Outcome variable: Price (₹).")
            res = run_anova(DF, GROUP_VARS[glabel], alpha)
        elif test.startswith("Pearson"):
            p1, p2 = st.columns(2)
            xl = p1.selectbox("Variable X", list(NUM_VARS), index=0, key="lab_p_x")
            yl = p2.selectbox("Variable Y", list(NUM_VARS), index=1, key="lab_p_y")
            res = run_pearson(DF, NUM_VARS[xl], NUM_VARS[yl], alpha)
        else:
            q1, q2 = st.columns(2)
            l1 = q1.selectbox("Variable 1", list(CAT_VARS), index=0, key="lab_c_1")
            l2 = q2.selectbox("Variable 2", list(CAT_VARS), index=2, key="lab_c_2")
            res = run_chi2(DF, CAT_VARS[l1], CAT_VARS[l2], alpha)
    st.markdown("&nbsp;", unsafe_allow_html=True)
    with st.container(key="panel_lab_out"):
        if isinstance(res, str):
            st.warning(res)
        else:
            render_result(res)


# ----------------------------------------------------------------------------
# PAGE: Dataset & Data Quality
# ----------------------------------------------------------------------------
def page_data():
    page_header("Dataset & Data Quality", "Inspect, validate and export the data behind every page.", CHIP_CUR)
    t1, t2, t3 = st.tabs(["Preview", "Data quality", "Export"])
    with t1:
        c1, c2 = st.columns([3, 1.4])
        q = c1.text_input("Filter preview by title", key="dp_q", placeholder="Type to filter…")
        tech = c2.toggle("Show technical columns (image URL / name / extension)", key="dp_tech")
        cols = ["ID", "Title", "Price", PRICE, "Rating", "Availability", "Availability Data Type"]
        if tech:
            cols += [c for c in TECH_COLS if c in RAW.columns]
        view = RAW[cols].copy()
        view = view.rename(columns={"Price": f"Price in CSV ({'£' if SRC_CUR == 'GBP' else '₹'})",
                                    "Availability": "Availability (simulated)"})
        if q.strip():
            view = view[view["Title"].str.contains(q.strip(), case=False, regex=False, na=False)]
        st.caption(f"Showing {len(view):,} of {len(RAW):,} rows")
        call_stretch(st.dataframe, view, hide_index=True, height=460, column_config={
            PRICE: st.column_config.NumberColumn(PRICE, format="₹%.2f"),
            f"Price in CSV ({'£' if SRC_CUR == 'GBP' else '₹'})": st.column_config.NumberColumn(format="%.2f"),
            "Image URL": st.column_config.LinkColumn("Image URL")})
    with t2:
        miss_total = int(RAW.isna().sum().sum())
        dup_rows = int(RAW.drop(columns=["ID"]).duplicated().sum())
        dup_titles = RAW[RAW["Title"].duplicated(keep=False)]
        bad_price = int((RAW["Price"].isna() | (RAW["Price"] <= 0)).sum())
        bad_rating = int((RAW["Rating"].isna() | ~RAW["Rating"].between(1, 5)).sum())
        m = st.columns(5)
        m[0].metric("Rows", f"{len(RAW):,}")
        m[1].metric("Columns", f"{RAW.shape[1] - 1}")
        m[2].metric("Missing cells", f"{miss_total}")
        m[3].metric("Duplicate rows", f"{dup_rows}")
        m[4].metric("Duplicate titles", f"{dup_titles['Title'].nunique()}")
        st.markdown("&nbsp;", unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.markdown("**Missing values by column**")
            mv = RAW.drop(columns=["ID"]).isna().sum().reset_index()
            mv.columns = ["Column", "Missing"]
            mv["Technical column"] = mv["Column"].isin(TECH_COLS)
            call_stretch(st.dataframe, mv, hide_index=True)
            st.caption("Missing image fields do not affect any price, rating or availability analysis.")
        with b:
            st.markdown("**Validation checks**")
            checks = pd.DataFrame([
                ("Prices missing or ≤ 0", bad_price),
                ("Ratings missing or outside 1–5", bad_rating),
                ("Availability values not in {In stock, Out of stock}", int((~RAW["Availability"].isin(AVAIL_VALUES)).sum())),
                ("Rows excluded from analysis", N_DROPPED),
            ], columns=["Check", "Count"])
            call_stretch(st.dataframe, checks, hide_index=True)
        if len(dup_titles):
            st.markdown("**Duplicate titles** (different rows, kept because prices/ratings differ and they are not exact duplicates)")
            call_stretch(st.dataframe, dup_titles[["ID", "Title", "Price", "Rating", "Availability"]], hide_index=True)
        st.markdown("**Currency & availability notes**")
        st.markdown(
            f"- The CSV has no currency column. Prices range {RAW['Price'].min():.2f}–{RAW['Price'].max():.2f} and the image URLs point to "
            "`books.toscrape.com`, which publishes prices in **£ (GBP)**. GBP is therefore the default assumption (an inference, not a stored fact).\n"
            f"- Current setting: **{'GBP × ' + format(RATE, 'g') + ' = INR (manual rate, not live)' if SRC_CUR == 'GBP' else 'values used as INR, no conversion'}**. "
            "The conversion happens once, in the data loader; every page reads that same processed table.\n"
            f"- `Availability Data Type` = {', '.join(sorted(RAW['Availability Data Type'].dropna().unique())) if 'Availability Data Type' in RAW else 'n/a'}. "
            "Availability is **simulated** and is not real-world stock.")
        st.markdown("**Descriptive statistics**")
        call_stretch(st.dataframe, DF[[PRICE, "Rating"]].describe().T.round(2))
    with t3:
        st.markdown("Export the processed data (prices in ₹).")
        inc = st.toggle("Include technical columns", key="ex_tech")
        cols = ["ID", "Title", "Price", PRICE, "Rating", "Availability", "Availability Data Type"] + (TECH_COLS if inc else [])
        out = RAW[cols].rename(columns={"Price": "Price in source CSV"})
        e1, e2 = st.columns(2)
        e1.download_button("⇩ Download CSV", out.to_csv(index=False).encode("utf-8-sig"), "bookiq_processed.csv", "text/csv")
        e2.download_button("⇩ Download JSON", out.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
                           "bookiq_processed.json", "application/json")
        st.caption("CSV is written as UTF-8 with BOM so Excel on Windows opens it correctly.")


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
ROUTES = {"Overview": page_overview, "Discovery": page_discovery, "Compare": page_compare,
          "Analytics": page_analytics, "Lab": page_lab, "Data": page_data}
label_to_key = {lbl: key for key, lbl in PAGES}
ROUTES[label_to_key.get(st.session_state["nav"], "Overview")]()
