"""
🏏 SupeRR Selector — Rajasthan Royals Edition
=============================================
IPL-Style Analytics Dashboard in Royal Pink & Royal Blue
Powered by Random Forest Machine Learning (R² = 0.97)
"""

import os, pickle, base64
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
PROC_DIR  = os.path.join(BASE, "processed")
MODEL_DIR = os.path.join(BASE, "models")
ASSET_DIR = os.path.join(BASE, "assets")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rajasthan Royals | SupeRR Selector",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helper: Asset to Base64 ───────────────────────────────────────────────────
def get_svg_base64(filename):
    path = os.path.join(ASSET_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

RR_BANNER_B64     = get_svg_base64("rr_banner.svg")
STADIUM_BG_B64    = get_svg_base64("stadium_backdrop.svg")
BALL_SVG_B64      = get_svg_base64("cricket_ball.svg")
STUMPS_SVG_B64    = get_svg_base64("stumps.svg")
TROPHY_SVG_B64    = get_svg_base64("trophy.svg")
PITCH_SVG_B64     = get_svg_base64("pitch.svg")

# ── RAJASTHAN ROYALS THEME (STADIUM PHOTOGRAPHY + ROYAL PINK & BLUE) ───────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700;800;900&family=Poppins:wght@300;400;500;600;700;800&display=swap');

    /* Base Stadium Atmosphere with Transparent Night Stadium Overlay */
    .stApp {{
        background-image: 
            linear-gradient(180deg, rgba(3, 6, 20, 0.88) 0%, rgba(38, 6, 45, 0.90) 45%, rgba(2, 4, 14, 0.96) 100%),
            url("https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?q=80&w=2000&auto=format&fit=crop"),
            url("data:image/svg+xml;base64,{STADIUM_BG_B64}");
        background-size: cover, cover, cover;
        background-position: center top;
        background-attachment: fixed;
        background-repeat: no-repeat;
        font-family: 'Poppins', sans-serif;
        color: #f1f5f9;
    }}

    /* Sports Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 800;
        letter-spacing: 0.5px;
    }}

    /* Rajasthan Royals Pink Title */
    .rr-title {{
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ff007f 0%, #ea1a85 30%, #ffd700 70%, #ff69b4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        margin: 0;
        line-height: 1.05;
        letter-spacing: 1px;
    }}
    .rr-subtitle {{
        font-size: 1rem;
        color: #ffd700;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 700;
        margin-top: 4px;
    }}

    /* Glassmorphism Card with High-Contrast Royal Pink Borders */
    .glass-card {{
        background: rgba(10, 14, 32, 0.82);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(234, 26, 133, 0.35);
        border-radius: 16px;
        padding: 1.3rem;
        box-shadow: 0 12px 35px -8px rgba(0, 0, 0, 0.8);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .glass-card:hover {{
        border-color: rgba(255, 0, 127, 0.7);
        box-shadow: 0 16px 45px -5px rgba(234, 26, 133, 0.45);
        transform: translateY(-3px);
    }}

    /* RR Royal Pink Metric Card */
    .metric-box {{
        background: linear-gradient(135deg, rgba(46, 8, 56, 0.88) 0%, rgba(13, 22, 54, 0.82) 100%);
        border: 1px solid rgba(234, 26, 133, 0.45);
        border-left: 5px solid #ea1a85;
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }}
    .metric-box.gold {{
        border-left: 5px solid #ffd700;
        background: linear-gradient(135deg, rgba(54, 44, 10, 0.88) 0%, rgba(13, 22, 54, 0.82) 100%);
        border-color: rgba(255, 215, 0, 0.45);
    }}
    .metric-box.blue {{
        border-left: 5px solid #00f2fe;
        background: linear-gradient(135deg, rgba(8, 32, 62, 0.88) 0%, rgba(10, 15, 38, 0.82) 100%);
        border-color: rgba(0, 242, 254, 0.45);
    }}
    .metric-box h2 {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.3rem;
        margin: 0;
        color: #ffffff;
        font-weight: 800;
        line-height: 1.1;
    }}
    .metric-box p {{
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #ffd700;
        margin: 0;
        font-weight: 700;
    }}

    /* Halla Bol Badge */
    .halla-bol-badge {{
        display: inline-flex;
        align-items: center;
        background: linear-gradient(90deg, #ff007f 0%, #ea1a85 100%);
        color: #ffffff;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 900;
        font-size: 0.85rem;
        padding: 3px 14px;
        border-radius: 20px;
        letter-spacing: 2px;
        box-shadow: 0 0 15px rgba(255, 0, 127, 0.7);
        text-transform: uppercase;
    }}

    /* Avatar with Royal Pink gradient */
    .player-avatar {{
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #ff007f 0%, #ffd700 100%);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 900;
        font-size: 1.5rem;
        color: #0b0f19;
        box-shadow: 0 0 20px rgba(255, 0, 127, 0.6);
    }}

    /* SupeRR Score Badge */
    .superr-badge {{
        display: inline-block;
        background: linear-gradient(90deg, #ff007f 0%, #ffd700 100%);
        color: #0b0f19;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 900;
        font-size: 1.35rem;
        padding: 2px 14px;
        border-radius: 20px;
        letter-spacing: 0.5px;
        box-shadow: 0 0 15px rgba(255, 0, 127, 0.5);
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(10, 14, 32, 0.85);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(234, 26, 133, 0.3);
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #94a3b8;
        border-radius: 10px;
        padding: 8px 18px;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, #ea1a85 0%, #9900ff 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 18px rgba(234, 26, 133, 0.6);
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(90deg, #ff007f 0%, #ea1a85 50%, #ffd700 100%);
        color: #0b0f19;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 900;
        font-size: 1.15rem;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.6rem;
        letter-spacing: 1px;
        box-shadow: 0 4px 20px rgba(255, 0, 127, 0.5);
        transition: all 0.2s ease;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 28px rgba(255, 0, 127, 0.8);
        color: #ffffff;
    }}

    /* Sidebar Background with Cricket Ball Turf Texture */
    section[data-testid="stSidebar"] {{
        background-image: 
            linear-gradient(180deg, rgba(6, 10, 24, 0.92) 0%, rgba(38, 8, 45, 0.94) 100%),
            url("https://images.unsplash.com/photo-1531415074868-036b107e775a?q=80&w=800&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        border-right: 2px solid rgba(234, 26, 133, 0.35);
    }}
</style>
""", unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    bat   = pd.read_csv(os.path.join(PROC_DIR, "batting_scored.csv"))
    bow   = pd.read_csv(os.path.join(PROC_DIR, "bowling_scored.csv"))
    look  = pd.read_csv(os.path.join(PROC_DIR, "player_lookup.csv"))
    fi_b  = pd.read_csv(os.path.join(MODEL_DIR, "feature_importance_bat.csv"))
    fi_w  = pd.read_csv(os.path.join(MODEL_DIR, "feature_importance_bow.csv"))
    
    strokes = pd.read_csv(os.path.join(PROC_DIR, "strokes_summary.csv")) if os.path.exists(os.path.join(PROC_DIR, "strokes_summary.csv")) else None
    dismiss = pd.read_csv(os.path.join(PROC_DIR, "dismissals_summary.csv")) if os.path.exists(os.path.join(PROC_DIR, "dismissals_summary.csv")) else None
    ps_df   = pd.read_csv(os.path.join(PROC_DIR, "pace_spin_summary.csv")) if os.path.exists(os.path.join(PROC_DIR, "pace_spin_summary.csv")) else None

    # merge teams
    bat = bat.merge(look[["player_id", "team_name"]].drop_duplicates("player_id"), on="player_id", how="left")
    bow = bow.merge(look[["player_id", "team_name"]].drop_duplicates("player_id"), on="player_id", how="left")
    bat["team_name"] = bat["team_name"].fillna("T20 Franchise")
    bow["team_name"] = bow["team_name"].fillna("T20 Franchise")

    return bat, bow, look, fi_b, fi_w, strokes, dismiss, ps_df


try:
    bat, bow, look, fi_b, fi_w, strokes_df, dismiss_df, ps_df = load_all_data()
except Exception as e:
    st.error(f"Error loading preprocessed data: {e}")
    st.stop()


# ── Plotly Theme: Royal Pink & Royal Blue ─────────────────────────────────────
def style_plotly_rr(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(10, 14, 32, 0.6)",
        plot_bgcolor="rgba(10, 14, 32, 0.6)",
        font=dict(family="Poppins, sans-serif", color="#f1f5f9"),
        margin=dict(l=30, r=20, t=40, b=30),
        legend=dict(bgcolor="rgba(6,10,24,0.7)", font=dict(color="#ffd700")),
    )
    fig.update_xaxes(gridcolor="rgba(234,26,133,0.15)", zerolinecolor="rgba(234,26,133,0.25)")
    fig.update_yaxes(gridcolor="rgba(234,26,133,0.15)", zerolinecolor="rgba(234,26,133,0.25)")
    return fig


# ── Spider Radar Function (Pink vs Gold) ──────────────────────────────────────
def build_radar_rr(r1, r2, p1_name, p2_name, role="batting"):
    if role == "batting":
        categories = ["Strike Rate", "Average", "Boundary %", "PP Strike Rate", "Death Strike Rate", "Consistency"]
        max_vals   = [200, 50, 65, 180, 260, 80]
        v1 = [
            min(100, (r1.get("strike_rate", 0) / max_vals[0]) * 100),
            min(100, (r1.get("batting_avg", 0) / max_vals[1]) * 100),
            min(100, (r1.get("boundary_pct", 0) / max_vals[2]) * 100),
            min(100, (r1.get("sr_powerplay", 0) / max_vals[3]) * 100),
            min(100, (r1.get("sr_death", 0) / max_vals[4]) * 100),
            min(100, (r1.get("consistency_pct", 0) / max_vals[5]) * 100),
        ]
        v2 = [
            min(100, (r2.get("strike_rate", 0) / max_vals[0]) * 100),
            min(100, (r2.get("batting_avg", 0) / max_vals[1]) * 100),
            min(100, (r2.get("boundary_pct", 0) / max_vals[2]) * 100),
            min(100, (r2.get("sr_powerplay", 0) / max_vals[3]) * 100),
            min(100, (r2.get("sr_death", 0) / max_vals[4]) * 100),
            min(100, (r2.get("consistency_pct", 0) / max_vals[5]) * 100),
        ]
    else:
        categories = ["Economy", "Wickets", "Strike Rate", "Dot Ball %", "PP Economy", "Death Economy"]
        v1 = [
            max(0, min(100, (12 - r1.get("economy", 10)) / 7 * 100)),
            min(100, (r1.get("wickets", 0) / 30) * 100),
            max(0, min(100, (30 - r1.get("strike_rate_b", 25)) / 20 * 100)),
            min(100, (r1.get("dot_ball_pct", 0) / 60) * 100),
            max(0, min(100, (11 - r1.get("econ_powerplay", 9)) / 6 * 100)),
            max(0, min(100, (14 - r1.get("econ_death", 11)) / 8 * 100)),
        ]
        v2 = [
            max(0, min(100, (12 - r2.get("economy", 10)) / 7 * 100)),
            min(100, (r2.get("wickets", 0) / 30) * 100),
            max(0, min(100, (30 - r2.get("strike_rate_b", 25)) / 20 * 100)),
            min(100, (r2.get("dot_ball_pct", 0) / 60) * 100),
            max(0, min(100, (11 - r2.get("econ_powerplay", 9)) / 6 * 100)),
            max(0, min(100, (14 - r2.get("econ_death", 11)) / 8 * 100)),
        ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=v1 + [v1[0]], theta=categories + [categories[0]],
        fill='toself', name=f"👑 {p1_name}",
        line=dict(color='#ff007f', width=3),
        fillcolor='rgba(255, 0, 127, 0.35)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=v2 + [v2[0]], theta=categories + [categories[0]],
        fill='toself', name=f"⚡ {p2_name}",
        line=dict(color='#ffd700', width=3),
        fillcolor='rgba(255, 215, 0, 0.3)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(234,26,133,0.2)"),
            angularaxis=dict(gridcolor="rgba(234,26,133,0.2)"),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=True, height=370,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    return style_plotly_rr(fig)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <div class="halla-bol-badge">★ HALLA BOL ★</div>
        <h1 style="margin:6px 0 0 0; font-size:2rem; color:#ea1a85; font-family:'Rajdhani',sans-serif;">
            RAJASTHAN ROYALS
        </h1>
        <div style="font-size:0.75rem; letter-spacing:2px; color:#ffd700; font-weight:700;">
            SUPERR SELECTOR v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "ROYAL WAR ROOM NAVIGATION",
        [
            "👑 The Royal Pavilion (Overview)",
            "🦁 RR Royals Spotlight",
            "🏏 Batting Pressure Arena",
            "⚾ Bowling Pressure Matrix",
            "⚡ 1v1 Battle Octagon",
            "🎛️ Live ML Formula Studio",
            "🎯 Tactical Dream XI Generator"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;">
        <span class="halla-bol-badge" style="background:#001e48; border:1px solid #ea1a85;">
            🔴 LIVE ML DRAFT ENGINE
        </span>
        <div style="margin-top:10px; font-size:0.75rem; color:#cbd5e1; line-height:1.4;">
            Official T20 / IPL dataset. <b>200,999 balls</b> analyzed across <b>883 matches</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# 1. THE ROYAL PAVILION (OVERVIEW)
# ════════════════════════════════════════════════════════════════════════════════
if menu == "👑 The Royal Pavilion (Overview)":
    # Custom Rajasthan Royals Tournament Header Banner
    if RR_BANNER_B64:
        st.markdown(f"""
        <div style="margin-bottom: 1.5rem; filter:drop-shadow(0 12px 30px rgba(234,26,133,0.4));">
            <img src="data:image/svg+xml;base64,{RR_BANNER_B64}" width="100%" style="border-radius:18px;" />
        </div>
        """, unsafe_allow_html=True)

    # Canva generated asset link banner
    st.markdown("""
    <div class="glass-card" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.4rem; padding:0.8rem 1.4rem; border:1px solid #ffd700;">
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-size:1.5rem;">🎨</span>
            <div>
                <b style="color:#ffd700;">Canva Official Design Generated:</b>
                <span style="color:#cbd5e1; font-size:0.85rem;"> Custom Rajasthan Royals Photorealistic Stadium Creative is ready on Canva.</span>
            </div>
        </div>
        <div>
            <a href="https://www.canva.com/M/MAHbRc_sV9U?utm_source=OC-AZb1vOWcedZR&utm_campaign=agent_connector_create_image_asset_opened" target="_blank" style="text-decoration:none;">
                <span class="halla-bol-badge" style="background:linear-gradient(90deg, #ffd700, #ff007f); color:#0b0f19;">
                    Open generated image ↗
                </span>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Big 4 Metric KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        c1.markdown("""
        <div class="metric-box">
            <p>Total Deliveries</p>
            <h2>200,999</h2>
        </div>""", unsafe_allow_html=True)
    with c2:
        c2.markdown("""
        <div class="metric-box blue">
            <p>Matches Analysed</p>
            <h2>883</h2>
        </div>""", unsafe_allow_html=True)
    with c3:
        c3.markdown("""
        <div class="metric-box gold">
            <p>Elite Batters Evaluated</p>
            <h2>564</h2>
        </div>""", unsafe_allow_html=True)
    with c4:
        c4.markdown("""
        <div class="metric-box">
            <p>Elite Bowlers Evaluated</p>
            <h2>552</h2>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top 3 MVP Spotlight Cards in Pink & Gold
    st.markdown("### 👑 Top Rated Clutch Masters (SupeRR Rank #1–3)")
    top_batters_3 = bat.nlargest(3, "superr_score")
    top_bowlers_3 = bow.nlargest(3, "superr_score")

    col_b, col_w = st.columns(2)
    with col_b:
        st.markdown("<h4 style='color:#ff007f;'>🏏 Top Batters Under Pressure</h4>", unsafe_allow_html=True)
        for _, p in top_batters_3.iterrows():
            initials = "".join([n[0] for n in p["player_name"].split()[:2]])
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom:0.7rem; display:flex; align-items:center; justify-content:space-between; padding:0.9rem 1.3rem;">
                <div style="display:flex; align-items:center; gap:14px;">
                    <div class="player-avatar">{initials}</div>
                    <div>
                        <div style="font-weight:800; font-size:1.2rem; color:#fff;">{p['player_name']}</div>
                        <div style="font-size:0.78rem; color:#ffd700;">{p['team_name']} • Runs: {int(p['total_runs'])} | SR: {p['strike_rate']}</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div class="superr-badge">{p['superr_score']:.1f}</div>
                    <div style="font-size:0.68rem; color:#ffd700; font-weight:700;">SUPERR SCORE</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_w:
        st.markdown("<h4 style='color:#00f2fe;'>⚾ Top Bowlers Under Pressure</h4>", unsafe_allow_html=True)
        for _, p in top_bowlers_3.iterrows():
            initials = "".join([n[0] for n in p["player_name"].split()[:2]])
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom:0.7rem; display:flex; align-items:center; justify-content:space-between; padding:0.9rem 1.3rem;">
                <div style="display:flex; align-items:center; gap:14px;">
                    <div class="player-avatar" style="background:linear-gradient(135deg, #00f2fe, #ea1a85);">{initials}</div>
                    <div>
                        <div style="font-weight:800; font-size:1.2rem; color:#fff;">{p['player_name']}</div>
                        <div style="font-size:0.78rem; color:#00f2fe;">{p['team_name']} • Wkts: {int(p['wickets'])} | Econ: {p['economy']}</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div class="superr-badge" style="background:linear-gradient(90deg, #00f2fe, #ea1a85); color:#ffffff;">{p['superr_score']:.1f}</div>
                    <div style="font-size:0.68rem; color:#00f2fe; font-weight:700;">SUPERR SCORE</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Deep Analytics Charts from 200k balls
    st.markdown("### 📊 Macro League Insights")
    g1, g2 = st.columns(2)
    with g1:
        if strokes_df is not None:
            fig_st = px.bar(
                strokes_df, x="count", y="stroke", orientation="h",
                color="count", color_continuous_scale="RdPu",
                title="🏏 Most Frequent Batting Strokes Across 200,000 Deliveries"
            )
            fig_st.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=350)
            st.plotly_chart(style_plotly_rr(fig_st), use_container_width=True)
    with g2:
        if ps_df is not None:
            fig_ps = px.pie(
                ps_df, names="pace_or_spin", values="wickets",
                hole=0.45, title="🎯 Pace vs Spin Wicket Share",
                color_discrete_sequence=["#ea1a85", "#ffd700"]
            )
            fig_ps.update_layout(height=350)
            st.plotly_chart(style_plotly_rr(fig_ps), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# 2. RR ROYALS SPOTLIGHT
# ════════════════════════════════════════════════════════════════════════════════
elif menu == "🦁 RR Royals Spotlight":
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <span class="halla-bol-badge">★ HALLA BOL ★</span>
        <h1 class="rr-title">Rajasthan Royals Squad Spotlight</h1>
        <p style="color:#ffd700; font-weight:600;">How the Pink Army stalwarts rank under high-pressure conditions in the ML Engine.</p>
    </div>
    """, unsafe_allow_html=True)

    rr_core_names = [
        "Jos Buttler", "Yashasvi Jaiswal", "Sanju Samson", "Shimron Hetmyer",
        "Riyan Parag", "Yuzvendra Chahal", "Ravichandran Ashwin", "Trent Boult",
        "Sandeep Sharma", "Avesh Khan", "Dhruv Jurel"
    ]

    rr_bat = bat[bat["player_name"].isin(rr_core_names)].copy().sort_values("superr_score", ascending=False)
    rr_bow = bow[bow["player_name"].isin(rr_core_names)].copy().sort_values("superr_score", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h3 style='color:#ff007f;'>🏏 Pink Army Batters</h3>", unsafe_allow_html=True)
        for _, r in rr_bat.iterrows():
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom:0.7rem; border-left:4px solid #ea1a85;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:1.25rem; font-weight:800; color:#fff;">{r['player_name']}</div>
                        <div style="font-size:0.8rem; color:#ffd700;">
                            Runs: {int(r['total_runs'])} | SR: {r['strike_rate']} | Avg: {r['batting_avg']} | Death SR: {r['sr_death']}
                        </div>
                    </div>
                    <div class="superr-badge">{r['superr_score']:.1f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 style='color:#ffd700;'>⚾ Pink Army Bowlers</h3>", unsafe_allow_html=True)
        for _, r in rr_bow.iterrows():
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom:0.7rem; border-left:4px solid #ffd700;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:1.25rem; font-weight:800; color:#fff;">{r['player_name']}</div>
                        <div style="font-size:0.8rem; color:#00f2fe;">
                            Wickets: {int(r['wickets'])} | Economy: {r['economy']} | Dot %: {r['dot_ball_pct']}% | Death Econ: {r['econ_death']}
                        </div>
                    </div>
                    <div class="superr-badge" style="background:#ffd700; color:#0b0f19;">{r['superr_score']:.1f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# 3. BATTING PRESSURE ARENA
# ════════════════════════════════════════════════════════════════════════════════
elif menu == "🏏 Batting Pressure Arena":
    st.markdown("""
    <div style="margin-bottom:1rem;">
        <span class="halla-bol-badge">ROYAL CLUTCH HITTING</span>
        <h1 class="rr-title" style="font-size:2.4rem;">Batting Pressure Arena</h1>
        <p style="color:#94a3b8;">Filter by minimum balls faced to eliminate small-sample noise and isolate proven clutch hitters.</p>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns([2, 3, 2, 2])
    with f1:
        team_opts = ["All Teams"] + sorted(bat["team_name"].unique())
        sel_team  = st.selectbox("Filter Team", team_opts, key="bat_team_filter")
    with f2:
        min_balls = st.slider("Min Balls Faced (Sample Filter)", min_value=30, max_value=1200, value=100, step=25, key="bat_min_balls")
    with f3:
        sort_by   = st.selectbox("Sort Table By", ["superr_score", "strike_rate", "batting_avg", "boundary_pct", "sr_death"], key="bat_sort_col")

    df_b = bat[bat["balls_faced"] >= min_balls].copy()
    if sel_team != "All Teams":
        df_b = df_b[df_b["team_name"] == sel_team]
    df_b = df_b.sort_values(sort_by, ascending=False).reset_index(drop=True)

    with f4:
        st.markdown(f"""
        <div class="metric-box pink" style="padding:0.7rem 1rem; margin-top:0.2rem;">
            <p>Qualified Batters</p>
            <h2 style="font-size:1.8rem; margin:0;">{len(df_b)} <span style="font-size:0.9rem; color:#ffd700;">/ {len(bat)}</span></h2>
        </div>
        """, unsafe_allow_html=True)

    # Fast GPU-accelerated Scatter with fixed axes so drops are instantly noticeable
    fig_scatter = px.scatter(
        df_b, x="strike_rate", y="batting_avg",
        size="total_runs", color="superr_score",
        hover_name="player_name",
        hover_data={"team_name": True, "balls_faced": True, "boundary_pct": True, "sr_death": True, "total_runs": True},
        color_continuous_scale="Plasma",
        render_mode="webgl",
        title=f"⚡ Clutch Quadrant — {len(df_b)} Batters Displayed (Filter: ≥ {min_balls} Balls Faced)",
        labels={"strike_rate": "Strike Rate (Scoring Speed)", "batting_avg": "Batting Average (Stability)"}
    )
    median_sr = float(df_b["strike_rate"].median()) if len(df_b) > 0 else 140.0
    median_av = float(df_b["batting_avg"].median()) if len(df_b) > 0 else 30.0
    fig_scatter.add_hline(y=median_av, line_dash="dash", line_color="rgba(255,215,0,0.5)", annotation_text=f"Median Avg: {median_av:.1f}")
    fig_scatter.add_vline(x=median_sr, line_dash="dash", line_color="rgba(255,215,0,0.5)", annotation_text=f"Median SR: {median_sr:.1f}")
    fig_scatter.update_layout(
        height=480,
        xaxis=dict(range=[80, 230], title="Strike Rate"),
        yaxis=dict(range=[10, 65], title="Batting Average")
    )
    st.plotly_chart(style_plotly_rr(fig_scatter), use_container_width=True)

    st.markdown(f"### 🔥 Phase Mastery: Top {min(15, len(df_b))} Batters (≥ {min_balls} Balls)")
    top15 = df_b.head(15)
    if len(top15) > 0:
        phase_melt = top15.melt(
            id_vars=["player_name"],
            value_vars=["sr_powerplay", "sr_middle", "sr_death"],
            var_name="Phase", value_name="Strike Rate"
        )
        phase_melt["Phase"] = phase_melt["Phase"].map({
            "sr_powerplay": "Powerplay (1-6)",
            "sr_middle": "Middle (7-15)",
            "sr_death": "Death Overs (16-20)"
        })
        fig_phase = px.bar(
            phase_melt, x="player_name", y="Strike Rate", color="Phase", barmode="group",
            color_discrete_sequence=["#00f2fe", "#ffd700", "#ff007f"]
        )
        fig_phase.update_layout(height=380, xaxis_tickangle=-35)
        st.plotly_chart(style_plotly_rr(fig_phase), use_container_width=True)

    st.markdown("### 📋 Batter Scorecard Leaderboard")
    show_cols = ["player_name", "team_name", "total_runs", "balls_faced", "strike_rate",
                 "batting_avg", "boundary_pct", "sr_death", "consistency_pct", "superr_score"]
    st.dataframe(
        df_b[show_cols].style.background_gradient(subset=["superr_score"], cmap="RdPu"),
        use_container_width=True, height=350
    )


# ════════════════════════════════════════════════════════════════════════════════
# 4. BOWLING PRESSURE MATRIX
# ════════════════════════════════════════════════════════════════════════════════
elif menu == "⚾ Bowling Pressure Matrix":
    st.markdown("""
    <div style="margin-bottom:1rem;">
        <span class="halla-bol-badge">ROYAL WICKET ARSENAL</span>
        <h1 class="rr-title" style="font-size:2.4rem;">Bowling Pressure Matrix</h1>
        <p style="color:#94a3b8;">Identify the most miserly economy rates, wicket hunters, and death-over specialists.</p>
    </div>
    """, unsafe_allow_html=True)

    col_img, col_txt = st.columns([1, 4])
    with col_img:
        st.markdown(f'<img src="data:image/svg+xml;base64,{STUMPS_SVG_B64}" width="140" style="filter:drop-shadow(0 0 15px rgba(255,0,127,0.4));" />', unsafe_allow_html=True)
    with col_txt:
        st.markdown("""
        <div class="glass-card">
            <h4 style="margin:0 0 6px 0; color:#ffd700;">⚡ Pitch Control & Dot Ball Mastery</h4>
            <p style="margin:0; font-size:0.85rem; color:#cbd5e1;">
                Death overs (16–20) demand ice-cold nerves. Bowlers maintaining an economy below 8.5 while bowling &gt;35% dot balls are awarded premium SupeRR bonus points.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([2, 3, 2, 2])
    with f1:
        team_opts = ["All Teams"] + sorted(bow["team_name"].unique())
        sel_team  = st.selectbox("Filter Team", team_opts, key="bow_team_filter")
    with f2:
        min_balls_bow = st.slider("Min Balls Bowled", min_value=24, max_value=1200, value=90, step=25, key="bow_min_balls")
    with f3:
        sort_by   = st.selectbox("Sort Table By", ["superr_score", "wickets", "economy", "dot_ball_pct", "econ_death"], key="bow_sort_col")

    df_w = bow[bow["balls_bowled"] >= min_balls_bow].copy()
    if sel_team != "All Teams":
        df_w = df_w[df_w["team_name"] == sel_team]
    asc_sort = (sort_by in ["economy", "econ_death"])
    df_w = df_w.sort_values(sort_by, ascending=asc_sort).reset_index(drop=True)

    with f4:
        st.markdown(f"""
        <div class="metric-box blue" style="padding:0.7rem 1rem; margin-top:0.2rem;">
            <p>Qualified Bowlers</p>
            <h2 style="font-size:1.8rem; margin:0;">{len(df_w)} <span style="font-size:0.9rem; color:#00f2fe;">/ {len(bow)}</span></h2>
        </div>
        """, unsafe_allow_html=True)

    fig_bw = px.scatter(
        df_w, x="economy", y="bowling_avg",
        size="wickets", color="superr_score",
        hover_name="player_name",
        hover_data={"team_name": True, "balls_bowled": True, "dot_ball_pct": True, "econ_death": True, "wickets": True},
        color_continuous_scale="Viridis",
        render_mode="webgl",
        title=f"🎯 Bowler Impact Quadrant — {len(df_w)} Bowlers Displayed (Filter: ≥ {min_balls_bow} Balls)",
        labels={"economy": "Economy Rate (Run Prevention)", "bowling_avg": "Bowling Average (Runs Per Wicket)"}
    )
    median_ec = float(df_w["economy"].median()) if len(df_w) > 0 else 8.5
    median_ba = float(df_w["bowling_avg"].median()) if len(df_w) > 0 else 25.0
    fig_bw.add_hline(y=median_ba, line_dash="dash", line_color="rgba(255,215,0,0.5)", annotation_text=f"Median Avg: {median_ba:.1f}")
    fig_bw.add_vline(x=median_ec, line_dash="dash", line_color="rgba(255,215,0,0.5)", annotation_text=f"Median Econ: {median_ec:.1f}")
    fig_bw.update_layout(
        height=460,
        xaxis=dict(range=[4.5, 12.5], title="Economy Rate"),
        yaxis=dict(range=[8, 55], title="Bowling Average")
    )
    st.plotly_chart(style_plotly_rr(fig_bw), use_container_width=True)

    st.markdown("### 🏏 22-Yard Target Zones")
    st.markdown(f'<img src="data:image/svg+xml;base64,{PITCH_SVG_B64}" width="100%" style="border-radius:12px; margin-bottom:1.5rem;" />', unsafe_allow_html=True)

    show_cols_w = ["player_name", "team_name", "innings_bowled", "balls_bowled", "wickets",
                   "economy", "bowling_avg", "dot_ball_pct", "econ_death", "superr_score"]
    st.dataframe(
        df_w[show_cols_w].style.background_gradient(subset=["superr_score"], cmap="YlGnBu"),
        use_container_width=True, height=350
    )


# ════════════════════════════════════════════════════════════════════════════════
# 5. 1v1 BATTLE OCTAGON
# ════════════════════════════════════════════════════════════════════════════════
elif menu == "⚡ 1v1 Battle Octagon":
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <span class="halla-bol-badge">HEAD-TO-HEAD DUEL</span>
        <h1 class="rr-title" style="font-size:2.4rem;">The 1v1 Battle Octagon</h1>
        <p style="color:#94a3b8;">Compare any two players side-by-side with holographic radar stats and AI Match Verdict.</p>
    </div>
    """, unsafe_allow_html=True)

    role = st.radio("Select Matchup Role", ["Batting Duel", "Bowling Duel"], horizontal=True)
    df_pool = bat if role == "Batting Duel" else bow
    names   = sorted(df_pool["player_name"].dropna().unique())

    c1, c2 = st.columns(2)
    with c1:
        p1 = st.selectbox("🔴 Select Challenger 1", names, index=0)
    with c2:
        default_idx = 1 if len(names) > 1 else 0
        p2 = st.selectbox("🟡 Select Challenger 2", names, index=default_idx)

    r1 = df_pool[df_pool["player_name"] == p1].iloc[0]
    r2 = df_pool[df_pool["player_name"] == p2].iloc[0]

    sc1, sc2 = r1["superr_score"], r2["superr_score"]
    winner = p1 if sc1 >= sc2 else p2
    win_score = max(sc1, sc2)

    st.markdown(f"""
    <div class="glass-card" style="text-align:center; border: 2px solid #ea1a85; background: linear-gradient(135deg, rgba(38,8,48,0.9) 0%, rgba(13,22,54,0.8) 100%); margin-bottom:1.5rem;">
        <div style="font-size:0.85rem; letter-spacing:2px; color:#ffd700; font-weight:800;">🏆 AI SELECTION VERDICT</div>
        <h2 style="font-size:2.4rem; color:#ffffff; margin:0.2rem 0;">{winner} IS RECOMMENDED FOR SELECTION</h2>
        <div style="color:#ffd700; font-size:0.95rem; font-weight:600;">
            SupeRR Clutch Rating of <b>{win_score:.1f}/100</b> gives decisive tactical edge under pressure.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        init1 = "".join([n[0] for n in p1.split()[:2]])
        st.markdown(f"""
        <div class="glass-card" style="border-top:4px solid #ea1a85; text-align:center;">
            <div class="player-avatar">{init1}</div>
            <h3 style="margin:0; font-size:1.5rem;">{p1}</h3>
            <div style="color:#ffd700; font-size:0.85rem;">{r1.get('team_name','?')}</div>
            <div class="superr-badge">{sc1:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        init2 = "".join([n[0] for n in p2.split()[:2]])
        st.markdown(f"""
        <div class="glass-card" style="border-top:4px solid #ffd700; text-align:center;">
            <div class="player-avatar" style="background:linear-gradient(135deg, #ffd700, #ea1a85);">{init2}</div>
            <h3 style="margin:0; font-size:1.5rem;">{p2}</h3>
            <div style="color:#ffd700; font-size:0.85rem;">{r2.get('team_name','?')}</div>
            <div class="superr-badge" style="background:#ffd700; color:#0b0f19;">{sc2:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🕸️ Head-to-Head Radar Matrix")
    radar_fig = build_radar_rr(r1, r2, p1, p2, role="batting" if role == "Batting Duel" else "bowling")
    st.plotly_chart(radar_fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# 6. LIVE ML FORMULA STUDIO
# ════════════════════════════════════════════════════════════════════════════════
elif menu == "🎛️ Live ML Formula Studio":
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <span class="halla-bol-badge">STRATEGY SANDBOX</span>
        <h1 class="rr-title" style="font-size:2.4rem;">Live ML Formula Studio</h1>
        <p style="color:#94a3b8;">Customize the importance weights of key KPIs and watch the machine learning rankings recompute live!</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🏏 Batting Weight Configuration")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        w_sr = st.slider("Strike Rate Weight", 5, 40, 20)
    with s2:
        w_avg = st.slider("Batting Avg Weight", 5, 40, 20)
    with s3:
        w_bnd = st.slider("Boundary % Weight", 5, 30, 15)
    with s4:
        w_dth = st.slider("Death Over SR Weight", 5, 30, 15)

    df_custom = bat.copy()
    sr_norm   = df_custom["strike_rate"].clip(0, 220) / 220
    avg_norm  = df_custom["batting_avg"].clip(0, 60) / 60
    bnd_norm  = df_custom["boundary_pct"].clip(0, 60) / 60
    dth_norm  = df_custom["sr_death"].clip(0, 280) / 280

    total_w = w_sr + w_avg + w_bnd + w_dth
    df_custom["custom_score"] = (
        (sr_norm * w_sr + avg_norm * w_avg + bnd_norm * w_bnd + dth_norm * w_dth) / total_w * 100
    ).round(2)

    top10_custom = df_custom.nlargest(10, "custom_score")[
        ["player_name", "team_name", "total_runs", "strike_rate", "batting_avg", "sr_death", "custom_score"]
    ].reset_index(drop=True)
    top10_custom.index += 1

    st.markdown("### 🏆 Recomputed Top 10 Batters (Custom Philosophy)")
    st.dataframe(
        top10_custom.style.background_gradient(subset=["custom_score"], cmap="RdPu"),
        use_container_width=True
    )


# ════════════════════════════════════════════════════════════════════════════════
# 7. TACTICAL DREAM XI GENERATOR
# ════════════════════════════════════════════════════════════════════════════════
elif menu == "🎯 Tactical Dream XI Generator":
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <span class="halla-bol-badge">ROYALS DRAFT ROOM</span>
        <h1 class="rr-title" style="font-size:2.4rem;">Tactical Dream XI Generator</h1>
        <p style="color:#ffd700; font-weight:600;">Select match venue and pitch profiles to draft an optimized Rajasthan Royals / T20 Playing XI.</p>
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        pitch = st.selectbox(
            "🏟️ Match Venue & Pitch Profile",
            [
                "🏟️ Sawai Mansingh Stadium (Jaipur Fortress - Balanced Track)",
                "🏟️ Wankhede / Chinnaswamy (Flat Batting Paradise - High Chase Pressure)",
                "🏟️ Chepauk / Eden Gardens (Turning Spin Track - Economy Focus)",
                "🏟️ Centurion / Melbourne (Pace & Bounce - High Seam & Death Impact)"
            ]
        )
    with t2:
        team_filter = st.selectbox("Pool Availability", ["Global League Pool (All Stars)", "India Players Only", "Overseas Only"])

    b1, b2, b3 = st.columns(3)
    with b1:
        n_bat = st.slider("Top/Middle Batters", 4, 6, 5)
    with b2:
        n_all = st.slider("All-Rounders", 1, 3, 2)
    with b3:
        n_bow = st.slider("Specialist Bowlers", 3, 5, 4)

    if st.button("🚀 DRAFT OPTIMAL PLAYING XI", type="primary"):
        pool_bat = bat[bat["balls_faced"] >= 50].copy()
        pool_bow = bow[bow["balls_bowled"] >= 36].copy()

        if "Flat Batting" in pitch:
            pool_bat["pitch_adj_score"] = pool_bat["superr_score"] * 0.7 + (pool_bat["strike_rate"] / 2) * 0.3
            pool_bow["pitch_adj_score"] = pool_bow["superr_score"] * 0.7 + (15 - pool_bow["econ_death"]) * 2.5
        elif "Turning Spin" in pitch:
            pool_bat["pitch_adj_score"] = pool_bat["superr_score"] * 0.7 + pool_bat["batting_avg"] * 0.4
            pool_bow["pitch_adj_score"] = pool_bow["superr_score"] * 0.6 + pool_bow["dot_ball_pct"] * 0.5
        else:
            pool_bat["pitch_adj_score"] = pool_bat["superr_score"]
            pool_bow["pitch_adj_score"] = pool_bow["superr_score"]

        all_rounders = set(pool_bat["player_name"]) & set(pool_bow["player_name"])
        ar_scores = {}
        for name in all_rounders:
            bs = pool_bat[pool_bat["player_name"] == name]["pitch_adj_score"].values[0]
            ws = pool_bow[pool_bow["player_name"] == name]["pitch_adj_score"].values[0]
            ar_scores[name] = (bs + ws) / 2

        sorted_ar = sorted(ar_scores.items(), key=lambda x: x[1], reverse=True)
        picked_ar = [k for k, v in sorted_ar[:n_all]]

        pure_bat = pool_bat[~pool_bat["player_name"].isin(picked_ar)].nlargest(n_bat, "pitch_adj_score")
        pure_bow = pool_bow[~pool_bow["player_name"].isin(picked_ar)].nlargest(n_bow, "pitch_adj_score")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="border: 2px solid #ea1a85; margin-bottom:1.2rem; background:linear-gradient(135deg, rgba(38,8,48,0.9) 0%, rgba(13,22,54,0.85) 100%);">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h3 style="margin:0; color:#ffd700;">👑 ROYAL SQUAD DRAFTED</h3>
                    <div style="color:#ea1a85; font-size:0.9rem; font-weight:700;">Tactical Strategy Profile: {pitch}</div>
                </div>
                <img src="data:image/svg+xml;base64,{TROPHY_SVG_B64}" width="55" />
            </div>
        </div>
        """, unsafe_allow_html=True)

        xi_records = []
        for i, (_, r) in enumerate(pure_bat.iterrows()):
            role_label = "Opener / Anchor" if i < 2 else ("Middle-Order Finisher" if i > 3 else "Pace Hitter")
            xi_records.append({
                "Batting Pos": len(xi_records) + 1,
                "Player": r["player_name"],
                "Team": r.get("team_name", "?"),
                "Tactical Role": role_label,
                "SupeRR Score": round(r["pitch_adj_score"], 1),
                "Key Metric": f"SR: {r['strike_rate']} | Avg: {r['batting_avg']}"
            })

        for name in picked_ar:
            tm = pool_bat[pool_bat["player_name"] == name]["team_name"].values[0]
            sc = ar_scores[name]
            xi_records.append({
                "Batting Pos": len(xi_records) + 1,
                "Player": name,
                "Team": tm,
                "Tactical Role": "Clutch All-Rounder",
                "SupeRR Score": round(sc, 1),
                "Key Metric": "Dual Threat Bat & Ball"
            })

        for i, (_, r) in enumerate(pure_bow.iterrows()):
            role_label = "Strike Pace Bowler" if i < 2 else "Death Overs Specialist"
            xi_records.append({
                "Batting Pos": len(xi_records) + 1,
                "Player": r["player_name"],
                "Team": r.get("team_name", "?"),
                "Tactical Role": role_label,
                "SupeRR Score": round(r["pitch_adj_score"], 1),
                "Key Metric": f"Econ: {r['economy']} | Wkts: {int(r['wickets'])}"
            })

        xi_table = pd.DataFrame(xi_records)

        for _, row in xi_table.iterrows():
            pos_color = "#ea1a85" if "Bat" in row["Tactical Role"] or "Opener" in row["Tactical Role"] else ("#ffd700" if "All" in row["Tactical Role"] else "#00f2fe")
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom:0.5rem; padding:0.8rem 1.4rem; display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center; gap:16px;">
                    <div style="font-family:'Rajdhani', sans-serif; font-size:1.6rem; font-weight:900; color:{pos_color}; width:28px;">
                        #{row['Batting Pos']}
                    </div>
                    <div>
                        <div style="font-size:1.15rem; font-weight:800; color:#ffffff;">{row['Player']}</div>
                        <div style="font-size:0.75rem; color:#ffd700;">{row['Team']} • <span style="color:{pos_color}; font-weight:700;">{row['Tactical Role']}</span></div>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:20px;">
                    <div style="font-size:0.85rem; color:#cbd5e1; text-align:right;">{row['Key Metric']}</div>
                    <div class="superr-badge" style="background:{pos_color}; color:#0a0e17; font-size:1.1rem; padding:2px 10px;">
                        {row['SupeRR Score']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        avg_team_score = xi_table["SupeRR Score"].mean()
        st.markdown(f"""
        <div style="text-align:right; margin-top:10px; font-size:1rem; color:#ffd700; font-weight:800;">
            TEAM AVERAGE SUPERR RATING: {avg_team_score:.1f} / 100
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            "📥 Download Rajasthan Royals Tactical XI (CSV)",
            xi_table.to_csv(index=False),
            "rajasthan_royals_xi.csv", "text/csv"
        )
