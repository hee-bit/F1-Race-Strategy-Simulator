import os
import json
import fastf1
import pandas as pd
import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count
import streamlit as st
from pathlib import Path
import base64

# -----------------------------
# 0-0. 경로 설정
# -----------------------------
BASE_DIR = Path(__file__).parent

TRACK_IMAGES_PATHS = {
    "Bahrain": BASE_DIR / "assets" / "tracks" / "bahrain.jpg",
    "Saudi Arabia": BASE_DIR / "assets" / "tracks" / "saudi_arabia.jpg",
    "Australia": BASE_DIR / "assets" / "tracks" / "australia.jpg",
    "Japan": BASE_DIR / "assets" / "tracks" / "japan.jpg",
    "Monaco": BASE_DIR / "assets" / "tracks" / "monaco.jpg"
}

TRACK_IMAGE_WIDTHS = {
    "Bahrain": 200,
    "Saudi Arabia": 4560,
    "Australia": 980,
    "Japan": 4220,
    "Monaco": 1820
}

LOGO_PATH = BASE_DIR / "assets" / "logos" / "f1_logo.jpg"

DRIVER_OPTIONS = {
    "Alexander Albon (Williams)": "ALB",
    "Max Verstappen (Red Bull)": "VER",
    "Fernando Alonso (Aston Martin)": "ALO",
    "Charles Leclerc (Ferrari)": "LEC",
    "Oliver Bearman (Haas)": "BEA",
    "Valtteri Bottas (Cadillac)": "BOT",
    "Pierre Gasly (Alpine)": "GAS",
    "Lewis Hamilton (Ferrari)": "HAM",
    "Nico Hulkenberg (Audi)": "HUL",
    "Liam Lawson (RB)": "LAW",
    "Lando Norris (McLaren)": "NOR",
    "Esteban Ocon (Haas)": "OCO",
    "Sergio Perez (Cadillac)": "PER",
    "Oscar Piastri (McLaren)": "PIA",
    "George Russell (Mercedes)": "RUS",
    "Carlos Sainz (Williams)": "SAI",
    "Lance Stroll (Aston Martin)": "STR",
}

# -----------------------------
# 0-1. 폴더 및 캐시 설정
# -----------------------------
CACHE_DIR = "cache"
PREPROCESSED_DIR = "preprocessed"
IS_SERVER = os.getenv("STREAMLIT_SERVER_PORT") is not None

if not IS_SERVER:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

if "cache_enabled" not in st.session_state:
    if not IS_SERVER:
        fastf1.Cache.enable_cache(CACHE_DIR)
    else:
        fastf1.Cache.disable_cache()
    st.session_state["cache_enabled"] = True

# -----------------------------
# 0-2. 정책 파라미터 및 확률 모델
# -----------------------------
MIN_POSITION_GAIN_TO_PIT = 0.10
MIN_TIME_GAIN_TO_PIT = 0.30

COARSE_SIM_N = 40
MID_SIM_N = 120
FINAL_SIM_N = 260

TOPK_MID = 16
TOPK_FINAL = 10

MAX_RIVALS = 7
MAX_TOTAL_CANDIDATES = 140

MIN_LAPS_BETWEEN_STOPS = 5
EARLIEST_PIT_AFTER_CURRENT = 3
STINT_EXTRA_MARGIN = 4

USE_MULTIPROCESSING = False

ALLOW_ZERO_STOP_DEFAULT = False
ALLOW_ZERO_STOP_ONLY_IF_LATE_RACE = True
LATE_RACE_LAPS_REMAINING_THRESHOLD = 3

FORCE_ONE_STOP_IF_TYRE_LIFE_AT_LEAST = 10
NO_STOP_TIME_PENALTY = 0.0
OLD_TYRE_EXTRA_PENALTY_PER_LAP = 0.18
VERY_OLD_TYRE_THRESHOLD = 14

TIME_PRIORITY_WEIGHT = 1.0
POSITION_PRIORITY_WEIGHT = 0.35

RECENT_LAPS_FOR_PACE = 5
OVERTAKE_BASE_GAP = 1.0
OVERTAKE_MIN_ADVANTAGE = 0.28

TRACK_PARAMS = {
    "Bahrain": {"overtake_factor": 1.15, "drs_factor": 1.20, "dirty_air_factor": 0.90, "traffic_factor": 0.90},
    "Saudi Arabia": {"overtake_factor": 1.10, "drs_factor": 1.15, "dirty_air_factor": 0.95, "traffic_factor": 0.92},
    "Australia": {"overtake_factor": 0.95, "drs_factor": 1.00, "dirty_air_factor": 1.00, "traffic_factor": 1.00},
    "Japan": {"overtake_factor": 0.82, "drs_factor": 0.88, "dirty_air_factor": 1.12, "traffic_factor": 1.08},
    "Monaco": {"overtake_factor": 0.35, "drs_factor": 0.45, "dirty_air_factor": 1.35, "traffic_factor": 1.25}
}

TRACK_INCIDENT_PROBS = {
    "Bahrain": {"SC": 0.004, "VSC": 0.006},
    "Saudi Arabia": {"SC": 0.018, "VSC": 0.008},
    "Australia": {"SC": 0.014, "VSC": 0.005},
    "Japan": {"SC": 0.012, "VSC": 0.006},
    "Monaco": {"SC": 0.025, "VSC": 0.010}
}

def load_image_binary(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

# -----------------------------
# 0-3. 커스텀 CSS
# -----------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    :root {
        --bg: #0b0f14;
        --panel: #141a22;
        --panel-2: #1a2230;
        --text: #f5f7fb;
        --muted: #98a2b3;
        --accent: #ff4d4f;
        --accent-2: #ff7a45;
        --border: rgba(255,255,255,0.08);
        --shadow: 0 10px 30px rgba(0,0,0,0.22);
        --radius: 20px;
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(255,77,79,0.12), transparent 25%),
            linear-gradient(180deg, #0b0f14 0%, #111827 100%);
        color: var(--text);
    }

    .block-container {
    max-width: 1520px;
    padding-top: 1.6rem;
    padding-bottom: 2rem;
   }

    section[data-testid="stSidebar"] {
        background: #0f141c;
        border-right: 1px solid var(--border);
    }

    h1, h2, h3, h4 {
        color: var(--text);
        letter-spacing: -0.02em;
    }

    p, label, .stCaption {
        color: var(--muted);
    }

    .stSelectbox label,
    .stNumberInput label,
    .stTextInput label,
    .stRadio label {
        color: var(--muted) !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(20, 26, 34, 0.92);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 16px;
        box-shadow: var(--shadow);
    }

    div[data-testid="stDataFrame"] {
        background: rgba(20, 26, 34, 0.92);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 8px;
    }

    .custom-card {
        background: rgba(20, 26, 34, 0.92);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 22px;
        box-shadow: var(--shadow);
        margin-bottom: 20px;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(255,77,79,0.13), rgba(20,26,34,0.95));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 28px;
        box-shadow: var(--shadow);
        min-height: 260px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 0.4rem;
    }

    .hero-sub {
        font-size: 1rem;
        color: var(--muted);
        line-height: 1.6;
    }

    .section-label {
    color: #cfd6e4;
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.8rem;
    white-space: normal !important;
    word-break: keep-all;
    overflow-wrap: break-word;
    }
    
    .left-result-shift {
    margin-left: 0.5cm;
    }
    
    .briefing-board {
    background: rgba(20, 26, 34, 0.92);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px;
    box-shadow: var(--shadow);
    margin-left: 26px;
    }

    .stButton > button {
        width: 100%;
        border: 0;
        border-radius: 14px;
        background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
        color: #ffffff !important;
        font-weight: 800;
        padding: 0.9rem 1.2rem;
    }

    .stButton > button:hover {
        filter: brightness(1.04);
        color: #ffffff !important;
    }

    .stButton > button:focus,
    .stButton > button:active {
        color: #ffffff !important;
        outline: none !important;
        box-shadow: none !important;
    }

    hr {
        margin: 0 !important;
        padding: 0 !important;
        border: 0 !important;
    }

    div[data-testid="stSpinner"] p {
    white-space: normal !important;
    word-break: keep-all !important;
    overflow-wrap: anywhere !important;
    line-height: 1.5 !important;
    font-size: 0.98rem !important;
    }

    div[data-testid="stSpinner"] {
        width: 100% !important;
    }

    div[data-testid="column"] {
        min-width: 0 !important;
    }
    
    </style>
    """, unsafe_allow_html=True)


# -----------------------------
# 1. 데이터 로드/전처리 비즈니스 로직
# -----------------------------
def load_race_laps(seasons, grands_prix):
    all_laps = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_gps = len(seasons) * len(grands_prix)
    idx = 0

    for year in seasons:
        for gp in grands_prix:
            idx += 1
            status_text.text(f"📥 FastF1 서버에서 데이터 다운로드 중: {year} {gp} ({idx}/{total_gps})")
            progress_bar.progress(idx / total_gps)
            try:
                session = fastf1.get_session(year, gp, "R")
                session.load(laps=True, telemetry=False, weather=False, messages=True)
                laps = session.laps.copy()

                needed_cols = [
                    "Driver", "DriverNumber", "LapTime", "LapNumber", "Compound",
                    "TyreLife", "PitInTime", "PitOutTime", "TrackStatus", "Position",
                    "FreshTyre", "IsAccurate", "Time", "Stint", "Team"
                ]
                available_cols = [c for c in needed_cols if c in laps.columns]
                laps = laps[available_cols]
                laps["Season"] = year
                laps["GrandPrix"] = gp
                all_laps.append(laps)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    return pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()

def filter_green_clean_laps(laps_df):
    if laps_df.empty:
        return pd.DataFrame()

    df = laps_df.copy().dropna(subset=["LapTime", "Compound", "TyreLife"])
    df = df[df["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]
    df = df[df["IsAccurate"] == True]
    df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]
    df = df[df["TrackStatus"].astype(str).str.contains("1", na=False)]
    df["LapTimeSeconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    return df

def build_tyre_model(laps_df):
    tyre_model = {}
    if laps_df.empty:
        return tyre_model

    global_mean = laps_df["LapTimeSeconds"].mean()

    for compound in ["SOFT", "MEDIUM", "HARD"]:
        df = laps_df[laps_df["Compound"] == compound].copy()
        if len(df) < 15:
            continue

        x = df["TyreLife"].astype(float).values
        y = df["LapTimeSeconds"].astype(float).values
        base_offset = y.mean() - global_mean
        
        if len(np.unique(x)) > 2:
            a, b, intercept = np.polyfit(x, y, 2)
        elif len(np.unique(x)) > 1:
            a = 0.0
            b, intercept = np.polyfit(x, y, 1)
        else:
            a, b, intercept = 0.0, 0.05, y.mean()

        recommended_stint = 15
        start_time = intercept + b * 1 + a * (1**2)
        for life in range(1, 50):
            if (intercept + b * life + a * (life**2)) - start_time >= 1.0:
                recommended_stint = life
                break

        driver_deg = {}
        for drv, grp in df.groupby("Driver"):
            if len(grp) >= 8 and len(np.unique(grp["TyreLife"])) > 1:
