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
    "Bahrain": 120,
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
# 0-2. 정책 파라미터
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
LATE_RACE_LAPS_REMAINING_THRESHOLD = 8

FORCE_ONE_STOP_IF_TYRE_LIFE_AT_LEAST = 10
NO_STOP_TIME_PENALTY = 6.0
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
        slope, intercept = np.polyfit(x, y, 1) if len(np.unique(x)) > 1 else (0.05, y.mean())

        recommended_stint = 15
        start_time = intercept + slope * 1
        for life in range(1, 50):
            if (intercept + slope * life) - start_time >= 1.0:
                recommended_stint = life
                break

        driver_deg = {}
        for drv, grp in df.groupby("Driver"):
            if len(grp) >= 8 and len(np.unique(grp["TyreLife"])) > 1:
                try:
                    s, _ = np.polyfit(grp["TyreLife"].astype(float).values, grp["LapTimeSeconds"].astype(float).values, 1)
                    driver_deg[drv] = round(float(max(0.01, min(s, slope * 2.0))), 4)
                except Exception:
                    driver_deg[drv] = round(float(slope), 4)

        tyre_model[compound] = {
            "base_offset": round(base_offset, 4),
            "deg_per_lap": round(float(slope), 4),
            "recommended_stint": int(recommended_stint),
            "driver_deg": driver_deg
        }

    return tyre_model

def build_driver_pace_model(clean_laps_df):
    if clean_laps_df.empty:
        return {}

    driver_stats = clean_laps_df.groupby("Driver")["LapTimeSeconds"].agg(["mean", "median", "count"]).reset_index()
    global_mean = clean_laps_df["LapTimeSeconds"].mean()

    return {
        row["Driver"]: {
            "base_pace": float(row["median"]),
            "pace_offset": float(row["median"] - global_mean)
        }
        for _, row in driver_stats.iterrows() if row["count"] >= 5
    }

def estimate_pit_loss_from_data(laps_df):
    if laps_df.empty:
        return {"median_pit_loss": 22.0, "recommended_max_pit_loss": 24.0}

    df = laps_df.copy().sort_values(["Driver", "Time"]).reset_index(drop=True)
    pit_losses = []

    for _, grp in df.groupby("Driver"):
        grp = grp.sort_values("LapNumber").reset_index(drop=True)
        for i in range(1, len(grp) - 1):
            if pd.notna(grp.iloc[i].get("PitInTime")) and pd.notna(grp.iloc[i + 1].get("PitOutTime")):
                if pd.notna(grp.iloc[i - 1].get("LapTime")) and pd.notna(grp.iloc[i + 1].get("LapTime")):
                    loss = (
                        pd.to_timedelta(grp.iloc[i]["LapTime"]).total_seconds()
                        + pd.to_timedelta(grp.iloc[i + 1]["LapTime"]).total_seconds()
                    ) - (pd.to_timedelta(grp.iloc[i - 1]["LapTime"]).total_seconds() * 2)
                    if 10 <= loss <= 45:
                        pit_losses.append(loss)

    if len(pit_losses) >= 5:
        return {
            "median_pit_loss": round(float(np.median(pit_losses)), 2),
            "recommended_max_pit_loss": round(float(np.percentile(pit_losses, 75)), 2)
        }

    return {"median_pit_loss": 22.0, "recommended_max_pit_loss": 24.0}

def save_preprocessed_data(raw_laps_df, clean_laps_df, tyre_model, driver_pace_model, pit_stats):
    raw_laps_df.to_csv(os.path.join(PREPROCESSED_DIR, "raw_laps.csv"), index=False)
    clean_laps_df.to_csv(os.path.join(PREPROCESSED_DIR, "clean_laps.csv"), index=False)

    with open(os.path.join(PREPROCESSED_DIR, "tyre_model.json"), "w", encoding="utf-8") as f:
        json.dump(tyre_model, f, indent=4, ensure_ascii=False)

    with open(os.path.join(PREPROCESSED_DIR, "driver_pace_model.json"), "w", encoding="utf-8") as f:
        json.dump(driver_pace_model, f, indent=4, ensure_ascii=False)

    with open(os.path.join(PREPROCESSED_DIR, "pit_stats.json"), "w", encoding="utf-8") as f:
        json.dump(pit_stats, f, indent=4, ensure_ascii=False)

def load_preprocessed_data():
    required_files = [
        os.path.join(PREPROCESSED_DIR, f)
        for f in ["raw_laps.csv", "clean_laps.csv", "tyre_model.json", "driver_pace_model.json", "pit_stats.json"]
    ]

    for f in required_files:
        if not os.path.exists(f):
            return None

    return (
        pd.read_csv(required_files[0]),
        pd.read_csv(required_files[1]),
        json.load(open(required_files[2], "r", encoding="utf-8")),
        json.load(open(required_files[3], "r", encoding="utf-8")),
        json.load(open(required_files[4], "r", encoding="utf-8"))
    )

def prepare_or_load_data():
    loaded = load_preprocessed_data()
    if loaded is not None:
        return loaded

    if not IS_SERVER:
        seasons = [2023, 2024]
        grands_prix = ["Bahrain", "Saudi Arabia", "Australia", "Japan", "Monaco"]
        
        raw_laps_df = load_race_laps(seasons, grands_prix)
        if raw_laps_df.empty:
            return None

        clean_laps_df = filter_green_clean_laps(raw_laps_df)
        tyre_model = build_tyre_model(clean_laps_df)
        driver_pace_model = build_driver_pace_model(clean_laps_df)
        pit_stats = estimate_pit_loss_from_data(raw_laps_df)

        save_preprocessed_data(raw_laps_df, clean_laps_df, tyre_model, driver_pace_model, pit_stats)
        return raw_laps_df, clean_laps_df, tyre_model, driver_pace_model, pit_stats
        
    return None

# -----------------------------
# 2. 전략 계산 보조 알고리즘
# -----------------------------
def adjust_pit_loss_for_track_status(green_pit_loss, safety_mode):
    if safety_mode == "SC":
        return round(green_pit_loss * 0.60, 2)
    elif safety_mode == "VSC":
        return round(green_pit_loss * 0.75, 2)
    return round(green_pit_loss, 2)

def get_track_params(track_name):
    return TRACK_PARAMS.get(track_name, {
        "overtake_factor": 1.0,
        "drs_factor": 1.0,
        "dirty_air_factor": 1.0,
        "traffic_factor": 1.0
    })

def estimate_current_tyre_life(current_compound, tyre_model, manual_tyre_life=None):
    if manual_tyre_life is not None and manual_tyre_life >= 1:
        return int(manual_tyre_life)
    return max(1, tyre_model[current_compound]["recommended_stint"] // 2) if current_compound in tyre_model else 8

def recommend_tyre_change_time(front_gap, rear_gap, safety_mode, current_position):
    min_gap = min(front_gap, rear_gap)
    target = 2.0 if current_position <= 3 else 2.2 if current_position <= 10 else 2.4

    if min_gap <= 1.0:
        target -= 0.2
    elif min_gap <= 2.0:
        target -= 0.1
    elif min_gap >= 5.0:
        target += 0.1

    if safety_mode == "SC":
        target += 0.15
    elif safety_mode == "VSC":
        target += 0.10

    target = max(1.8, min(target, 2.6))

    comment = (
        "기록급 피트스탑 필요" if target <= 1.9 else
        "매우 빠른 타이어 교체 필요" if target <= 2.1 else
        "상위권 유지 가능: 빠른 피트 작업 필요" if target <= 2.3 else
        "여유는 있지만 더 빠를수록 유리"
    )

    return {
        "baseline_tyre_change_time": 2.2,
        "recommended_max_tyre_change_time": round(target, 2),
        "comment": comment
    }

def traffic_penalty(front_gap, track_name):
    base = 0.80 if front_gap <= 0.5 else 0.45 if front_gap <= 1.0 else 0.22 if front_gap <= 1.8 else 0.10 if front_gap <= 2.5 else 0.0
    return base * get_track_params(track_name)["traffic_factor"]

def dirty_air_penalty(front_gap, track_name):
    base = 0.12 if front_gap <= 1.0 else 0.05 if front_gap <= 2.0 else 0.0
    return base * get_track_params(track_name)["dirty_air_factor"]

def rear_pressure_penalty(rear_gap):
    return 0.10 if rear_gap <= 0.8 else 0.05 if rear_gap <= 1.5 else 0.0

def drs_gain(front_gap, track_name, drs_available=True):
    base = 0.22 if drs_available and front_gap <= 1.0 else 0.10 if drs_available and front_gap <= 1.5 else 0.0
    return base * get_track_params(track_name)["drs_factor"]

def warmup_penalty(laps_since_stop, compound):
    return {
        "SOFT": {1: 0.35, 2: 0.10},
        "MEDIUM": {1: 0.65, 2: 0.22},
        "HARD": {1: 0.95, 2: 0.38}
    }.get(compound, {}).get(laps_since_stop, 0.0)

def undercut_bonus(laps_since_stop, compound, front_gap, track_name):
    gain = {
        "SOFT": {1: 0.55, 2: 0.22},
        "MEDIUM": {1: 0.38, 2: 0.15},
        "HARD": {1: 0.18, 2: 0.08}
    }.get(compound, {}).get(laps_since_stop, 0.0)

    gain += 0.10 if front_gap <= 1.5 else -0.05 if front_gap >= 4.0 else 0.0
    return max(0.0, gain * get_track_params(track_name)["overtake_factor"])

def safety_car_deg_factor(safety_mode):
    return 0.30 if safety_mode == "SC" else 0.55 if safety_mode == "VSC" else 1.00

def build_recent_pace_lookup(clean_laps_df, track_name, current_lap, lookback=RECENT_LAPS_FOR_PACE):
    df = clean_laps_df[clean_laps_df["GrandPrix"].str.lower() == track_name.lower()]
    df = df[df["LapNumber"] < current_lap]
    if df.empty:
        return {}

    return {
        drv: float(grp.sort_values("LapNumber")["LapTimeSeconds"].tail(lookback).median())
        for drv, grp in df.groupby("Driver") if len(grp) >= 2
    }

def get_effective_pace_offset(driver, driver_pace_model, recent_pace_lookup, base_lap):
    long_term = driver_pace_model.get(driver, {}).get("pace_offset", 0.0)
    if driver in recent_pace_lookup:
        return 0.45 * long_term + 0.55 * (recent_pace_lookup[driver] - base_lap)
    return long_term

def generate_strategy_candidates(total_laps, current_lap, tyre_model, current_tyre_life, allow_zero_stop=ALLOW_ZERO_STOP_DEFAULT):
    candidates = []
    tyre_types = list(tyre_model.keys())
    remaining_laps = total_laps - current_lap + 1

    allow_zero = True if ALLOW_ZERO_STOP_ONLY_IF_LATE_RACE and remaining_laps <= LATE_RACE_LAPS_REMAINING_THRESHOLD else allow_zero_stop
    if current_tyre_life < FORCE_ONE_STOP_IF_TYRE_LIFE_AT_LEAST and allow_zero:
        candidates.append([])

    for next_tyre in tyre_types:
        rec1 = tyre_model[next_tyre]["recommended_stint"]
        for pit1 in range(current_lap + EARLIEST_PIT_AFTER_CURRENT, min(total_laps - 1, current_lap + rec1 + STINT_EXTRA_MARGIN) + 1):
            candidates.append([{"pit_lap": pit1, "next_tyre": next_tyre}])

    if remaining_laps >= 12:
        for t1, t2 in product(tyre_types, repeat=2):
            rec1 = tyre_model[t1]["recommended_stint"]
            rec2 = tyre_model[t2]["recommended_stint"]
            for pit1 in range(current_lap + EARLIEST_PIT_AFTER_CURRENT, min(total_laps - MIN_LAPS_BETWEEN_STOPS - 1, current_lap + rec1 + STINT_EXTRA_MARGIN) + 1):
                for pit2 in range(pit1 + MIN_LAPS_BETWEEN_STOPS, min(total_laps - 1, pit1 + rec2 + STINT_EXTRA_MARGIN) + 1):
                    candidates.append([
                        {"pit_lap": pit1, "next_tyre": t1},
                        {"pit_lap": pit2, "next_tyre": t2}
                    ])

    if len(candidates) > MAX_TOTAL_CANDIDATES:
        candidates = [candidates[i] for i in np.linspace(0, len(candidates) - 1, MAX_TOTAL_CANDIDATES, dtype=int)]

    return candidates

def build_rival_states_from_reference(raw_laps_df, track_name, current_lap, my_driver, driver_pace_model, recent_pace_lookup, base_lap, tyre_model, total_laps):
    df = raw_laps_df[
        (raw_laps_df["GrandPrix"].str.lower() == track_name.lower()) &
        (raw_laps_df["LapNumber"] == current_lap - 1)
    ].dropna(subset=["Driver", "Position", "Time"]).sort_values("Position")

    if df.empty:
        return [], 0.0

    leader_time = pd.to_timedelta(df.iloc[0]["Time"]).total_seconds()
    my_row = df[df["Driver"] == my_driver]
    my_initial_race_time = max(0.0, pd.to_timedelta(my_row.iloc[0]["Time"]).total_seconds() - leader_time) if not my_row.empty else 0.0

    rivals = []
    for _, row in df[df["Driver"] != my_driver].head(MAX_RIVALS).iterrows():
        comp = row["Compound"] if row.get("Compound") in ["SOFT", "MEDIUM", "HARD"] else "MEDIUM"
        try:
            tyre_life = int(float(row["TyreLife"])) if pd.notna(row.get("TyreLife")) else 8
        except Exception:
            tyre_life = 8

        race_time = max(0.0, pd.to_timedelta(row["Time"]).total_seconds() - leader_time)
        pace_offset = get_effective_pace_offset(row["Driver"], driver_pace_model, recent_pace_lookup, base_lap)

        rec_stint = tyre_model.get(comp, {"recommended_stint": 15})["recommended_stint"]
        pit_cand = current_lap + max(1, rec_stint - tyre_life)
        rival_strat = [{"pit_lap": pit_cand, "next_tyre": "HARD" if comp == "MEDIUM" else "MEDIUM"}] if pit_cand < total_laps - 5 else []

        rivals.append({
            "driver": row["Driver"],
            "race_time": float(race_time),
            "position": int(row["Position"]),
            "compound": comp,
            "tyre_life": tyre_life,
            "laps_since_stop": tyre_life,
            "pace_offset": pace_offset,
            "front_gap": 99.0,
            "rear_gap": 99.0,
            "strategy": rival_strat,
            "strategy_index": 0
        })

    for i in range(len(rivals)):
        rivals[i]["front_gap"] = 99.0 if i == 0 else max(0.2, rivals[i]["race_time"] - rivals[i - 1]["race_time"])
        rivals[i]["rear_gap"] = 99.0 if i == len(rivals) - 1 else max(0.2, rivals[i + 1]["race_time"] - rivals[i]["race_time"])

    return rivals, my_initial_race_time

def build_fallback_rival_states(selected_rivals, driver_pace_model, recent_pace_lookup, base_lap, rng):
    rivals = []
    for i, drv in enumerate(selected_rivals):
        pace_offset = get_effective_pace_offset(drv, driver_pace_model, recent_pace_lookup, base_lap)
        rivals.append({
            "driver": drv,
            "race_time": float(rng.uniform(0.5, 6.5)),
            "position": i + 1,
            "compound": ["MEDIUM", "HARD", "SOFT"][i % 3],
            "tyre_life": int(rng.integers(4, 16)),
            "laps_since_stop": int(rng.integers(4, 16)),
            "pace_offset": pace_offset,
            "front_gap": float(rng.uniform(0.7, 3.5)),
            "rear_gap": float(rng.uniform(0.7, 3.5)),
            "strategy": [],
            "strategy_index": 0
        })
    return rivals

def clone_car_state(car):
    cloned = car.copy()
    cloned["strategy"] = [x.copy() for x in car["strategy"]]
    return cloned

def clone_rivals(rivals):
    return [clone_car_state(r) for r in rivals]

# [수정 가미] 연료 소모 패널티 계산을 위해 current_lap, total_laps 인자 추가
def predict_driver_lap_time(driver, base_lap, pace_offset, compound, tyre_life, tyre_model, front_gap, rear_gap, drs_available, laps_since_stop, rng, track_name, current_lap, total_laps, safety_mode="NONE"):
    info = tyre_model.get(compound, {"base_offset": 0.0, "deg_per_lap": 0.05, "driver_deg": {}})
    deg = info.get("driver_deg", {}).get(driver, info["deg_per_lap"])
    
    # 연료 소모에 따른 무게 패널티 연산 (남은 바퀴 수가 많을수록 차가 무거우므로 느려짐)
    fuel_weight_penalty = (total_laps - current_lap) * 0.06
    
    lap_time = base_lap + pace_offset + info["base_offset"] + (deg * safety_car_deg_factor(safety_mode)) * tyre_life + fuel_weight_penalty
    noise = rng.normal(0, 0.18)

    if tyre_life >= VERY_OLD_TYRE_THRESHOLD:
        lap_time += (tyre_life - VERY_OLD_TYRE_THRESHOLD + 1) * OLD_TYRE_EXTRA_PENALTY_PER_LAP

    if safety_mode == "SC":
        return (lap_time * 1.32) + noise
    if safety_mode == "VSC":
        return (lap_time * 1.12) + noise

    return (
        lap_time
        + traffic_penalty(front_gap, track_name)
        + dirty_air_penalty(front_gap, track_name)
        + rear_pressure_penalty(rear_gap)
        - drs_gain(front_gap, track_name, drs_available)
        + warmup_penalty(laps_since_stop, compound)
        - undercut_bonus(laps_since_stop, compound, front_gap, track_name)
        + noise
    )

def can_overtake(gap_to_front, lap_advantage, front_gap, track_name, drs_available):
    p = get_track_params(track_name)
    return (
        (gap_to_front <= OVERTAKE_BASE_GAP * p["overtake_factor"]) and
        (lap_advantage >= (OVERTAKE_MIN_ADVANTAGE / max(p["overtake_factor"], 0.35)) * (0.88 if drs_available else 1.0)) and
        (front_gap <= 1.5)
    )

def update_positions_with_overtake_logic(all_cars, lap_times, track_name):
    all_cars.sort(key=lambda x: x["race_time"])
    changed, cnt = True, 0

    while changed and cnt < 10:
        changed, cnt = False, cnt + 1
        for i in range(1, len(all_cars)):
            front, back = all_cars[i - 1], all_cars[i]
            gap = max(0.0, back["race_time"] - front["race_time"])
            if can_overtake(gap, lap_times[front["driver"]] - lap_times[back["driver"]], gap, track_name, gap <= 1.0) and back["race_time"] < front["race_time"] + 0.08:
                all_cars[i - 1], all_cars[i] = all_cars[i], all_cars[i - 1]
                changed = True

    for idx, car in enumerate(all_cars):
        car["position"] = idx + 1
        car["front_gap"] = 99.0 if idx == 0 else max(0.2, car["race_time"] - all_cars[idx - 1]["race_time"])
        car["rear_gap"] = 99.0 if idx == len(all_cars) - 1 else max(0.2, all_cars[idx + 1]["race_time"] - car["race_time"])

def simulate_race_once(total_laps, current_lap, base_lap, tyre_model, my_state, rivals, adjusted_pit_loss, rng, track_name, safety_mode="NONE"):
    my_state, rivals = clone_car_state(my_state), clone_rivals(rivals)
    all_cars = [my_state] + rivals

    sc_rem = min(4, max(2, (total_laps - current_lap + 1) // 6)) if safety_mode == "SC" else 0
    vsc_rem = min(2, max(1, (total_laps - current_lap + 1) // 10)) if safety_mode == "VSC" else 0

    for lap in range(current_lap, total_laps + 1):
        mode = "SC" if sc_rem > 0 else "VSC" if vsc_rem > 0 else "NONE"

        if sc_rem > 0:
            sc_rem -= 1
        elif vsc_rem > 0:
            vsc_rem -= 1

        lap_times = {}

        for car in all_cars:
            just_pitted = False

            if car["strategy_index"] < len(car["strategy"]) and lap == car["strategy"][car["strategy_index"]]["pit_lap"]:
                penalty = adjusted_pit_loss
                r = rng.random()
                if r < 0.02:
                    penalty += rng.uniform(4.0, 7.0)
                elif r < 0.08:
                    penalty += rng.uniform(1.2, 2.5)

                car["race_time"] += penalty
                car["compound"] = car["strategy"][car["strategy_index"]]["next_tyre"]
                car["tyre_life"], car["laps_since_stop"], car["strategy_index"], just_pitted = 0, 1, car["strategy_index"] + 1, True

            # [수정 가미] 랩타임 계산 시 현재 루프의 lap과 total_laps 변수를 인자로 전달하도록 수정
            lt = predict_driver_lap_time(
                car["driver"], base_lap, car["pace_offset"], car["compound"], car["tyre_life"],
                tyre_model, car["front_gap"], car.get('rear_gap', 2.0), car["front_gap"] <= 1.0,
                car["laps_since_stop"], rng, track_name, lap, total_laps, mode
            )

            if just_pitted:
                lt += 0.45 if car["front_gap"] <= 1.5 else 0.20 if car["front_gap"] <= 3.0 else 0.0

            car["race_time"] += lt
            car["tyre_life"] += 1
            car["laps_since_stop"] += 1
            lap_times[car["driver"]] = lt

        update_positions_with_overtake_logic(all_cars, lap_times, track_name)

    me = next(c for c in all_cars if c["driver"] == my_state["driver"])
    return {"finish_time": round(me["race_time"], 2), "position": int(me["position"])}

def simulate_many(total_laps, current_lap, base_lap, tyre_model, my_state, rivals, adjusted_pit_loss, track_name, safety_mode="NONE", n=300, seed=42):
    rng = np.random.default_rng(seed)
    results = [
        simulate_race_once(total_laps, current_lap, base_lap, tyre_model, my_state, rivals, adjusted_pit_loss, rng, track_name, safety_mode)
        for _ in range(n)
    ]
    df = pd.DataFrame(results)

    return {
        "expected_finish_time": round(float(np.mean(df["finish_time"])), 2),
        "finish_time_std": round(float(np.std(df["finish_time"])), 4),
        "expected_position": round(float(np.mean(df["position"])), 2),
        "most_likely_position": int(df["position"].mode().iloc[0])
    }

def sort_result_df(result_df):
    return result_df.sort_values(
        by=["strategy_score", "expected_finish_time", "expected_position", "finish_time_std"],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)

def strategy_to_row(strategy, sim, current_tyre_life):
    score = TIME_PRIORITY_WEIGHT * sim["expected_finish_time"] + POSITION_PRIORITY_WEIGHT * sim["expected_position"]
    penalty = 0.0

    if len(strategy) == 0:
        penalty = NO_STOP_TIME_PENALTY + (
            max(0, current_tyre_life - FORCE_ONE_STOP_IF_TYRE_LIFE_AT_LEAST + 1) * 0.45
            if current_tyre_life >= FORCE_ONE_STOP_IF_TYRE_LIFE_AT_LEAST else 0.0
        )

    return {
        "stops": len(strategy),
        "pit_laps": [x["pit_lap"] for x in strategy],
        "next_tyres": [x["next_tyre"] for x in strategy],
        "expected_finish_time": sim["expected_finish_time"],
        "finish_time_std": sim["finish_time_std"],
        "expected_position": sim["expected_position"],
        "most_likely_position": sim["most_likely_position"],
        "strategy_score": round(float(score + penalty), 4),
        "no_stop_penalty": round(float(penalty), 4)
    }

def build_my_state(my_driver, current_position, current_compound, current_tyre_life, front_gap, rear_gap, driver_pace_model, recent_pace_lookup, base_lap, strategy, my_initial_race_time):
    return {
        "driver": my_driver,
        "race_time": my_initial_race_time,
        "position": current_position,
        "compound": current_compound,
        "tyre_life": current_tyre_life,
        "laps_since_stop": current_tyre_life,
        "pace_offset": get_effective_pace_offset(my_driver, driver_pace_model, recent_pace_lookup, base_lap),
        "front_gap": front_gap,
        "rear_gap": rear_gap,
        "strategy": strategy,
        "strategy_index": 0
    }

def simulate_strategy_job(args):
    strategy, total_laps, current_lap, base_lap, tyre_model, current_position, current_compound, current_tyre_life, front_gap, rear_gap, driver_pace_model, recent_pace_lookup, my_driver, rivals, adjusted_pit_loss, track_name, safety_mode, n, seed, my_initial_race_time = args

    my_state = build_my_state(
        my_driver, current_position, current_compound, current_tyre_life,
        front_gap, rear_gap, driver_pace_model, recent_pace_lookup,
        base_lap, strategy, my_initial_race_time
    )

    sim = simulate_many(total_laps, current_lap, base_lap, tyre_model, my_state, rivals, adjusted_pit_loss, track_name, safety_mode, n, seed)
    return strategy_to_row(strategy, sim, current_tyre_life)

def run_batch_simulations(categories_or_strategies, total_laps, current_lap, base_lap, tyre_model, current_position, current_compound, current_tyre_life, front_gap, rear_gap, driver_pace_model, recent_pace_lookup, my_driver, rivals, adjusted_pit_loss, track_name, safety_mode, n, seed_base, my_initial_race_time):
    jobs = [
        (
            strat, total_laps, current_lap, base_lap, tyre_model, current_position,
            current_compound, current_tyre_life, front_gap, rear_gap, driver_pace_model,
            recent_pace_lookup, my_driver, rivals, adjusted_pit_loss, track_name,
            safety_mode, n, seed_base + idx, my_initial_race_time
        )
        for idx, strat in enumerate(categories_or_strategies)
    ]

    if USE_MULTIPROCESSING and len(jobs) >= 4:
        with Pool(processes=max(1, min(cpu_count() - 1, 6))) as pool:
            return pool.map(simulate_strategy_job, jobs)

    return [simulate_strategy_job(j) for j in jobs]

def evaluate_strategies(total_laps, current_lap, current_compound, current_position, front_gap, rear_gap, base_lap, tyre_model, adjusted_pit_loss, driver_pace_model, my_driver, track_name, raw_laps_df, clean_laps_df, safety_mode, current_tyre_life):
    candidates = generate_strategy_candidates(total_laps, current_lap, tyre_model, current_tyre_life)
    recent_pace_lookup = build_recent_pace_lookup(clean_laps_df, track_name, current_lap)

    rivals, my_initial_race_time = build_rival_states_from_reference(
        raw_laps_df, track_name, current_lap, my_driver, driver_pace_model,
        recent_pace_lookup, base_lap, tyre_model, total_laps
    )

    if not rivals:
        friend_names = [d for d in driver_pace_model.keys() if d != my_driver][:MAX_RIVALS]
        while len(friend_names) < MAX_RIVALS:
            friend_names.append(f"RIVAL{len(friend_names)+1}")
        rivals = build_fallback_rival_states(friend_names[:MAX_RIVALS], driver_pace_model, recent_pace_lookup, base_lap, np.random.default_rng(2026))
        my_initial_race_time = 2.5

    coarse = run_batch_simulations(
        candidates, total_laps, current_lap, base_lap, tyre_model, current_position,
        current_compound, current_tyre_life, front_gap, rear_gap, driver_pace_model,
        recent_pace_lookup, my_driver, rivals, adjusted_pit_loss, track_name,
        safety_mode, COARSE_SIM_N, 1000, my_initial_race_time
    )
    coarse_df = sort_result_df(pd.DataFrame(coarse))
    if coarse_df.empty:
        return coarse_df

    mid_strats = [
        [{"pit_lap": l, "next_tyre": t} for l, t in zip(r["pit_laps"], r["next_tyres"])]
        for _, r in coarse_df.head(min(TOPK_MID, len(coarse_df))).iterrows()
    ]
    mid = run_batch_simulations(
        mid_strats, total_laps, current_lap, base_lap, tyre_model, current_position,
        current_compound, current_tyre_life, front_gap, rear_gap, driver_pace_model,
        recent_pace_lookup, my_driver, rivals, adjusted_pit_loss, track_name,
        safety_mode, MID_SIM_N, 3000, my_initial_race_time
    )
    mid_df = sort_result_df(pd.DataFrame(mid))

    final_strats = [
        [{"pit_lap": l, "next_tyre": t} for l, t in zip(r["pit_laps"], r["next_tyres"])]
        for _, r in mid_df.head(min(TOPK_FINAL, len(mid_df))).iterrows()
    ]
    final = run_batch_simulations(
        final_strats, total_laps, current_lap, base_lap, tyre_model, current_position,
        current_compound, current_tyre_life, front_gap, rear_gap, driver_pace_model,
        recent_pace_lookup, my_driver, rivals, adjusted_pit_loss, track_name,
        safety_mode, FINAL_SIM_N, 5000, my_initial_race_time
    )

    result_df = sort_result_df(pd.DataFrame(final))

    pit_df = result_df[result_df["stops"] > 0].copy()
    stay_df = result_df[result_df["stops"] == 0].copy()

    if not pit_df.empty and not stay_df.empty:
        best_p = sort_result_df(pit_df).iloc[0]
        best_s = sort_result_df(stay_df).iloc[0]

        if (
            (best_s["expected_finish_time"] - best_p["expected_finish_time"] >= MIN_TIME_GAIN_TO_PIT) or
            (best_s["expected_position"] - best_p["expected_position"] >= MIN_POSITION_GAIN_TO_PIT) or
            (current_tyre_life >= FORCE_ONE_STOP_IF_TYRE_LIFE_AT_LEAST)
        ):
            result_df = pd.concat([pd.DataFrame([best_p]), sort_result_df(pit_df).iloc[1:]], ignore_index=True)

    return sort_result_df(result_df)

def recommend_stop_count(result_df):
    if result_df.empty:
        return {"best_stop_count": None, "summary_table": pd.DataFrame(), "comment": "전략 데이터가 없습니다."}

    summary = result_df.groupby("stops").agg({
        "expected_position": "mean",
        "expected_finish_time": "mean",
        "finish_time_std": "mean",
        "strategy_score": "mean"
    }).reset_index().sort_values(by=["strategy_score", "expected_finish_time", "expected_position"])

    best_cnt = int(summary.iloc[0]["stops"])

    return {
        "best_stop_count": best_cnt,
        "summary_table": summary,
        "comment": "시간 기준으로는 무피트도 가능하지만, 참고용입니다." if best_cnt == 0 else f"시간 기준으로 가장 유리한 전략군은 {best_cnt}회 피트 전략입니다."
    }

def normalize_track_name(track_name):
    for t in ["Bahrain", "Saudi Arabia", "Australia", "Japan", "Monaco"]:
        if track_name.strip().lower() == t.lower():
            return t
    return None

def format_strategy_display(result_df):
    display_df = result_df.copy()
    display_df["pit_laps"] = display_df["pit_laps"].apply(lambda x: " - ".join(map(str, x)) if x else "No Stop")
    display_df["next_tyres"] = display_df["next_tyres"].apply(lambda x: " → ".join(x) if x else "-")
    display_df = display_df.rename(columns={
        "stops": "Stops",
        "pit_laps": "White Window",
        "next_tyres": "Tyre Plan",
        "expected_finish_time": "Exp Finish Time",
        "finish_time_std": "Std",
        "expected_position": "Exp Position",
        "most_likely_position": "Likely Position",
        "strategy_score": "Score",
        "no_stop_penalty": "No Stop Penalty"
    })
    return display_df

# -----------------------------
# 3. Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="F1 Race Strategy Simulator", layout="wide")
    inject_custom_css()

    loaded = prepare_or_load_data()
    if loaded is None:
        st.error("데이터를 불러오거나 저장하는 데 실패했습니다.")
        return

    raw_laps_df, clean_laps_df, tyre_model, driver_pace_model, pit_stats = loaded
    base_lap = clean_laps_df["LapTimeSeconds"].astype(float).mean()

    green_pit_loss = pit_stats["median_pit_loss"]

    main_left, main_right = st.columns([0.92, 1.68])

    with main_left:
        st.sidebar.header("Race Control Input")

        selected_driver_label = st.sidebar.selectbox(
            "시뮬레이션할 내 드라이버 선택",
            list(DRIVER_OPTIONS.keys())
        )
        my_driver = DRIVER_OPTIONS[selected_driver_label]

        track_name_input = st.sidebar.selectbox(
            "현재 트랙 이름",
            ["Bahrain", "Saudi Arabia", "Australia", "Japan", "Monaco"]
        )
        track_name = normalize_track_name(track_name_input)

        total_laps = st.sidebar.number_input("총 랩 수", min_value=1, max_value=100, value=57)
        current_lap = st.sidebar.number_input("현재 랩", min_value=1, max_value=100, value=25)
        current_compound = st.sidebar.selectbox("현재 타이어 타입", ["SOFT", "MEDIUM", "HARD"], index=1)
        current_tyre_life_manual = st.sidebar.number_input(
            "현재 타이어 사용 랩 수 (모르면 0)",
            min_value=0,
            max_value=60,
            value=12
        )
        current_position = st.sidebar.number_input("현재 순위(Position)", min_value=1, max_value=20, value=3)
        front_gap = st.sidebar.number_input("앞차와의 간격(초)", min_value=0.0, max_value=60.0, value=1.2, step=0.1)
        rear_gap = st.sidebar.number_input("뒷차와의 간격(초)", min_value=0.0, max_value=60.0, value=2.5, step=0.1)
        safety_mode = st.sidebar.selectbox("세이프티카 여부", ["NONE", "SC", "VSC"])

        use_auto_pit_loss = st.sidebar.radio(
            "피트 손실시간 자동 계산 여부",
            ["자동계산 사용(Y)", "수동 입력(N)"]
        )
        if "자동계산" not in use_auto_pit_loss:
            green_pit_loss = st.sidebar.number_input(
                "그린 플래그 기준 피트 손실시간(초)",
                min_value=10.0,
                max_value=50.0,
                value=22.0,
                step=0.5
            )

        st.sidebar.markdown("---")
        start_calc = st.sidebar.button("시뮬레이션 실행 및 최적 전략 계산")

        st.markdown(
            '''
            <div class="hero-card">
                <div class="hero-title">F1 Race Strategy Simulator</div>
                <div class="hero-sub">
                    FastF1 기반 실주행 랩 데이터를 사용해 현재 레이스 상황에서
                    가장 유리한 피트 전략을 몬테카를로 방식으로 예측합니다.
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-label">💡 시스템 안내 보드 (System Guide)</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <ul style="margin-bottom: 30px; padding-left: 20px; color: #98a2b3; font-size: 0.9rem;">
                <li><b>실시간 데이터 동기화</b>: 좌측 사이드바 제어창에서 선택된 옵션들은 우측 모니터링 보드와 실시간 연동됩니다.</li>
                <li><b>몬테카를로 시뮬레이션 알고리즘</b>: FastF1 실데이터 모델링을 기반으로 수백 가지 레이스 시나리오를 예측 연산합니다.</li>
            </ul>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-label">⚙️ 레이스 컨트롤 전략 보조 가이드</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <ul style="margin-bottom: 30px; padding-left: 20px; color: #98a2b3; font-size: 0.9rem;">
                <li><b>트랙 성향 인자 자동 연산</b>: 서킷별 DRS 효율, Dirty Air 영향성 및 교통(Traffic) 정체 패널티가 상시 반영 중입니다.</li>
                <li><b>실시간 연산 준비</b>: 입력 데이터를 확인하신 후 좌측 사이드바 하단의 주황색 트리거 버튼을 눌러 시뮬레이션을 개시하세요.</li>
            </ul>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-label">🔧 피트 레인 손실 추정치</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 0.9rem; color: #98a2b3; margin-bottom: 15px;">'
            '• 경주용 차가 새로운 타이어로 갈아끼우기 위해 피트 레인을 통과할 때 손해 보는 총 시간입니다.'
            '</div>',
            unsafe_allow_html=True
        )

        m1, m2 = st.columns(2)
        with m1:
            st.metric("중앙값 피트 손실", f"{pit_stats['median_pit_loss']} 초")
        with m2:
            st.metric("권장 최대값", f"{pit_stats['recommended_max_pit_loss']} 초")

        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="section-label">🛞 타이어 열화율</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 0.9rem; color: #98a2b3; margin-bottom: 10px;">'
            '• 주행할수록 타이어가 닳아 한 바퀴를 도는 데 시간이 얼마나 더 걸리는지(초) 나타낸 열화 모델입니다.'
            '</div>',
            unsafe_allow_html=True
        )

        tyre_table = [
            [t, i["base_offset"], i["deg_per_lap"], i["recommended_stint"]]
            for t, i in tyre_model.items()
        ]
        st.dataframe(
            pd.DataFrame(
                tyre_table,
                columns=["타이어", "성능차(초)", "열화율", "권장 스틴트(랩)"]
            ),
            use_container_width=True,
            hide_index=True
        )

    with main_right:
        right_stage = st.empty()

        if not start_calc:
            with right_stage.container():
                st.markdown(f"<h2>🏎️ 현재 선택된 서킷: {track_name}</h2>", unsafe_allow_html=True)
                path = TRACK_IMAGES_PATHS.get(track_name)
                if path and path.exists():
                    st.image(str(path), use_container_width=True)

        else:
            adjusted_pit_loss = adjust_pit_loss_for_track_status(green_pit_loss, safety_mode)
            current_tyre_life = estimate_current_tyre_life(
                current_compound,
                tyre_model,
                current_tyre_life_manual if current_tyre_life_manual > 0 else None
            )
            tyre_change_info = recommend_tyre_change_time(front_gap, rear_gap, safety_mode, current_position)

            with right_stage.container():
                st.markdown("<div style='height: 72px;'></div>", unsafe_allow_html=True)
                st.info("몬테카를로 시뮬레이션 연산을 시작합니다. 잠시만 기다려주세요.")

                with st.spinner("수백 개의 조합을 기반으로 몬테카를로 시뮬레이션 실행 중..."):
                    result_df = evaluate_strategies(
                        total_laps=total_laps,
                        current_lap=current_lap,
                        current_compound=current_compound,
                        current_position=current_position,
                        front_gap=front_gap,
                        rear_gap=rear_gap,
                        base_lap=base_lap,
                        tyre_model=tyre_model,
                        adjusted_pit_loss=adjusted_pit_loss,
                        driver_pace_model=driver_pace_model,
                        my_driver=my_driver,
                        track_name=track_name,
                        raw_laps_df=raw_laps_df,
                        clean_laps_df=clean_laps_df,
                        safety_mode=safety_mode,
                        current_tyre_life=current_tyre_life
                    )

            with right_stage.container():
                st.markdown(f"<h2>🏎️ 현재 선택된 서킷: {track_name}</h2>", unsafe_allow_html=True)

                if result_df.empty:
                    st.warning("전략 계산 결과가 없습니다. 현재 랩이 너무 경기 후반일 수 있습니다.")
                else:
                    stop_count_info = recommend_stop_count(result_df)
                    best = result_df.iloc[0]
                    possible_stops = sorted(result_df["stops"].unique().tolist())

                    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

                    res_left, res_right = st.columns([0.94, 1.30], gap="large")

                    with res_left:
                        left_pad, left_content = st.columns([0.08, 0.92])

                        with left_pad:
                            st.markdown("")

                        with left_content:
                            st.markdown('<div class="section-label">=== 피트 횟수 분석 ===</div>', unsafe_allow_html=True)
                            st.dataframe(stop_count_info['summary_table'], use_container_width=True, hide_index=True)
                            st.info(stop_count_info['comment'])

                            st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

                            st.markdown('<div class="section-label">=== 추천 전략 TOP 10 ===</div>', unsafe_allow_html=True)
                            st.dataframe(result_df.head(10), use_container_width=True, hide_index=True)

                            st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

                            st.metric("예상 평균 순위", f"{best['expected_position']} 위")
                            st.metric("예상 가능성 순위", f"{best['most_likely_position']} 위")
                            st.metric("완주 시간 변동성(표준편차)", f"{best['finish_time_std']}")

                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    with res_right:
                        
                        st.markdown('<div class="section-label">=== 최종 추천 브리핑 ===</div>', unsafe_allow_html=True)

                        report_markdown = f"""
* **드라이버**: {selected_driver_label} ({my_driver})
* **트랙**: {track_name}
* **세이프티카 상태**: {safety_mode}
* **현재 추정 타이어 라이프**: {current_tyre_life}랩
* **일반 주행 기준 피트 손실시간**: {green_pit_loss}초
* **현재 상황 반영 피트 손실시간**: {adjusted_pit_loss}초

* **기본 타이어 교체 시간 기준**: {tyre_change_info['baseline_tyre_change_time']}초
* **추천 최대 타이어 교체 시간**: {tyre_change_info['recommended_max_tyre_change_time']}초
* **해석**: {tyre_change_info['comment']}

* **이번 경기에서 고려 가능한 피트 횟수**: {possible_stops}회입니다.
* **데이터상 추천되는 피트 횟수**: 약 {stop_count_info['best_stop_count']}회입니다.
"""
                        st.markdown(report_markdown)

                        if best["stops"] == 0:
                            st.warning(
                                "이 결과는 참고용 무피트 전략입니다.\n\n"
                                "**추천 다음 타이어:** 현재 타이어 유지"
                            )
                        else:
                            st.success(
                                f"이때 추천 피트 랩은 \n**{best['pit_laps']}**입니다.\n\n"
                                f"**추천 다음 타이어:** {best['next_tyres']}"
                            )

                        st.write(f"⏱️ **예상 평균 남은 경기 시간:** `{best['expected_finish_time']}초`")
                        st.write(f"🎯 **전략 종합 점수(낮을수록 유리):** `{best['strategy_score']}`")

                        st.markdown('<div style="margin-top:15px;"></div>', unsafe_allow_html=True)

                        if best["stops"] == 0:
                            st.info(
                                "💡 **추천:** 아주 후반전이 아니라면 무피트 전략은 참고만 하고, "
                                "실전에서는 1회 피트 전략도 함께 비교하는 것이 좋습니다."
                            )
                        else:
                            st.success(
                                f"💡 **추천:** 현재 상황에서는 타이어 교체를 최대 "
                                f"**{tyre_change_info['recommended_max_tyre_change_time']}초** 이내에 끝내고, "
                                f"**{best['pit_laps']}랩**에 피트하는 전략이 가장 유리합니다."
                            )
                        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
