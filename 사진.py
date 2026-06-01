import os
import json
import time
from pathlib import Path
from itertools import product
from multiprocessing import Pool, cpu_count

import fastf1
import pandas as pd
import numpy as np
import streamlit as st

# =============================
# 0. 기본 경로 / 앱 설정
# =============================
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / ".fastf1_cache"
PREPROCESSED_DIR = BASE_DIR / "preprocessed"
ASSETS_DIR = BASE_DIR / "assets"

TRACK_IMAGES_PATHS = {
    "Bahrain": ASSETS_DIR / "tracks" / "bahrain.jpg",
    "Saudi Arabia": ASSETS_DIR / "tracks" / "saudi_arabia.jpg",
    "Australia": ASSETS_DIR / "tracks" / "australia.jpg",
    "Japan": ASSETS_DIR / "tracks" / "japan.jpg",
    "Monaco": ASSETS_DIR / "tracks" / "monaco.jpg"
}

LOGO_PATH = ASSETS_DIR / "logos" / "f1_logo.jpg"

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

# 폴더 생성
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Streamlit 페이지 설정
st.set_page_config(page_title="F1 Race Strategy Simulator", layout="wide")

# FastF1 캐시 활성화
if "fastf1_cache_enabled" not in st.session_state:
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    st.session_state["fastf1_cache_enabled"] = True

# =============================
# 1. 정책 파라미터
# =============================
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

# =============================
# 2. 디자인 CSS
# =============================
def inject_custom_css():
    st.markdown("""
    <style>
    :root {
        --bg: #0b0f14;
        --bg-2: #111827;
        --panel: rgba(20, 26, 34, 0.92);
        --panel-soft: rgba(26, 34, 48, 0.88);
        --text: #f5f7fb;
        --muted: #98a2b3;
        --accent: #ff4d4f;
        --accent-2: #ff7a45;
        --line: rgba(255,255,255,0.08);
        --shadow: 0 10px 30px rgba(0,0,0,0.22);
        --radius: 22px;
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(255,77,79,0.14), transparent 24%),
            radial-gradient(circle at bottom left, rgba(255,122,69,0.10), transparent 20%),
            linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: #0f141c;
        border-right: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: var(--muted);
    }

    h1, h2, h3, h4 {
        color: var(--text);
        letter-spacing: -0.02em;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(255,77,79,0.14), rgba(20,26,34,0.96));
        border: 1px solid var(--line);
        border-radius: 26px;
        padding: 30px;
        box-shadow: var(--shadow);
        min-height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 20px;
    }

    .hero-badge {
        display: inline-block;
        width: fit-content;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,77,79,0.14);
        color: #ffd2d2;
        border: 1px solid rgba(255,255,255,0.08);
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 16px;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text);
        line-height: 1.15;
        margin-bottom: 0.55rem;
    }

    .hero-sub {
        font-size: 1rem;
        color: var(--muted);
        line-height: 1.7;
        max-width: 60ch;
    }

    .custom-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 22px;
        box-shadow: var(--shadow);
        margin-bottom: 18px;
    }

    .card-title {
        color: var(--text);
        font-size: 1.02rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .card-subtitle {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 1rem;
        line-height: 1.6;
    }

    div[data-testid="stMetric"] {
        background: var(--panel-soft);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 14px;
        box-shadow: none;
    }

    div[data-testid="stDataFrame"] {
        background: transparent;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 8px;
    }

    .stAlert {
        border-radius: 16px;
        border: 1px solid var(--line);
    }

    .stButton > button {
        width: 100%;
        border: 0;
        border-radius: 15px;
        background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
        color: white;
        font-weight: 800;
        padding: 0.95rem 1.2rem;
        box-shadow: 0 10px 24px rgba(255, 77, 79, 0.18);
    }

    .summary-box {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px;
        margin-top: 8px;
    }

    .summary-box ul {
        margin: 0;
        padding-left: 1rem;
        color: var(--muted);
    }

    .summary-box li {
        margin-bottom: 0.45rem;
    }

    .mini-note {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.65;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================
# 3. UI 헬퍼
# =============================
def render_card_start(title, subtitle=None):
    sub_html = f'<div class="card-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="custom-card">
            <div class="card-title">{title}</div>
            {sub_html}
        """,
        unsafe_allow_html=True
    )

def render_card_end():
    st.markdown("</div>", unsafe_allow_html=True)

def load_image_binary(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

def format_strategy_table(result_df):
    if result_df.empty:
        return result_df

    display_df = result_df.copy()
    display_df["pit_laps"] = display_df["pit_laps"].apply(
        lambda x: "No Stop" if not x else " / ".join(map(str, x))
    )
    display_df["next_tyres"] = display_df["next_tyres"].apply(
        lambda x: "-" if not x else " → ".join(x)
    )

    display_df = display_df.rename(columns={
        "stops": "Stops",
        "pit_laps": "Pit Laps",
        "next_tyres": "Tyre Plan",
        "expected_finish_time": "Exp Finish Time",
        "finish_time_std": "Time Std",
        "expected_position": "Exp Position",
        "most_likely_position": "Likely Position",
        "strategy_score": "Strategy Score",
        "no_stop_penalty": "No Stop Penalty"
    })

    preferred_cols = [
        "Stops", "Pit Laps", "Tyre Plan",
        "Exp Finish Time", "Time Std",
        "Exp Position", "Likely Position",
        "Strategy Score", "No Stop Penalty"
    ]
    existing_cols = [c for c in preferred_cols if c in display_df.columns]
    return display_df[existing_cols]

# =============================
# 4. FastF1 로딩 함수
# =============================
def load_single_session_with_retry(year, gp, retries=3, delay=3):
    last_error = None
    for _ in range(retries):
        try:
            session = fastf1.get_session(year, gp, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=True)
            return session
        except Exception as e:
            last_error = e
            time.sleep(delay)
    raise last_error

def load_race_laps(seasons, grands_prix):
    all_laps = []
    errors = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_gps = len(seasons) * len(grands_prix)
    idx = 0

    for year in seasons:
        for gp in grands_prix:
            idx += 1
            status_text.text(f"📥 FastF1 데이터 다운로드 중: {year} {gp} ({idx}/{total_gps})")
            progress_bar.progress(idx / total_gps)

            try:
                session = load_single_session_with_retry(year, gp, retries=3, delay=2)
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

            except Exception as e:
                errors.append(f"{year} {gp}: {type(e).__name__} - {e}")

    progress_bar.empty()
    status_text.empty()

    if errors:
        st.warning("일부 GP 데이터 로드에 실패했습니다.")
        for err in errors[:10]:
            st.caption(err)

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
        slope, intercept = np.polyfit(x, y, 1) if len(np.unique(x)) > 1 else (0.05, y.mean())
        base_offset = y.mean() - global_mean

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
    raw_laps_df.to_csv(PREPROCESSED_DIR / "raw_laps.csv", index=False)
    clean_laps_df.to_csv(PREPROCESSED_DIR / "clean_laps.csv", index=False)

    with open(PREPROCESSED_DIR / "tyre_model.json", "w", encoding="utf-8") as f:
        json.dump(tyre_model, f, indent=4, ensure_ascii=False)

    with open(PREPROCESSED_DIR / "driver_pace_model.json", "w", encoding="utf-8") as f:
        json.dump(driver_pace_model, f, indent=4, ensure_ascii=False)

    with open(PREPROCESSED_DIR / "pit_stats.json", "w", encoding="utf-8") as f:
        json.dump(pit_stats, f, indent=4, ensure_ascii=False)

def load_preprocessed_data():
    required_files = [
        PREPROCESSED_DIR / "raw_laps.csv",
        PREPROCESSED_DIR / "clean_laps.csv",
        PREPROCESSED_DIR / "tyre_model.json",
        PREPROCESSED_DIR / "driver_pace_model.json",
        PREPROCESSED_DIR / "pit_stats.json"
    ]
    for f in required_files:
        if not f.exists():
            return None

    return (
        pd.read_csv(required_files[0]),
        pd.read_csv(required_files[1]),
        json.load(open(required_files[2], "r", encoding="utf-8")),
        json.load(open(required_files[3], "r", encoding="utf-8")),
        json.load(open(required_files[4], "r", encoding="utf-8"))
    )

@st.cache_data(show_spinner=False)
def prepare_or_load_data_cached():
    loaded = load_preprocessed_data()
    if loaded is not None:
        return loaded

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

# =============================
# 5. 전략 계산 함수
# =============================
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
    return {"SOFT": {1: 0.35, 2: 0.10}, "MEDIUM": {1: 0.65, 2: 0.22}, "HARD": {1: 0.95, 2: 0.38}}.get(compound, {}).get(laps_since_stop, 0.0)

def undercut_bonus(laps_since_stop, compound, front_gap, track_name):
    gain = {"SOFT": {1: 0.55, 2: 0.22}, "MEDIUM": {1: 0.38, 2: 0.15}, "HARD": {1: 0.18, 2: 0.08}}.get(compound, {}).get(laps_since_stop, 0.0)
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
                    candidates.append([{"pit_lap": pit1, "next_tyre": t1}, {"pit_lap": pit2, "next_tyre": t2}])

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

def predict_driver_lap_time(driver, base_lap, pace_offset, compound, tyre_life, tyre_model, front_gap, rear_gap, drs_available, laps_since_stop, rng, track_name, safety_mode="NONE"):
    info = tyre_model.get(compound, {"base_offset": 0.0, "deg_per_lap": 0.05, "driver_deg": {}})
    deg = info.get("driver_deg", {}).get(driver, info["deg_per_lap"])

    lap_time = base_lap + pace_offset + info["base_offset"] + (deg * safety_car_deg_factor(safety_mode)) * tyre_life
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

            lt = predict_driver_lap_time(
                car["driver"], base_lap, car["pace_offset"], car["compound"], car["tyre_life"],
                tyre_model, car["front_gap"], car.get("rear_gap", 2.0), car["front_gap"] <= 1.0,
                car["laps_since_stop"], rng, track_name, mode
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
    results = [simulate_race_once(total_laps, current_lap, base_lap, tyre_model, my_state, rivals, adjusted_pit_loss, rng, track_name, safety_mode) for _ in range(n)]
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
    (
        strategy, total_laps, current_lap, base_lap, tyre_model, current_position,
        current_compound, current_tyre_life, front_gap, rear_gap, driver_pace_model,
        recent_pace_lookup, my_driver, rivals, adjusted_pit_loss, track_name,
        safety_mode, n, seed, my_initial_race_time
    ) = args

    my_state = build_my_state(
        my_driver, current_position, current_compound, current_tyre_life, front_gap, rear_gap,
        driver_pace_model, recent_pace_lookup, base_lap, strategy, my_initial_race_time
    )
    sim = simulate_many(total_laps, current_lap, base_lap, tyre_model, my_state, rivals, adjusted_pit_loss, track_name, safety_mode, n, seed)
    return strategy_to_row(strategy, sim, current_tyre_life)

def run_batch_simulations(strategies, total_laps, current_lap, base_lap, tyre_model, current_position, current_compound, current_tyre_life, front_gap, rear_gap, driver_pace_model, recent_pace_lookup, my_driver, rivals, adjusted_pit_loss, track_name, safety_mode, n, seed_base, my_initial_race_time):
    jobs = [
        (
            strat, total_laps, current_lap, base_lap, tyre_model, current_position,
            current_compound, current_tyre_life, front_gap, rear_gap, driver_pace_model,
            recent_pace_lookup, my_driver, rivals, adjusted_pit_loss, track_name,
            safety_mode, n, seed_base + idx, my_initial_race_time
        )
        for idx, strat in enumerate(strategies)
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
        rival_names = [d for d in driver_pace_model.keys() if d != my_driver][:MAX_RIVALS]
        while len(rival_names) < MAX_RIVALS:
            rival_names.append(f"RIVAL{len(rival_names)+1}")
        rivals = build_fallback_rival_states(rival_names[:MAX_RIVALS], driver_pace_model, recent_pace_lookup, base_lap, np.random.default_rng(2026))
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

# =============================
# 6. 메인 앱
# =============================
def main():
    inject_custom_css()

    loaded = prepare_or_load_data_cached()
    if loaded is None:
        st.error("FastF1 데이터를 불러오지 못했습니다. 인터넷 연결, FastF1 버전, 또는 preprocessed 폴더를 확인하세요.")
        st.info("가능하면 로컬에서 한 번 실행해 preprocessed 파일들을 만든 뒤 배포하는 것을 추천합니다.")
        return

    raw_laps_df, clean_laps_df, tyre_model, driver_pace_model, pit_stats = loaded
    base_lap = clean_laps_df["LapTimeSeconds"].astype(float).mean()

    st.sidebar.header("Race Control Input")

    selected_driver_label = st.sidebar.selectbox(
        "Driver",
        list(DRIVER_OPTIONS.keys()),
        index=list(DRIVER_OPTIONS.keys()).index("Max Verstappen (Red Bull)") if "Max Verstappen (Red Bull)" in DRIVER_OPTIONS else 0
    )
    my_driver = DRIVER_OPTIONS[selected_driver_label]

    track_name_input = st.sidebar.selectbox(
        "Track",
        ["Bahrain", "Saudi Arabia", "Australia", "Japan", "Monaco"]
    )
    track_name = normalize_track_name(track_name_input)

    total_laps = st.sidebar.number_input("Total Laps", min_value=1, max_value=100, value=57)
    current_lap = st.sidebar.number_input("Current Lap", min_value=1, max_value=100, value=25)
    current_compound = st.sidebar.selectbox("Current Compound", ["SOFT", "MEDIUM", "HARD"], index=1)
    current_tyre_life_manual = st.sidebar.number_input("Current Tyre Life", min_value=0, max_value=60, value=12)

    current_position = st.sidebar.number_input("Current Position", min_value=1, max_value=20, value=3)
    front_gap = st.sidebar.number_input("Gap to Front (s)", min_value=0.0, max_value=60.0, value=1.2, step=0.1)
    rear_gap = st.sidebar.number_input("Gap to Rear (s)", min_value=0.0, max_value=60.0, value=2.5, step=0.1)
    safety_mode = st.sidebar.selectbox("Race Neutralization", ["NONE", "SC", "VSC"])

    use_auto_pit_loss = st.sidebar.radio("Pit Loss", ["Auto", "Manual"])
    if use_auto_pit_loss == "Auto":
        green_pit_loss = pit_stats["median_pit_loss"]
    else:
        green_pit_loss = st.sidebar.number_input("Green Flag Pit Loss (s)", min_value=10.0, max_value=50.0, value=22.0, step=0.5)

    start_calc = st.sidebar.button("Run Monte Carlo Strategy Simulation")

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("""
        <div class="hero-card">
            <div class="hero-badge">Race Strategy Control</div>
            <div class="hero-title">F1 Monte Carlo Strategy Simulator</div>
            <div class="hero-sub">
                FastF1 기반 레이스 데이터를 활용해 현재 레이스 상황에서 가장 유리한
                피트 스톱 전략을 추정합니다. 타이어 열화, 피트 손실, 트래픽, DRS,
                Safety Car 조건을 함께 반영합니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        logo_data = load_image_binary(LOGO_PATH)
        if logo_data:
            st.image(logo_data, width=110)

        track_img_path = TRACK_IMAGES_PATHS.get(track_name)
        if track_img_path and track_img_path.exists():
            render_card_start(f"{track_name} Circuit", "현재 선택된 서킷 레이아웃입니다.")
            st.image(str(track_img_path), use_container_width=True)
            render_card_end()
        else:
            render_card_start("Track Preview", "트랙 이미지가 없으면 이 영역은 비워둡니다.")
            st.info("assets/tracks 폴더에 트랙 이미지를 넣으면 자동 표시됩니다.")
            render_card_end()

    top1, top2 = st.columns([1.25, 0.95])

    with top1:
        render_card_start("Tyre Degradation Model", "학습 데이터로부터 계산한 컴파운드별 기본 성능 차이와 열화 추정치입니다.")
        tyre_table = [
            [tyre, info["base_offset"], info["deg_per_lap"], info["recommended_stint"]]
            for tyre, info in tyre_model.items()
        ]
        tyre_df = pd.DataFrame(
            tyre_table,
            columns=["Compound", "Base Offset (s)", "Deg / Lap", "Recommended Stint"]
        )
        st.dataframe(tyre_df, use_container_width=True, hide_index=True)
        render_card_end()

    with top2:
        render_card_start("Pit Lane Window", "현재 모델이 추정한 피트레인 손실 시간 기준입니다.")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Median Pit Loss", f"{pit_stats['median_pit_loss']} s")
        with m2:
            st.metric("Recommended Max", f"{pit_stats['recommended_max_pit_loss']} s")
        st.markdown("""
        <div class="summary-box">
            <ul>
                <li>Auto 모드에서는 데이터 기반 median pit loss를 사용합니다.</li>
                <li>SC / VSC 상황에서는 실제 손실 시간을 자동 보정합니다.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        render_card_end()

    if not start_calc:
        render_card_start("Simulation Guide", "왼쪽 입력 패널에서 현재 레이스 상황을 입력한 뒤 시뮬레이션을 실행하세요.")
        st.markdown("""
        <div class="mini-note">
            추천 입력 순서: Driver → Track → Current Lap → Compound → Tyre Life → Position → Gap → Safety Mode.
            입력이 끝나면 왼쪽 사이드바의 <b>Run Monte Carlo Strategy Simulation</b> 버튼을 누르면 됩니다.
        </div>
        """, unsafe_allow_html=True)
        render_card_end()
        return

    adjusted_pit_loss = adjust_pit_loss_for_track_status(green_pit_loss, safety_mode)
    current_tyre_life = estimate_current_tyre_life(
        current_compound,
        tyre_model,
        current_tyre_life_manual if current_tyre_life_manual > 0 else None
    )
    tyre_change_info = recommend_tyre_change_time(front_gap, rear_gap, safety_mode, current_position)

    with st.spinner("수백 개 조합에 대해 몬테카를로 시뮬레이션을 실행 중입니다..."):
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

    if result_df.empty:
        st.warning("전략 계산 결과가 없습니다. 현재 랩이 너무 후반이거나 입력 조건이 제한적일 수 있습니다.")
        return

    stop_count_info = recommend_stop_count(result_df)
    best = result_df.iloc[0]
    possible_stops = sorted(result_df["stops"].unique().tolist())
    display_result_df = format_strategy_table(result_df.head(10))

    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Recommended Stops", f"{stop_count_info['best_stop_count']} stop")
    with k2:
        st.metric("Expected Position", f"{best['expected_position']} P")
    with k3:
        st.metric("Likely Finish", f"{best['most_likely_position']} P")
    with k4:
        st.metric("Pit Loss Applied", f"{adjusted_pit_loss} s")

    res_left, res_right = st.columns([1.2, 0.9])

    with res_left:
        render_card_start("Top Strategy Ranking", "최종 후보 전략 중 상위 10개를 정렬한 결과입니다.")
        st.dataframe(display_result_df, use_container_width=True, hide_index=True)
        render_card_end()

        render_card_start("Stop Count Summary", "피트 횟수별 평균 성능 요약입니다.")
        st.dataframe(stop_count_info["summary_table"], use_container_width=True, hide_index=True)
        render_card_end()

    with res_right:
        render_card_start("Strategy Briefing", "현재 입력 조건 기준으로 가장 우세한 전략 해석입니다.")
        st.markdown(f"""
        <div class="summary-box">
            <ul>
                <li><b>Driver</b>: {selected_driver_label} ({my_driver})</li>
                <li><b>Track</b>: {track_name}</li>
                <li><b>Safety Mode</b>: {safety_mode}</li>
                <li><b>Current Position</b>: P{current_position}</li>
                <li><b>Current Tyre</b>: {current_compound} / {current_tyre_life} laps used</li>
                <li><b>Pit Loss Applied</b>: {adjusted_pit_loss} s</li>
                <li><b>Possible Stop Counts</b>: {possible_stops}</li>
                <li><b>Recommended Max Tyre Change</b>: {tyre_change_info['recommended_max_tyre_change_time']} s</li>
                <li><b>Tyre Change Comment</b>: {tyre_change_info['comment']}</li>
                <li><b>Model Comment</b>: {stop_count_info['comment']}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if best["stops"] == 0:
            st.info("후반전이 아니라면 무피트 전략은 참고용으로만 보고, 1회 피트 전략과 병행 판단하는 것이 안전합니다.")
        else:
            st.success(
                f"추천 전략은 {best['stops']}회 피트입니다. "
                f"주요 피트 랩은 {best['pit_laps']} 이고, "
                f"교체 타이어 계획은 {best['next_tyres']} 입니다."
            )
        render_card_end()

if __name__ == "__main__":
    main()
