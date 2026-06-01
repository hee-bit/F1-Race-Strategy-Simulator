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


# =========================
# 기본 경로 설정
# =========================
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
PREPROCESSED_DIR = BASE_DIR / "preprocessed"
ASSETS_DIR = BASE_DIR / "assets"
TRACKS_DIR = ASSETS_DIR / "tracks"
LOGO_DIR = ASSETS_DIR / "logos"

CACHE_DIR.mkdir(exist_ok=True)
PREPROCESSED_DIR.mkdir(exist_ok=True)

TRACK_IMAGES_PATHS = {
    "Bahrain": TRACKS_DIR / "bahrain.jpg",
    "Saudi Arabia": TRACKS_DIR / "saudi_arabia.jpg",
    "Australia": TRACKS_DIR / "australia.jpg",
    "Japan": TRACKS_DIR / "japan.jpg",
    "Monaco": TRACKS_DIR / "monaco.jpg",
}

F1_LOGO_PATH = LOGO_DIR / "f1_logo.jpg"


# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="F1 Race Strategy Simulator",
    page_icon="🏎️",
    layout="wide",
)


# =========================
# FastF1 캐시 활성화
# =========================
try:
    fastf1.Cache.enable_cache(str(CACHE_DIR))
except Exception:
    pass


# =========================
# CSS
# =========================
def inject_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f8fb 0%, #eef2f7 100%);
        }

        .main .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            max-width: 1400px;
        }

        h1, h2, h3, h4 {
            color: #111827;
            letter-spacing: -0.02em;
        }

        .hero-card {
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            color: white;
            border-radius: 24px;
            padding: 30px 32px;
            margin-bottom: 1.25rem;
            box-shadow: 0 14px 40px rgba(17, 24, 39, 0.18);
        }

        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .hero-subtitle {
            font-size: 1.03rem;
            color: rgba(255,255,255,0.82);
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.65);
            border-radius: 22px;
            padding: 22px 22px;
            box-shadow: 0 10px 30px rgba(31, 41, 55, 0.08);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
        }

        .card-title {
            font-size: 1.28rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.3rem;
        }

        .card-subtitle {
            font-size: 0.98rem;
            color: #6b7280;
            margin-bottom: 0.9rem;
        }

        .section-label {
            font-size: 1.1rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.6rem;
        }

        /* 버튼: 회색 -> 흰색 */
        div.stButton > button {
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d7dde8 !important;
            border-radius: 16px !important;
            font-size: 20px !important;
            font-weight: 800 !important;
            min-height: 62px !important;
            padding: 0.9rem 1.2rem !important;
            width: 100% !important;
            box-shadow: 0 6px 18px rgba(17, 24, 39, 0.06) !important;
        }

        div.stButton > button:hover {
            background: #f9fbff !important;
            border: 1px solid #b8c4d9 !important;
            transform: translateY(-1px);
        }

        div.stButton > button:focus {
            box-shadow: 0 0 0 0.2rem rgba(59, 130, 246, 0.18) !important;
        }

        /* 입력창 크게 */
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            min-height: 58px !important;
            border-radius: 15px !important;
            font-size: 18px !important;
            background: #ffffff !important;
            border: 1px solid #d6dbe6 !important;
        }

        label[data-testid="stWidgetLabel"] p {
            font-size: 1.02rem !important;
            font-weight: 700 !important;
            color: #1f2937 !important;
        }

        /* metric 카드 크게 */
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            padding: 22px 24px;
            box-shadow: 0 8px 22px rgba(17, 24, 39, 0.06);
        }

        [data-testid="stMetricLabel"] {
            font-size: 17px !important;
            font-weight: 700 !important;
            color: #6b7280 !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 2.25rem !important;
            font-weight: 800 !important;
            color: #111827 !important;
        }

        .track-img {
            width: 100%;
            border-radius: 18px;
            overflow: hidden;
            margin-top: 0.3rem;
            margin-bottom: 0.4rem;
            border: 1px solid #e5e7eb;
        }

        .result-box {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 8px 22px rgba(17, 24, 39, 0.05);
            margin-bottom: 0.9rem;
        }

        .result-title {
            font-size: 1.08rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.45rem;
        }

        .briefing-box {
            background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #dbe4f0;
            border-radius: 20px;
            padding: 20px 22px;
            color: #1f2937;
            font-size: 1.03rem;
            line-height: 1.7;
            box-shadow: 0 10px 28px rgba(17,24,39,0.05);
        }

        .small-note {
            color: #6b7280;
            font-size: 0.95rem;
        }

        .stDataFrame, .stTable {
            background: white;
            border-radius: 18px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# UI 헬퍼
# =========================
def render_card_start(title, subtitle=None):
    sub_html = f'<div class="card-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="card-title">{title}</div>
            {sub_html}
        """,
        unsafe_allow_html=True,
    )


def render_card_end():
    st.markdown("</div>", unsafe_allow_html=True)


def load_image_binary(path: Path):
    if path.exists():
        with open(path, "rb") as f:
            return f.read()
    return None


def format_strategy_table(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame()

    display_df = df.copy()

    if "pit_laps" in display_df.columns:
        display_df["pit_laps"] = display_df["pit_laps"].apply(
            lambda x: ", ".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)
        )

    if "next_tyres" in display_df.columns:
        display_df["next_tyres"] = display_df["next_tyres"].apply(
            lambda x: " → ".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)
        )

    rename_map = {
        "stops": "Stops",
        "pit_laps": "Pit Laps",
        "next_tyres": "Tyre Plan",
        "predicted_total_time": "Predicted Time (s)",
        "expected_rank": "Expected Rank",
    }
    display_df = display_df.rename(columns=rename_map)

    if "Predicted Time (s)" in display_df.columns:
        display_df["Predicted Time (s)"] = display_df["Predicted Time (s)"].round(2)

    if "Expected Rank" in display_df.columns:
        display_df["Expected Rank"] = display_df["Expected Rank"].round(2)

    return display_df


# =========================
# 데이터 로드 관련
# =========================
def load_preprocessed_data():
    raw_path = PREPROCESSED_DIR / "raw_laps.csv"
    clean_path = PREPROCESSED_DIR / "clean_laps.csv"
    tyre_model_path = PREPROCESSED_DIR / "tyre_model.json"
    driver_model_path = PREPROCESSED_DIR / "driver_pace_model.json"
    pit_stats_path = PREPROCESSED_DIR / "pit_stats.json"

    required = [raw_path, clean_path, tyre_model_path, driver_model_path, pit_stats_path]
    if not all(p.exists() for p in required):
        return None

    try:
        raw_laps_df = pd.read_csv(raw_path)
        clean_laps_df = pd.read_csv(clean_path)

        with open(tyre_model_path, "r", encoding="utf-8") as f:
            tyre_model = json.load(f)

        with open(driver_model_path, "r", encoding="utf-8") as f:
            driver_pace_model = json.load(f)

        with open(pit_stats_path, "r", encoding="utf-8") as f:
            pit_stats = json.load(f)

        return raw_laps_df, clean_laps_df, tyre_model, driver_pace_model, pit_stats

    except Exception as e:
        st.error(f"전처리 데이터 로드 실패: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_race_laps(seasons, grands_prix):
    all_laps = []
    failures = []

    for season, gp in product(seasons, grands_prix):
        try:
            session = fastf1.get_session(season, gp, "R")
            session.load()
            laps = session.laps.copy()

            laps["Season"] = season
            laps["GP"] = gp

            required_cols = [
                "Driver", "LapNumber", "LapTime", "Compound", "TyreLife",
                "PitInTime", "PitOutTime", "Stint", "TrackStatus"
            ]
            existing_cols = [c for c in required_cols if c in laps.columns]
            laps = laps[existing_cols + ["Season", "GP"]]

            all_laps.append(laps)

        except Exception as e:
            failures.append(f"{season} {gp}: {type(e).__name__} - {e}")

    if failures:
        st.warning(
            "일부 GP 데이터 로드에 실패했습니다.\n" + "\n".join(failures)
        )

    if not all_laps:
        return pd.DataFrame()

    raw = pd.concat(all_laps, ignore_index=True)
    return raw


def preprocess_laps(raw_laps_df):
    if raw_laps_df is None or raw_laps_df.empty:
        return pd.DataFrame()

    df = raw_laps_df.copy()

    df = df[df["LapTime"].notna()].copy()
    df["LapTimeSec"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()

    if "TyreLife" in df.columns:
        df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")
    else:
        df["TyreLife"] = np.nan

    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df = df.dropna(subset=["LapNumber", "LapTimeSec"])

    q1 = df["LapTimeSec"].quantile(0.01)
    q99 = df["LapTimeSec"].quantile(0.99)
    df = df[(df["LapTimeSec"] >= q1) & (df["LapTimeSec"] <= q99)].copy()

    return df


def build_tyre_deg_model(clean_laps_df):
    if clean_laps_df.empty:
        return {}

    model = {}
    grouped = clean_laps_df.dropna(subset=["Compound"]).groupby("Compound")

    for compound, group in grouped:
        if "TyreLife" in group.columns and group["TyreLife"].notna().sum() >= 5:
            x = group["TyreLife"].fillna(group["TyreLife"].median()).values
            y = group["LapTimeSec"].values
            if len(np.unique(x)) >= 2:
                coef = np.polyfit(x, y, 1)
                model[compound] = {
                    "base": float(coef[1]),
                    "deg_per_lap": float(max(coef[0], 0.0))
                }
            else:
                model[compound] = {
                    "base": float(group["LapTimeSec"].median()),
                    "deg_per_lap": 0.08
                }
        else:
            model[compound] = {
                "base": float(group["LapTimeSec"].median()),
                "deg_per_lap": 0.08
            }

    return model


def build_driver_pace_model(clean_laps_df):
    if clean_laps_df.empty:
        return {}

    model = (
        clean_laps_df.groupby("Driver")["LapTimeSec"]
        .median()
        .sort_values()
        .to_dict()
    )
    return {k: float(v) for k, v in model.items()}


def estimate_pit_stats(clean_laps_df):
    pit_loss = 21.5
    pit_work = 2.5

    try:
        df = clean_laps_df.copy()
        if "PitInTime" in df.columns and "PitOutTime" in df.columns:
            pit_in_count = df["PitInTime"].notna().sum()
            pit_out_count = df["PitOutTime"].notna().sum()

            if pit_in_count > 0 or pit_out_count > 0:
                pit_loss = 21.5
                pit_work = 2.5
    except Exception:
        pass

    return {
        "pit_lane_loss_median": float(pit_loss),
        "pit_work_time_median": float(pit_work)
    }


def save_preprocessed_data(raw_laps_df, clean_laps_df, tyre_model, driver_pace_model, pit_stats):
    raw_laps_df.to_csv(PREPROCESSED_DIR / "raw_laps.csv", index=False)
    clean_laps_df.to_csv(PREPROCESSED_DIR / "clean_laps.csv", index=False)

    with open(PREPROCESSED_DIR / "tyre_model.json", "w", encoding="utf-8") as f:
        json.dump(tyre_model, f, ensure_ascii=False, indent=2)

    with open(PREPROCESSED_DIR / "driver_pace_model.json", "w", encoding="utf-8") as f:
        json.dump(driver_pace_model, f, ensure_ascii=False, indent=2)

    with open(PREPROCESSED_DIR / "pit_stats.json", "w", encoding="utf-8") as f:
        json.dump(pit_stats, f, ensure_ascii=False, indent=2)


def prepare_or_load_data():
    loaded = load_preprocessed_data()
    if loaded is not None:
        return loaded

    st.info("preprocessed 데이터가 없어 FastF1에서 데이터를 불러옵니다. 처음 1회는 조금 오래 걸릴 수 있습니다.")

    seasons = [2023, 2024]
    grands_prix = ["Bahrain", "Saudi Arabia", "Australia", "Japan", "Monaco"]

    raw_laps_df = load_race_laps(seasons, grands_prix)
    if raw_laps_df.empty:
        st.error("FastF1 데이터를 불러오지 못했습니다. 인터넷 연결, FastF1 버전, 또는 preprocessed 폴더를 확인하세요.")
        return None

    clean_laps_df = preprocess_laps(raw_laps_df)
    tyre_model = build_tyre_deg_model(clean_laps_df)
    driver_pace_model = build_driver_pace_model(clean_laps_df)
    pit_stats = estimate_pit_stats(clean_laps_df)

    try:
        save_preprocessed_data(raw_laps_df, clean_laps_df, tyre_model, driver_pace_model, pit_stats)
    except Exception as e:
        st.warning(f"전처리 파일 저장에는 실패했지만 앱은 계속 실행됩니다: {e}")

    return raw_laps_df, clean_laps_df, tyre_model, driver_pace_model, pit_stats


# =========================
# 시뮬레이션 로직
# =========================
def recommend_tyre_change_time(safety_mode, pit_stats):
    base = pit_stats.get("pit_work_time_median", 2.5)

    if safety_mode == "SC":
        rec = min(base, 2.3)
        comment = "세이프티카 상황에서는 피트 손실이 줄어 비교적 공격적인 타이밍이 가능합니다."
    elif safety_mode == "VSC":
        rec = min(base + 0.1, 2.5)
        comment = "VSC 상황에서는 일반 상황보다 유리하지만 SC만큼 극적이지는 않습니다."
    else:
        rec = max(base, 2.5)
        comment = "일반 레이스 상황 기준 추천 피트 작업 시간입니다."

    return {
        "recommended_max_tyre_change_time": round(rec, 2),
        "comment": comment
    }


def adjust_pit_loss_for_track_status(base_pit_loss, safety_mode):
    if safety_mode == "SC":
        return round(base_pit_loss * 0.65, 2)
    if safety_mode == "VSC":
        return round(base_pit_loss * 0.8, 2)
    return round(base_pit_loss, 2)


def simulate_strategy(args):
    (
        total_laps,
        current_lap,
        current_compound,
        tyre_age,
        pit_loss,
        tyre_model,
        chosen_stops,
        pit_laps,
        next_tyres,
        driver_base
    ) = args

    remaining_laps = total_laps - current_lap + 1
    if remaining_laps <= 0:
        return None

    compounds = set(tyre_model.keys())
    if current_compound not in compounds:
        current_compound = next(iter(compounds)) if compounds else "MEDIUM"

    current_lap_time_model = tyre_model.get(current_compound, {"base": driver_base, "deg_per_lap": 0.08})

    total_time = 0.0
    current_stint_start = current_lap
    current_age = tyre_age
    active_compound = current_compound

    full_plan_laps = list(pit_laps) + [total_laps + 1]
    full_plan_tyres = list(next_tyres)

    for i, end_lap in enumerate(full_plan_laps):
        stint_end = min(end_lap - 1, total_laps)
        model = tyre_model.get(active_compound, {"base": driver_base, "deg_per_lap": 0.08})

        for lap in range(current_stint_start, stint_end + 1):
            age = current_age + (lap - current_stint_start)
            lap_time = model["base"] + model["deg_per_lap"] * age
            total_time += lap_time

        if end_lap <= total_laps:
            total_time += pit_loss
            current_stint_start = end_lap
            current_age = 0
            if i < len(full_plan_tyres):
                active_compound = full_plan_tyres[i]

    expected_rank = 1 + max(0, (total_time - (driver_base * remaining_laps)) / 18.0)

    return {
        "stops": chosen_stops,
        "pit_laps": list(pit_laps),
        "next_tyres": list(next_tyres),
        "predicted_total_time": round(total_time, 2),
        "expected_rank": round(expected_rank, 2),
    }


def generate_strategy_candidates(total_laps, current_lap, compounds, max_stops=2):
    remaining = total_laps - current_lap
    if remaining <= 3:
        return []

    candidates = []

    for stops in range(1, max_stops + 1):
        if stops == 1:
            possible_pit_laps = list(range(current_lap + 5, total_laps - 5))
            for pit1 in possible_pit_laps:
                for tyre1 in compounds:
                    candidates.append((stops, [pit1], [tyre1]))

        elif stops == 2:
            possible_pit1 = list(range(current_lap + 5, total_laps - 12))
            for pit1 in possible_pit1:
                possible_pit2 = list(range(pit1 + 8, total_laps - 4))
                for pit2 in possible_pit2:
                    for tyre1 in compounds:
                        for tyre2 in compounds:
                            candidates.append((stops, [pit1, pit2], [tyre1, tyre2]))

    return candidates


def run_strategy_simulation(total_laps, current_lap, current_compound, tyre_age, pit_loss, tyre_model, driver_base):
    compounds = list(tyre_model.keys()) if tyre_model else ["SOFT", "MEDIUM", "HARD"]

    candidates = generate_strategy_candidates(total_laps, current_lap, compounds, max_stops=2)
    if not candidates:
        return pd.DataFrame()

    tasks = [
        (
            total_laps,
            current_lap,
            current_compound,
            tyre_age,
            pit_loss,
            tyre_model,
            stops,
            pit_laps,
            next_tyres,
            driver_base
        )
        for stops, pit_laps, next_tyres in candidates
    ]

    try:
        workers = max(1, min(cpu_count() - 1, 4))
        with Pool(processes=workers) as pool:
            results = pool.map(simulate_strategy, tasks)
    except Exception:
        results = [simulate_strategy(t) for t in tasks]

    results = [r for r in results if r is not None]
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values(["expected_rank", "predicted_total_time"]).reset_index(drop=True)
    return df


# =========================
# 메인
# =========================
def main():
    inject_custom_css()

    if F1_LOGO_PATH.exists():
        try:
            st.logo(str(F1_LOGO_PATH))
        except Exception:
            pass

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">F1 Race Strategy Simulator</div>
            <div class="hero-subtitle">
                타이어 열화, 피트 손실, 현재 경기 상황을 반영해 최적 전략을 계산하는 스트림릿 대시보드
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    loaded = prepare_or_load_data()
    if loaded is None:
        st.stop()

    raw_laps_df, clean_laps_df, tyre_model, driver_pace_model, pit_stats = loaded

    left, right = st.columns([1.08, 1.02], gap="large")

    with left:
        render_card_start("레이스 입력", "현재 상황을 입력하면 전략 후보를 계산합니다.")

        gp_options = ["Bahrain", "Saudi Arabia", "Australia", "Japan", "Monaco"]
        selected_gp = st.selectbox("그랑프리", gp_options, index=0)

        track_img_path = TRACK_IMAGES_PATHS.get(selected_gp)
        if track_img_path and track_img_path.exists():
            st.image(str(track_img_path), use_container_width=True)

        driver_options = sorted(list(driver_pace_model.keys())) if driver_pace_model else ["VER", "LEC", "HAM", "NOR"]
        selected_driver = st.selectbox("드라이버", driver_options, index=0 if driver_options else None)

        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            total_laps = st.number_input("총 랩 수", min_value=20, max_value=80, value=57, step=1)
        with c2:
            current_lap = st.number_input("현재 랩", min_value=1, max_value=80, value=15, step=1)

        c3, c4 = st.columns([1, 1], gap="large")
        with c3:
            current_compound = st.selectbox(
                "현재 타이어 컴파운드",
                options=["SOFT", "MEDIUM", "HARD"],
                index=1
            )
        with c4:
            tyre_age = st.number_input("현재 타이어 사용 랩 수", min_value=0, max_value=50, value=8, step=1)

        # 네가 요청한 부분: 좌/우 균형 배치 + 크기 확대
        c5, c6 = st.columns([1, 1], gap="large")
        with c5:
            tyre_deg_override = st.number_input(
                "타이어 열화율 (초/랩)",
                min_value=0.00,
                max_value=1.50,
                value=0.08,
                step=0.01,
                help="현재 선택한 타이어의 랩당 성능 저하를 직접 조정합니다."
            )
        with c6:
            pit_lane_loss_override = st.number_input(
                "피트 레인 손실 추정치 (초)",
                min_value=5.0,
                max_value=40.0,
                value=float(pit_stats.get("pit_lane_loss_median", 21.5)),
                step=0.5,
                help="피트 인/아웃으로 잃는 전체 시간 추정치입니다."
            )

        safety_mode = st.selectbox(
            "세이프티카 상태",
            ["NONE", "VSC", "SC"],
            index=0
        )

        btn1, btn2 = st.columns([1, 1], gap="medium")
        with btn1:
            run_sim = st.button("시뮬레이션 실행", key="run_sim")
        with btn2:
            calc_best = st.button("최적 전략 계산", key="calc_best")

        render_card_end()

    with right:
        render_card_start("데이터 기반 추정", "전처리 파일 또는 FastF1 데이터를 바탕으로 계산된 기준값입니다.")

        adjusted_pit_loss = adjust_pit_loss_for_track_status(
            pit_lane_loss_override,
            safety_mode
        )

        tyre_change_info = recommend_tyre_change_time(safety_mode, pit_stats)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("피트 손실 추정", f"{adjusted_pit_loss:.2f}s")
        with m2:
            st.metric("추천 타이어 작업", f"{tyre_change_info['recommended_max_tyre_change_time']:.2f}s")

        m3, m4 = st.columns(2)
        with m3:
            st.metric("기본 피트 작업", f"{pit_stats.get('pit_work_time_median', 2.5):.2f}s")
        with m4:
            st.metric("드라이버 기준 페이스", f"{driver_pace_model.get(selected_driver, 90.0):.2f}s")

        st.markdown(
            f"""
            <div class="briefing-box">
                <b>브리핑</b><br>
                현재 세이프티카 상태는 <b>{safety_mode}</b>이며, 이에 따라 피트 손실은
                <b>{adjusted_pit_loss:.2f}초</b>로 계산됩니다.<br><br>
                현재 입력한 타이어 열화율은 <b>{tyre_deg_override:.2f}초/랩</b>이며,
                전략 시뮬레이션 시 선택 타이어 성능 모델에 반영됩니다.<br><br>
                {tyre_change_info["comment"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_card_end()

    if run_sim or calc_best:
        if current_lap >= total_laps:
            st.error("현재 랩은 총 랩 수보다 작아야 합니다.")
            st.stop()

        sim_tyre_model = dict(tyre_model) if tyre_model else {
            "SOFT": {"base": 89.8, "deg_per_lap": 0.12},
            "MEDIUM": {"base": 90.6, "deg_per_lap": 0.08},
            "HARD": {"base": 91.2, "deg_per_lap": 0.06},
        }

        if current_compound not in sim_tyre_model:
            sim_tyre_model[current_compound] = {"base": 90.5, "deg_per_lap": tyre_deg_override}
        else:
            sim_tyre_model[current_compound]["deg_per_lap"] = float(tyre_deg_override)

        driver_base = float(driver_pace_model.get(selected_driver, 90.0))

        with st.spinner("전략 시뮬레이션 계산 중..."):
            strategy_df = run_strategy_simulation(
                total_laps=total_laps,
                current_lap=current_lap,
                current_compound=current_compound,
                tyre_age=tyre_age,
                pit_loss=adjusted_pit_loss,
                tyre_model=sim_tyre_model,
                driver_base=driver_base
            )

        if strategy_df.empty:
            st.warning("생성된 전략 후보가 없습니다. 현재 랩이 너무 후반부일 수 있습니다.")
            st.stop()

        best = strategy_df.iloc[0].to_dict()
        display_df = format_strategy_table(strategy_df.head(12))

        st.markdown("## 전략 결과")

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("추천 피트 횟수", f"{int(best['stops'])}회")
        with k2:
            pit_laps_text = ", ".join(map(str, best["pit_laps"])) if isinstance(best["pit_laps"], list) else str(best["pit_laps"])
            st.metric("추천 피트 랩", pit_laps_text)
        with k3:
            st.metric("예상 평균 순위", f"{best['expected_rank']:.2f}")

        r1, r2 = st.columns([1.15, 1], gap="large")

        with r1:
            st.markdown(
                """
                <div class="result-box">
                    <div class="result-title">전략 후보 표</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        with r2:
            next_tyres_text = " → ".join(best["next_tyres"]) if isinstance(best["next_tyres"], list) else str(best["next_tyres"])

            st.markdown(
                f"""
                <div class="briefing-box">
                    <b>최적 전략 브리핑</b><br><br>
                    현재 조건에서 가장 유리한 전략은 <b>{int(best["stops"])} 스톱</b> 전략입니다.<br>
                    추천 피트 랩은 <b>{pit_laps_text}</b>이며,
                    이후 타이어 운용 계획은 <b>{next_tyres_text}</b>입니다.<br><br>
                    예측 총 주행 시간은 <b>{best["predicted_total_time"]:.2f}초</b>,
                    예상 평균 순위는 <b>{best["expected_rank"]:.2f}위</b>입니다.<br><br>
                    추천 피트 작업 시간은 <b>{tyre_change_info["recommended_max_tyre_change_time"]:.2f}초 이내</b>입니다.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption("참고: 이 결과는 데이터 기반 전략 비교용 시뮬레이션이며, 실제 경기의 교통·세이프티카 변동·언더컷 경쟁까지 완전히 반영한 것은 아닙니다.")


if __name__ == "__main__":
    main()
