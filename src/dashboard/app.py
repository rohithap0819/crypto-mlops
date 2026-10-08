import os
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Crypto MLOps Control Center",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIG
# ============================================================

API_BASE_URL = os.getenv(
    "CRYPTO_API_URL",
    "http://127.0.0.1:8000",
)

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
]

INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "1h": "1h",
    "4h": "4h",
}


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       STREAMLIT CHROME
       ======================================================== */

    header[data-testid="stHeader"] {
        display: none !important;
    }

    [data-testid="stToolbar"] {
        display: none !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* ========================================================
       APPLICATION BACKGROUND
       ======================================================== */

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 80% 0%,
                rgba(37, 99, 235, 0.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 15% 20%,
                rgba(14, 165, 233, 0.04),
                transparent 25%
            ),
            #070d18;
    }

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0 !important;
    }

    .block-container {
        max-width: 100% !important;
        padding-top: 0.65rem !important;
        padding-bottom: 3rem !important;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #050a13 0%,
                #07101d 100%
            );
        border-right: 1px solid rgba(148, 163, 184, 0.09);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    .sidebar-brand {
        padding: 5px 0 0 0;
    }

    .sidebar-title {
        color: #f8fafc;
        font-size: 23px;
        font-weight: 850;
        letter-spacing: -0.7px;
    }

    .sidebar-subtitle {
        color: #64748b;
        font-size: 13px;
        margin-top: 4px;
    }

    .sidebar-section {
        color: #64748b;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 800;
        margin-top: 37px;
        margin-bottom: 11px;
    }

    .sidebar-info-card {
        background:
            linear-gradient(
                145deg,
                rgba(17, 27, 45, 0.96),
                rgba(13, 22, 37, 0.96)
            );
        border: 1px solid rgba(148, 163, 184, 0.11);
        border-radius: 16px;
        padding: 15px 16px;
        margin-top: 15px;
        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.12);
    }

    .sidebar-info-title {
        color: #cbd5e1;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        font-weight: 800;
        margin-bottom: 7px;
    }

    .sidebar-info-text {
        color: #64748b;
        font-size: 12px;
        line-height: 1.75;
    }

    .sidebar-info-highlight {
        color: #94a3b8;
        font-size: 12px;
        line-height: 1.8;
    }

    /* ========================================================
       TOP HEADER
       ======================================================== */

    .top-title-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 24px;
        width: 100%;
        padding: 18px 6px 22px 6px;
        margin: 0 0 20px 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.10);
    }

    .title-left {
        min-width: 0;
    }

    .main-title {
        color: #f8fafc;
        font-size: 39px;
        font-weight: 850;
        line-height: 1.08;
        letter-spacing: -1.3px;
    }

    .main-subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 8px;
        line-height: 1.4;
    }

    .header-right {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-shrink: 0;
    }

    .live-badge {
        display: flex;
        align-items: center;
        gap: 9px;
        padding: 10px 15px;
        border-radius: 999px;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.23);
        color: #34d399;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.4px;
        white-space: nowrap;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow:
            0 0 0 4px rgba(34, 197, 94, 0.08),
            0 0 12px rgba(34, 197, 94, 0.70);
    }

    .model-badge {
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(96, 165, 250, 0.08);
        border: 1px solid rgba(96, 165, 250, 0.18);
        color: #93c5fd;
        font-size: 12px;
        font-weight: 700;
        white-space: nowrap;
    }

    /* ========================================================
       SECTION HEADERS
       ======================================================== */

    .section-row {
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 20px;
        margin: 28px 0 14px 0;
    }

    .section-title {
        color: #f8fafc;
        font-size: 19px;
        line-height: 1.2;
        font-weight: 850;
        letter-spacing: -0.4px;
    }

    .section-helper {
        color: #64748b;
        font-size: 12px;
        text-align: right;
    }

    /* ========================================================
       MARKET CARDS
       ======================================================== */

    .market-card-link {
        display: block;
        text-decoration: none !important;
        color: inherit !important;
    }

    .market-card {
        position: relative;
        overflow: hidden;
        min-height: 214px;
        padding: 20px 19px 18px 19px;
        border-radius: 19px;
        border: 1px solid rgba(148, 163, 184, 0.11);
        background:
            linear-gradient(
                145deg,
                rgba(17, 27, 45, 0.98),
                rgba(10, 19, 33, 0.98)
            );
        box-shadow:
            0 12px 32px rgba(0, 0, 0, 0.13);
        transition:
            transform 0.16s ease,
            border-color 0.16s ease,
            box-shadow 0.16s ease;
    }

    .market-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background:
            linear-gradient(
                90deg,
                #38bdf8,
                #60a5fa,
                transparent
            );
        opacity: 0.85;
    }

    .market-card:hover {
        transform: translateY(-3px);
        border-color: rgba(96, 165, 250, 0.30);
        box-shadow:
            0 18px 38px rgba(0, 0, 0, 0.24);
    }

    .market-card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
    }

    .market-symbol {
        color: #f8fafc;
        font-size: 16px;
        font-weight: 850;
    }

    .hold-pill,
    .up-pill,
    .down-pill {
        border-radius: 8px;
        padding: 6px 9px;
        font-size: 10px;
        font-weight: 850;
        letter-spacing: 0.3px;
        white-space: nowrap;
    }

    .hold-pill {
        color: #cbd5e1;
        background: #172235;
        border: 1px solid rgba(148, 163, 184, 0.17);
    }

    .up-pill {
        color: #34d399;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.18);
    }

    .down-pill {
        color: #fb7185;
        background: rgba(244, 63, 94, 0.08);
        border: 1px solid rgba(244, 63, 94, 0.18);
    }

    .market-price {
        color: #f8fafc;
        font-size: 24px;
        font-weight: 850;
        margin-top: 28px;
        letter-spacing: -0.7px;
    }

    .market-meta {
        color: #64748b;
        font-size: 12px;
        margin-top: 18px;
        line-height: 1.8;
    }

    .market-meta strong {
        color: #cbd5e1;
        font-weight: 750;
    }

    .confidence-track {
        width: 100%;
        height: 4px;
        margin-top: 13px;
        overflow: hidden;
        border-radius: 999px;
        background: #182333;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 999px;
        background:
            linear-gradient(
                90deg,
                #38bdf8,
                #60a5fa
            );
    }

    /* ========================================================
       SYSTEM HEALTH STRIP
       ======================================================== */

    .system-strip {
        display: grid;
        grid-template-columns:
            1fr 1fr 1fr 1fr 1fr;
        gap: 12px;
        margin-top: 16px;
    }

    .system-item {
        background:
            linear-gradient(
                145deg,
                #101a2b,
                #0d1625
            );
        border: 1px solid rgba(148, 163, 184, 0.10);
        border-radius: 14px;
        padding: 13px 15px;
    }

    .system-label {
        color: #64748b;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        font-weight: 800;
    }

    .system-value {
        color: #f8fafc;
        font-size: 15px;
        font-weight: 800;
        margin-top: 5px;
    }

    .system-pass {
        color: #34d399;
    }

    /* ========================================================
       HERO
       ======================================================== */

    .hero-card {
        position: relative;
        overflow: hidden;
        padding: 25px;
        border-radius: 21px;
        border: 1px solid rgba(148, 163, 184, 0.11);
        background:
            radial-gradient(
                circle at 100% 0%,
                rgba(59, 130, 246, 0.11),
                transparent 30%
            ),
            linear-gradient(
                145deg,
                #111b2d,
                #0b1422
            );
        box-shadow:
            0 16px 40px rgba(0, 0, 0, 0.17);
    }

    .hero-symbol {
        color: #64748b;
        font-size: 12px;
        font-weight: 850;
        letter-spacing: 1.8px;
    }

    .hero-price {
        color: #f8fafc;
        font-size: 44px;
        line-height: 1;
        font-weight: 900;
        margin-top: 10px;
        letter-spacing: -1.5px;
    }

    .hero-signal {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 10px;
    }

    /* ========================================================
       KPI CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        min-height: 98px;
        background:
            linear-gradient(
                145deg,
                #101a2b,
                #0d1625
            );
        border: 1px solid rgba(148, 163, 184, 0.11);
        border-radius: 15px;
        padding: 14px 16px;
        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.10);
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricLabel"] p {
        color: #cbd5e1 !important;
        font-weight: 650 !important;
    }

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] *,
    div[data-testid="stMetricValue"] div {
        color: #f8fafc !important;
        font-weight: 850 !important;
    }

    div[data-testid="stMetricDelta"] * {
        color: #94a3b8 !important;
    }

    /* ========================================================
       STREAMLIT BUTTONS
       ======================================================== */

    .stButton > button {
        width: 100%;
        min-height: 41px;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        background: #f8fafc;
        color: #172033;
        font-weight: 700;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        background: #ffffff;
        color: #0f172a;
        border-color: rgba(96, 165, 250, 0.40);
    }

    /* ========================================================
       SELECTBOX / RADIO
       ======================================================== */

    div[data-baseweb="select"] > div {
        border-radius: 10px !important;
    }

    div[role="radiogroup"] {
        gap: 7px;
    }

    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.10);
    }

    /* ========================================================
       RESPONSIVE
       ======================================================== */

    @media (max-width: 1100px) {

        .main-title {
            font-size: 32px;
        }

        .system-strip {
            grid-template-columns:
                repeat(2, 1fr);
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# API HELPERS
# ============================================================

def api_get(
    endpoint,
    params=None,
    timeout=8,
):
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            params=params,
            timeout=timeout,
        )

        response.raise_for_status()

        return response.json()

    except Exception:
        return None


def records_from_response(data):
    if data is None:
        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in [
            "data",
            "records",
            "items",
            "results",
        ]:

            value = data.get(key)

            if isinstance(value, list):
                return value

    return []


def to_dataframe(data):
    records = records_from_response(data)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def pick(
    row,
    *names,
    default=None,
):
    if row is None:
        return default

    for name in names:

        if (
            name in row
            and pd.notna(row[name])
        ):
            return row[name]

    return default


def latest_row_for_symbol(
    df,
    symbol,
):
    if (
        df.empty
        or "symbol" not in df.columns
    ):
        return None

    rows = df[
        df["symbol"]
        .astype(str)
        .str.upper()
        == symbol.upper()
    ].copy()

    if rows.empty:
        return None

    if "open_time_ms" in rows.columns:

        rows = rows.sort_values(
            "open_time_ms"
        )

    elif (
        "candle_open_time_ms"
        in rows.columns
    ):

        rows = rows.sort_values(
            "candle_open_time_ms"
        )

    elif "open_time" in rows.columns:

        rows = rows.sort_values(
            "open_time"
        )

    return rows.iloc[-1].to_dict()


def format_price(value):
    if value is None or pd.isna(value):
        return "—"

    value = float(value)

    if value >= 1000:
        return f"${value:,.2f}"

    if value >= 1:
        return f"${value:,.2f}"

    if value >= 0.1:
        return f"${value:,.4f}"

    return f"${value:,.6f}"


def format_percent(value):
    if value is None or pd.isna(value):
        return "—"

    return f"{float(value) * 100:.2f}%"


def normalize_datetime(series):
    if series is None:
        return pd.Series(
            dtype="datetime64[ns, UTC]"
        )

    result = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )

    if result.isna().all():

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        result = pd.to_datetime(
            numeric,
            unit="ms",
            errors="coerce",
            utc=True,
        )

    return result


def get_prediction_signal(row):
    value = pick(
        row or {},
        "signal",
        "ensemble_signal",
        "raw_prediction",
        default="HOLD",
    )

    return str(value).upper()


def get_probability_up(row):
    return pick(
        row or {},
        "ensemble_probability_up",
        "probability_up",
        default=None,
    )


def get_confidence(row):
    return pick(
        row or {},
        "ensemble_confidence",
        "confidence",
        default=None,
    )


def signal_class(signal):
    signal = str(signal).upper()

    if signal == "UP":
        return "up-pill"

    if signal == "DOWN":
        return "down-pill"

    return "hold-pill"


def normalize_history_timestamps(df):
    if df.empty:
        return df

    result = df.copy()

    timestamp_column = None

    for candidate in [
        "candle_open_time_ms",
        "open_time_ms",
        "open_time",
        "timestamp",
        "time",
    ]:

        if candidate in result.columns:
            timestamp_column = candidate
            break

    if timestamp_column is None:
        return result

    result["timestamp"] = normalize_datetime(
        result[timestamp_column]
    )

    return result.dropna(
        subset=["timestamp"]
    )


def fetch_prediction_history(
    symbol,
    limit=200,
):
    data = api_get(
        "/predictions/history",
        params={
            "symbol": symbol,
            "limit": limit,
        },
        timeout=10,
    )

    return normalize_history_timestamps(
        to_dataframe(data)
    )


# ============================================================
# LOAD CURRENT DATA
# ============================================================

market_latest_data = api_get(
    "/market/latest"
)

prediction_latest_data = api_get(
    "/predictions/latest"
)

monitoring_latest_data = api_get(
    "/monitoring/latest"
)

market_latest_df = to_dataframe(
    market_latest_data
)

prediction_latest_df = to_dataframe(
    prediction_latest_data
)

# /monitoring/latest returns ONE object.
if isinstance(
    monitoring_latest_data,
    dict,
):
    monitoring_latest_row = (
        monitoring_latest_data
    )
else:
    monitoring_latest_row = {}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="sidebar-brand">

            <div class="sidebar-title">
                ₿ Crypto MLOps
            </div>

            <div class="sidebar-subtitle">
                Live prediction & monitoring
            </div>

        </div>
        """
    )

    st.html(
        """
        <div class="sidebar-section">
            Market
        </div>
        """
    )

    selected_from_sidebar = st.selectbox(
        "Market",
        ["All", *SYMBOLS],
        index=0,
        label_visibility="collapsed",
    )

    st.html(
        """
        <div class="sidebar-section">
            Dashboard
        </div>
        """
    )

    if st.button(
        "↻ Refresh data",
        use_container_width=True,
    ):
        st.rerun()

    st.html(
        """
        <div class="sidebar-info-card">

            <div class="sidebar-info-title">
                Data flow
            </div>

            <div class="sidebar-info-text">
                Binance → SQLite → Resident
                Inference → FastAPI → Streamlit
            </div>

        </div>

        <div class="sidebar-info-card">

            <div class="sidebar-info-title">
                Production model
            </div>

            <div class="sidebar-info-highlight">
                CatBoost + GRU<br>
                5-minute direction<br>
                Confidence threshold: 0.55
            </div>

        </div>

        <div class="sidebar-info-card">

            <div class="sidebar-info-title">
                Inference architecture
            </div>

            <div class="sidebar-info-highlight">
                Resident model service<br>
                55% CatBoost / 45% GRU<br>
                Continuous 1-minute pipeline
            </div>

        </div>
        """
    )


# ============================================================
# SELECTED SYMBOL
# ============================================================

query_symbol = None

try:
    query_symbol = st.query_params.get(
        "symbol"
    )
except Exception:
    query_symbol = None

if isinstance(
    query_symbol,
    list,
):

    query_symbol = (
        query_symbol[0]
        if query_symbol
        else None
    )

if query_symbol not in SYMBOLS:
    query_symbol = None

if query_symbol:

    selected_symbol = query_symbol

elif selected_from_sidebar != "All":

    selected_symbol = selected_from_sidebar

else:

    selected_symbol = None


# ============================================================
# TOP HEADER
# ============================================================

model_version = pick(
    monitoring_latest_row,
    "model_version",
    default="catboost_v1_gru_v1_ensemble_v1",
)

metric_time = pick(
    monitoring_latest_row,
    "metric_time",
    default=None,
)

if metric_time:
    last_update_text = str(
        metric_time
    ).replace(
        "T",
        " "
    ).replace(
        "+00:00",
        " UTC"
    )
else:
    last_update_text = "Waiting for telemetry"


st.html(
    f"""
    <div class="top-title-bar">

        <div class="title-left">

            <div class="main-title">
                ₿ Crypto MLOps Control Center
            </div>

            <div class="main-subtitle">
                Live market intelligence, model inference
                and production health monitoring
            </div>

        </div>

        <div class="header-right">

            <div class="live-badge">
                <span class="live-dot"></span>
                LIVE PIPELINE
            </div>

            <div class="model-badge">
                {model_version}
            </div>

        </div>

    </div>

    <div
        style="
            color:#475569;
            font-size:11px;
            margin:-9px 4px 2px 4px;
        "
    >
        Last telemetry update: {last_update_text}
    </div>
    """
)


# ################################################################
# ################################################################
# ALL MARKET VIEW
# ################################################################

if selected_symbol is None:

    # ============================================================
    # MARKET OVERVIEW
    # ============================================================

    st.html(
        """
        <div class="section-row">

            <div class="section-title">
                Market overview
            </div>

            <div class="section-helper">
                Click an asset to open its detailed view
            </div>

        </div>
        """
    )

    card_columns = st.columns(5)

    for index, symbol in enumerate(SYMBOLS):

        market_row = latest_row_for_symbol(
            market_latest_df,
            symbol,
        )

        prediction_row = (
            latest_row_for_symbol(
                prediction_latest_df,
                symbol,
            )
        )

        price = pick(
            market_row or {},
            "close_price",
            "close",
            default=None,
        )

        signal = get_prediction_signal(
            prediction_row
        )

        confidence = get_confidence(
            prediction_row
        )

        probability_up = get_probability_up(
            prediction_row
        )

        confidence_value = (
            float(confidence)
            if confidence is not None
            else 0.0
        )

        signal_css = signal_class(
            signal
        )

        with card_columns[index]:

            st.html(
                f"""
                <a
                    class="market-card-link"
                    href="?symbol={symbol}"
                    target="_self"
                >

                    <div class="market-card">

                        <div class="market-card-top">

                            <div class="market-symbol">
                                {symbol}
                            </div>

                            <div class="{signal_css}">
                                {signal}
                            </div>

                        </div>

                        <div class="market-price">
                            {format_price(price)}
                        </div>

                        <div class="market-meta">

                            Confidence:
                            <strong>
                                {format_percent(confidence)}
                            </strong>

                            <br>

                            UP probability:
                            <strong>
                                {format_percent(probability_up)}
                            </strong>

                        </div>

                        <div class="confidence-track">

                            <div
                                class="confidence-fill"
                                style="
                                    width:
                                    {min(
                                        max(
                                            confidence_value,
                                            0
                                        ),
                                        1
                                    ) * 100:.1f}%;
                                "
                            ></div>

                        </div>

                    </div>

                </a>
                """
            )

    # ============================================================
    # SYSTEM STATUS STRIP
    # ============================================================

    data_status = str(
        pick(
            monitoring_latest_row,
            "data_freshness_status",
            default="UNKNOWN",
        )
    )

    sequence_status = str(
        pick(
            monitoring_latest_row,
            "sequence_continuity_status",
            default="UNKNOWN",
        )
    )

    prediction_status = str(
        pick(
            monitoring_latest_row,
            "prediction_completeness_status",
            default="UNKNOWN",
        )
    )

    latency = pick(
        monitoring_latest_row,
        "mean_inference_latency_ms",
        default=None,
    )

    drift = pick(
        monitoring_latest_row,
        "feature_abs_z_gt3_pct",
        default=None,
    )

    st.html(
        f"""
        <div class="system-strip">

            <div class="system-item">

                <div class="system-label">
                    Data freshness
                </div>

                <div class="
                    system-value
                    {'system-pass' if data_status == 'PASS' else ''}
                ">
                    ● {data_status}
                </div>

            </div>

            <div class="system-item">

                <div class="system-label">
                    Sequence continuity
                </div>

                <div class="
                    system-value
                    {'system-pass' if sequence_status == 'PASS' else ''}
                ">
                    ● {sequence_status}
                </div>

            </div>

            <div class="system-item">

                <div class="system-label">
                    Prediction completeness
                </div>

                <div class="
                    system-value
                    {'system-pass' if prediction_status == 'PASS' else ''}
                ">
                    ● {prediction_status}
                </div>

            </div>

            <div class="system-item">

                <div class="system-label">
                    Mean inference
                </div>

                <div class="system-value">
                    {
                        f"{float(latency):.1f} ms"
                        if latency is not None
                        else "—"
                    }
                </div>

            </div>

            <div class="system-item">

                <div class="system-label">
                    Feature drift
                </div>

                <div class="system-value">
                    {
                        f"{float(drift):.2f}%"
                        if drift is not None
                        else "—"
                    }
                    |z| > 3
                </div>

            </div>

        </div>
        """
    )

    # ============================================================
    # ENSEMBLE BAR CHART
    # ============================================================

    st.html(
        """
        <div class="section-row">

            <div class="section-title">
                Ensemble market view
            </div>

            <div class="section-helper">
                Current probability of a positive 5-minute direction
            </div>

        </div>
        """
    )

    chart_rows = []

    for symbol in SYMBOLS:

        prediction_row = (
            latest_row_for_symbol(
                prediction_latest_df,
                symbol,
            )
        )

        probability_up = (
            get_probability_up(
                prediction_row
            )
        )

        if probability_up is not None:

            chart_rows.append(
                {
                    "symbol": symbol,
                    "UP probability": (
                        float(
                            probability_up
                        ) * 100
                    ),
                }
            )

    chart_df = pd.DataFrame(
        chart_rows
    )

    if not chart_df.empty:

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=chart_df["symbol"],
                y=chart_df[
                    "UP probability"
                ],
                text=[
                    f"{value:.1f}%"
                    for value
                    in chart_df[
                        "UP probability"
                    ]
                ],
                textposition="outside",
                textfont=dict(
                    color="#f8fafc",
                    size=12,
                ),
                marker=dict(
                    color="#3b82f6",
                    line=dict(
                        width=0,
                    ),
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "UP probability: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_hline(
            y=50,
            line_dash="dot",
            line_color="rgba(148,163,184,0.45)",
            annotation_text="50%",
            annotation_position="right",
            annotation_font=dict(
                color="#cbd5e1",
                size=11,
            ),
        )

        fig.update_layout(
            height=330,
            margin=dict(
                l=0,
                r=0,
                t=25,
                b=5,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#f8fafc"
            ),
            xaxis=dict(
                tickfont=dict(
                    color="#cbd5e1",
                    size=12,
                ),
                gridcolor="rgba(0,0,0,0)",
                zeroline=False,
            ),
            yaxis=dict(
                title="UP probability (%)",
                range=[0, 100],
                tickfont=dict(
                    color="#cbd5e1",
                    size=12,
                ),
                title_font=dict(
                    color="#f8fafc",
                    size=12,
                ),
                gridcolor="rgba(148,163,184,0.08)",
                zeroline=False,
            ),
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

    # ============================================================
    # LOAD ALL PREDICTION HISTORY ONCE
    # ============================================================

    all_history = {}

    for symbol in SYMBOLS:

        history_df = fetch_prediction_history(
            symbol,
            limit=200,
        )

        if not history_df.empty:
            all_history[symbol] = (
                history_df
            )

    # ============================================================
    # UP PROBABILITY LINE CHART
    # ============================================================

    st.html(
        """
        <div class="section-row">

            <div class="section-title">
                UP probability over time
            </div>

            <div class="section-helper">
                Historical ensemble probability across all assets
            </div>

        </div>
        """
    )

    probability_fig = go.Figure()

    probability_has_data = False

    for symbol in SYMBOLS:

        history_df = all_history.get(
            symbol,
            pd.DataFrame(),
        )

        if (
            history_df.empty
            or "ensemble_probability_up"
            not in history_df.columns
        ):
            continue

        probability_values = (
            pd.to_numeric(
                history_df[
                    "ensemble_probability_up"
                ],
                errors="coerce",
            ) * 100
        )

        valid_mask = (
            history_df[
                "timestamp"
            ].notna()
            & probability_values.notna()
        )

        plot_df = history_df.loc[
            valid_mask
        ].copy()

        plot_df["value"] = (
            probability_values[
                valid_mask
            ]
        )

        if plot_df.empty:
            continue

        probability_has_data = True

        probability_fig.add_trace(
            go.Scatter(
                x=plot_df["timestamp"],
                y=plot_df["value"],
                mode="lines",
                name=symbol,
                line=dict(
                    width=2,
                ),
                hovertemplate=(
                    f"<b>{symbol}</b><br>"
                    "UP probability: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

    if probability_has_data:

        probability_fig.add_hline(
            y=50,
            line_dash="dot",
            line_color="rgba(148,163,184,0.40)",
            annotation_text="50% neutral",
            annotation_position="right",
            annotation_font=dict(
                color="#cbd5e1",
                size=11,
            ),
        )

        probability_fig.update_layout(
            height=390,
            margin=dict(
                l=0,
                r=0,
                t=30,
                b=5,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#f8fafc"
            ),
            hovermode="x unified",
            xaxis=dict(
                tickfont=dict(
                    color="#cbd5e1",
                    size=11,
                ),
                gridcolor="rgba(148,163,184,0.06)",
                zeroline=False,
            ),
            yaxis=dict(
                title="UP probability (%)",
                range=[0, 100],
                tickfont=dict(
                    color="#cbd5e1",
                    size=11,
                ),
                title_font=dict(
                    color="#f8fafc",
                    size=12,
                ),
                gridcolor="rgba(148,163,184,0.08)",
                zeroline=False,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(
                    color="#f8fafc",
                    size=12,
                ),
            ),
        )

        st.plotly_chart(
            probability_fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

    else:

        st.info(
            "Prediction history is not available yet."
        )

    # ============================================================
    # CONFIDENCE LINE CHART
    # ============================================================

    st.html(
        """
        <div class="section-row">

            <div class="section-title">
                Prediction confidence over time
            </div>

            <div class="section-helper">
                Model confidence versus the 55% action threshold
            </div>

        </div>
        """
    )

    confidence_fig = go.Figure()

    confidence_has_data = False

    for symbol in SYMBOLS:

        history_df = all_history.get(
            symbol,
            pd.DataFrame(),
        )

        if (
            history_df.empty
            or "ensemble_confidence"
            not in history_df.columns
        ):
            continue

        confidence_values = (
            pd.to_numeric(
                history_df[
                    "ensemble_confidence"
                ],
                errors="coerce",
            ) * 100
        )

        valid_mask = (
            history_df[
                "timestamp"
            ].notna()
            & confidence_values.notna()
        )

        plot_df = history_df.loc[
            valid_mask
        ].copy()

        plot_df["value"] = (
            confidence_values[
                valid_mask
            ]
        )

        if plot_df.empty:
            continue

        confidence_has_data = True

        confidence_fig.add_trace(
            go.Scatter(
                x=plot_df["timestamp"],
                y=plot_df["value"],
                mode="lines",
                name=symbol,
                line=dict(
                    width=2,
                ),
                hovertemplate=(
                    f"<b>{symbol}</b><br>"
                    "Confidence: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

    if confidence_has_data:

        confidence_fig.add_hline(
            y=55,
            line_dash="dot",
            line_color="rgba(245,158,11,0.65)",
            annotation_text="55% action threshold",
            annotation_position="right",
            annotation_font=dict(
                color="#fbbf24",
                size=11,
            ),
        )

        confidence_fig.update_layout(
            height=390,
            margin=dict(
                l=0,
                r=0,
                t=30,
                b=5,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#f8fafc"
            ),
            hovermode="x unified",
            xaxis=dict(
                tickfont=dict(
                    color="#cbd5e1",
                    size=11,
                ),
                gridcolor="rgba(148,163,184,0.06)",
                zeroline=False,
            ),
            yaxis=dict(
                title="Confidence (%)",
                range=[45, 65],
                tickfont=dict(
                    color="#cbd5e1",
                    size=11,
                ),
                title_font=dict(
                    color="#f8fafc",
                    size=12,
                ),
                gridcolor="rgba(148,163,184,0.08)",
                zeroline=False,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(
                    color="#f8fafc",
                    size=12,
                ),
            ),
        )

        st.plotly_chart(
            confidence_fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

    else:

        st.info(
            "Confidence history is not available yet."
        )

    # ============================================================
    # FINAL MLOPS HEALTH
    # ============================================================

    st.html(
        """
        <div class="section-row">

            <div class="section-title">
                MLOps health
            </div>

            <div class="section-helper">
                Production pipeline telemetry
            </div>

        </div>
        """
    )

    health_cols = st.columns(5)

    with health_cols[0]:

        st.metric(
            "Data freshness",
            str(
                pick(
                    monitoring_latest_row,
                    "data_freshness_status",
                    default="UNKNOWN",
                )
            ),
        )

    with health_cols[1]:

        st.metric(
            "Sequence continuity",
            str(
                pick(
                    monitoring_latest_row,
                    "sequence_continuity_status",
                    default="UNKNOWN",
                )
            ),
        )

    with health_cols[2]:

        st.metric(
            "Prediction completeness",
            str(
                pick(
                    monitoring_latest_row,
                    "prediction_completeness_status",
                    default="UNKNOWN",
                )
            ),
        )

    with health_cols[3]:

        latency = pick(
            monitoring_latest_row,
            "mean_inference_latency_ms",
            default=None,
        )

        st.metric(
            "Mean inference",
            (
                f"{float(latency):.1f} ms"
                if latency is not None
                else "—"
            ),
        )

    with health_cols[4]:

        drift = pick(
            monitoring_latest_row,
            "feature_abs_z_gt3_pct",
            default=None,
        )

        st.metric(
            "Features |z| > 3",
            (
                f"{float(drift):.2f}%"
                if drift is not None
                else "—"
            ),
        )


# ################################################################
# ################################################################
# SELECTED COIN VIEW
# ################################################################

else:

    # ============================================================
    # BACK BUTTON
    # ============================================================

    back_col, _ = st.columns(
        [1, 8]
    )

    with back_col:

        if st.button(
            "← All markets"
        ):

            try:
                st.query_params.clear()
            except Exception:
                pass

            st.rerun()

    # ============================================================
    # CONTROLS
    # ============================================================

    control_col_1, control_col_2 = (
        st.columns(2)
    )

    with control_col_1:

        interval_label = st.radio(
            "Timeframe",
            list(INTERVALS.keys()),
            index=0,
            horizontal=True,
        )

    with control_col_2:

        history_limit = st.radio(
            "History",
            [100, 500, 1000, 5000],
            index=1,
            horizontal=True,
        )

    # ============================================================
    # LOAD SELECTED COIN DATA
    # ============================================================

    market_history_data = api_get(
        "/market/history",
        params={
            "symbol": selected_symbol,
            "interval": INTERVALS[
                interval_label
            ],
            "limit": history_limit,
        },
        timeout=10,
    )

    prediction_history_data = api_get(
        "/predictions/history",
        params={
            "symbol": selected_symbol,
            "limit": min(
                history_limit,
                1000,
            ),
        },
        timeout=10,
    )

    market_history_df = to_dataframe(
        market_history_data
    )

    prediction_history_df = (
        normalize_history_timestamps(
            to_dataframe(
                prediction_history_data
            )
        )
    )

    selected_market = latest_row_for_symbol(
        market_latest_df,
        selected_symbol,
    )

    selected_prediction = (
        latest_row_for_symbol(
            prediction_latest_df,
            selected_symbol,
        )
    )

    current_price = pick(
        selected_market or {},
        "close_price",
        "close",
        default=None,
    )

    signal = get_prediction_signal(
        selected_prediction
    )

    probability_up = get_probability_up(
        selected_prediction
    )

    confidence = get_confidence(
        selected_prediction
    )

    selected_model_version = pick(
        selected_prediction or {},
        "model_version",
        default=model_version,
    )

    inference_latency = pick(
        selected_prediction or {},
        "inference_latency_ms",
        default=None,
    )

    # ============================================================
    # HERO
    # ============================================================

    signal_css = signal_class(
        signal
    )

    st.html(
        f"""
        <div class="hero-card">

            <div class="hero-symbol">
                {selected_symbol}
            </div>

            <div class="hero-price">
                {format_price(current_price)}
            </div>

            <div class="hero-signal">
                Current ensemble signal:

                <span class="{signal_css}">
                    {signal}
                </span>
            </div>

        </div>
        """
    )

    # ============================================================
    # KPIs
    # ============================================================

    k1, k2, k3, k4, k5 = (
        st.columns(5)
    )

    with k1:

        st.metric(
            "Current price",
            format_price(
                current_price
            ),
        )

    with k2:

        st.metric(
            "UP probability",
            format_percent(
                probability_up
            ),
        )

    with k3:

        st.metric(
            "Confidence",
            format_percent(
                confidence
            ),
        )

    with k4:

        st.metric(
            "Inference latency",
            (
                f"{float(inference_latency):.1f} ms"
                if inference_latency is not None
                else "—"
            ),
        )

    with k5:

        st.metric(
            "Model version",
            str(
                selected_model_version
            ),
        )

    # ============================================================
    # MARKET HISTORY
    # ============================================================

    if not market_history_df.empty:

        market_history_df = (
            market_history_df.copy()
        )

        timestamp_column = None

        for candidate in [
            "open_time",
            "open_time_ms",
            "timestamp",
            "time",
        ]:

            if (
                candidate
                in market_history_df.columns
            ):

                timestamp_column = candidate
                break

        if timestamp_column:

            market_history_df[
                "timestamp"
            ] = normalize_datetime(
                market_history_df[
                    timestamp_column
                ]
            )

            market_history_df = (
                market_history_df.dropna(
                    subset=["timestamp"]
                )
            )

        close_column = next(
            (
                column
                for column in [
                    "close_price",
                    "close",
                ]
                if column
                in market_history_df.columns
            ),
            None,
        )

        open_column = next(
            (
                column
                for column in [
                    "open_price",
                    "open",
                ]
                if column
                in market_history_df.columns
            ),
            None,
        )

        high_column = next(
            (
                column
                for column in [
                    "high_price",
                    "high",
                ]
                if column
                in market_history_df.columns
            ),
            None,
        )

        low_column = next(
            (
                column
                for column in [
                    "low_price",
                    "low",
                ]
                if column
                in market_history_df.columns
            ),
            None,
        )

        volume_column = next(
            (
                column
                for column in [
                    "volume",
                ]
                if column
                in market_history_df.columns
            ),
            None,
        )

        # ========================================================
        # PRICE CHART
        # ========================================================

        st.html(
            """
            <div class="section-row">

                <div class="section-title">
                    Price & technical view
                </div>

                <div class="section-helper">
                    Live market candles
                </div>

            </div>
            """
        )

        if all(
            column is not None
            for column in [
                "timestamp",
                open_column,
                high_column,
                low_column,
                close_column,
            ]
        ):

            candle_fig = go.Figure()

            candle_fig.add_trace(
                go.Candlestick(
                    x=market_history_df[
                        "timestamp"
                    ],
                    open=market_history_df[
                        open_column
                    ],
                    high=market_history_df[
                        high_column
                    ],
                    low=market_history_df[
                        low_column
                    ],
                    close=market_history_df[
                        close_column
                    ],
                    name=selected_symbol,
                )
            )

            close_numeric = pd.to_numeric(
                market_history_df[
                    close_column
                ],
                errors="coerce",
            )

            # EMA 20

            ema_values = (
                close_numeric
                .ewm(
                    span=20,
                    adjust=False,
                )
                .mean()
            )

            candle_fig.add_trace(
                go.Scatter(
                    x=market_history_df[
                        "timestamp"
                    ],
                    y=ema_values,
                    mode="lines",
                    name="EMA 20",
                    line=dict(
                        width=1.5,
                    ),
                )
            )

            # Bollinger Bands

            rolling_mean = (
                close_numeric
                .rolling(
                    window=20
                )
                .mean()
            )

            rolling_std = (
                close_numeric
                .rolling(
                    window=20
                )
                .std()
            )

            upper_band = (
                rolling_mean
                + 2 * rolling_std
            )

            lower_band = (
                rolling_mean
                - 2 * rolling_std
            )

            candle_fig.add_trace(
                go.Scatter(
                    x=market_history_df[
                        "timestamp"
                    ],
                    y=upper_band,
                    mode="lines",
                    name="Bollinger upper",
                    line=dict(
                        width=1,
                        dash="dot",
                    ),
                )
            )

            candle_fig.add_trace(
                go.Scatter(
                    x=market_history_df[
                        "timestamp"
                    ],
                    y=lower_band,
                    mode="lines",
                    name="Bollinger lower",
                    line=dict(
                        width=1,
                        dash="dot",
                    ),
                )
            )

            candle_fig.update_layout(
                height=560,
                margin=dict(
                    l=0,
                    r=0,
                    t=10,
                    b=10,
                ),
                paper_bgcolor=(
                    "rgba(0,0,0,0)"
                ),
                plot_bgcolor=(
                    "rgba(0,0,0,0)"
                ),
                font=dict(
                    color="#f8fafc"
                ),
                xaxis=dict(
                    gridcolor=(
                        "rgba(148,163,184,0.08)"
                    ),
                    tickfont=dict(
                        color="#cbd5e1",
                        size=11,
                    ),
                    rangeslider=dict(
                        visible=False
                    ),
                ),
                yaxis=dict(
                    gridcolor=(
                        "rgba(148,163,184,0.08)"
                    ),
                    tickfont=dict(
                        color="#cbd5e1",
                        size=11,
                    ),
                    title="Price",
                    title_font=dict(
                        color="#f8fafc",
                        size=12,
                    ),
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    font=dict(
                        color="#f8fafc",
                        size=12,
                    ),
                ),
            )

            st.plotly_chart(
                candle_fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

        # ========================================================
        # VOLUME
        # ========================================================

        if (
            volume_column is not None
            and "timestamp"
            in market_history_df.columns
        ):

            volume_fig = go.Figure()

            volume_fig.add_trace(
                go.Bar(
                    x=market_history_df[
                        "timestamp"
                    ],
                    y=pd.to_numeric(
                        market_history_df[
                            volume_column
                        ],
                        errors="coerce",
                    ),
                    name="Volume",
                    marker_color="#2563eb",
                )
            )

            volume_fig.update_layout(
                height=230,
                margin=dict(
                    l=0,
                    r=0,
                    t=5,
                    b=5,
                ),
                paper_bgcolor=(
                    "rgba(0,0,0,0)"
                ),
                plot_bgcolor=(
                    "rgba(0,0,0,0)"
                ),
                font=dict(
                    color="#f8fafc"
                ),
                xaxis=dict(
                    gridcolor=(
                        "rgba(148,163,184,0.08)"
                    ),
                    tickfont=dict(
                        color="#cbd5e1",
                        size=11,
                    ),
                ),
                yaxis=dict(
                    gridcolor=(
                        "rgba(148,163,184,0.08)"
                    ),
                    tickfont=dict(
                        color="#cbd5e1",
                        size=11,
                    ),
                    title="Volume",
                    title_font=dict(
                        color="#f8fafc",
                        size=12,
                    ),
                ),
                showlegend=False,
            )

            st.plotly_chart(
                volume_fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

    # ============================================================
    # MODEL CONSENSUS
    # ============================================================

    st.html(
        """
        <div class="section-row">

            <div class="section-title">
                Model consensus
            </div>

            <div class="section-helper">
                CatBoost / GRU / ensemble
            </div>

        </div>
        """
    )

    consensus_columns = st.columns(3)

    catboost_probability = pick(
        selected_prediction or {},
        "catboost_probability_up",
        default=None,
    )

    gru_probability = pick(
        selected_prediction or {},
        "gru_probability_up",
        default=None,
    )

    ensemble_probability = pick(
        selected_prediction or {},
        "ensemble_probability_up",
        default=None,
    )

    with consensus_columns[0]:

        st.metric(
            "CatBoost UP",
            format_percent(
                catboost_probability
            ),
        )

    with consensus_columns[1]:

        st.metric(
            "GRU UP",
            format_percent(
                gru_probability
            ),
        )

    with consensus_columns[2]:

        st.metric(
            "Ensemble UP",
            format_percent(
                ensemble_probability
            ),
        )

    # ============================================================
    # SELECTED COIN HISTORY
    # ============================================================

    if not prediction_history_df.empty:

        # --------------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------------

        if (
            "timestamp"
            in prediction_history_df.columns
            and "ensemble_probability_up"
            in prediction_history_df.columns
        ):

            st.html(
                """
                <div class="section-row">

                    <div class="section-title">
                        Historical ensemble probability
                    </div>

                    <div class="section-helper">
                        Recent model output
                    </div>

                </div>
                """
            )

            probability_fig = go.Figure()

            probability_fig.add_trace(
                go.Scatter(
                    x=prediction_history_df[
                        "timestamp"
                    ],
                    y=(
                        pd.to_numeric(
                            prediction_history_df[
                                "ensemble_probability_up"
                            ],
                            errors="coerce",
                        ) * 100
                    ),
                    mode="lines",
                    name="UP probability",
                    line=dict(
                        width=2,
                    ),
                )
            )

            probability_fig.add_hline(
                y=50,
                line_dash="dot",
                line_color="rgba(148,163,184,0.40)",
                annotation_text="50% neutral",
                annotation_position="right",
                annotation_font=dict(
                    color="#cbd5e1",
                    size=11,
                ),
            )

            probability_fig.update_layout(
                height=310,
                margin=dict(
                    l=0,
                    r=0,
                    t=20,
                    b=5,
                ),
                paper_bgcolor=(
                    "rgba(0,0,0,0)"
                ),
                plot_bgcolor=(
                    "rgba(0,0,0,0)"
                ),
                font=dict(
                    color="#f8fafc"
                ),
                xaxis=dict(
                    gridcolor=(
                        "rgba(148,163,184,0.08)"
                    ),
                    tickfont=dict(
                        color="#cbd5e1"
                    ),
                ),
                yaxis=dict(
                    title="UP probability (%)",
                    range=[0, 100],
                    tickfont=dict(
                        color="#cbd5e1"
                    ),
                    title_font=dict(
                        color="#f8fafc"
                    ),
                ),
                showlegend=False,
            )

            st.plotly_chart(
                probability_fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

        # --------------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------------

        if (
            "timestamp"
            in prediction_history_df.columns
            and "ensemble_confidence"
            in prediction_history_df.columns
        ):

            st.html(
                """
                <div class="section-row">

                    <div class="section-title">
                        Prediction confidence
                    </div>

                    <div class="section-helper">
                        Confidence versus 55% action threshold
                    </div>

                </div>
                """
            )

            confidence_fig = go.Figure()

            confidence_fig.add_trace(
                go.Scatter(
                    x=prediction_history_df[
                        "timestamp"
                    ],
                    y=(
                        pd.to_numeric(
                            prediction_history_df[
                                "ensemble_confidence"
                            ],
                            errors="coerce",
                        ) * 100
                    ),
                    mode="lines",
                    name="Confidence",
                    line=dict(
                        width=2,
                    ),
                )
            )

            confidence_fig.add_hline(
                y=55,
                line_dash="dot",
                line_color="rgba(245,158,11,0.65)",
                annotation_text="55% action threshold",
                annotation_position="right",
                annotation_font=dict(
                    color="#fbbf24",
                    size=11,
                ),
            )

            confidence_fig.update_layout(
                height=310,
                margin=dict(
                    l=0,
                    r=0,
                    t=20,
                    b=5,
                ),
                paper_bgcolor=(
                    "rgba(0,0,0,0)"
                ),
                plot_bgcolor=(
                    "rgba(0,0,0,0)"
                ),
                font=dict(
                    color="#f8fafc"
                ),
                xaxis=dict(
                    gridcolor=(
                        "rgba(148,163,184,0.08)"
                    ),
                    tickfont=dict(
                        color="#cbd5e1"
                    ),
                ),
                yaxis=dict(
                    title="Confidence (%)",
                    range=[45, 65],
                    tickfont=dict(
                        color="#cbd5e1"
                    ),
                    title_font=dict(
                        color="#f8fafc"
                    ),
                ),
                showlegend=False,
            )

            st.plotly_chart(
                confidence_fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

        # ========================================================
        # TABLE
        # ========================================================

        st.html(
            """
            <div class="section-row">

                <div class="section-title">
                    Recent predictions
                </div>

                <div class="section-helper">
                    Latest model decisions
                </div>

            </div>
            """
        )

        table_df = (
            prediction_history_df.copy()
        )

        preferred_columns = [
            "timestamp",
            "ensemble_probability_up",
            "ensemble_probability_down",
            "ensemble_confidence",
            "raw_prediction",
            "signal",
            "actionable",
            "inference_latency_ms",
            "model_version",
        ]

        display_columns = [
            column
            for column in preferred_columns
            if column
            in table_df.columns
        ]

        if display_columns:

            table_df = table_df[
                display_columns
            ].copy()

            table_df = table_df.rename(
                columns={
                    "timestamp": "Time",
                    "ensemble_probability_up": "UP probability",
                    "ensemble_probability_down": "DOWN probability",
                    "ensemble_confidence": "Confidence",
                    "raw_prediction": "Raw prediction",
                    "signal": "Signal",
                    "actionable": "Actionable",
                    "inference_latency_ms": "Latency (ms)",
                    "model_version": "Model",
                }
            )

            for column in [
                "UP probability",
                "DOWN probability",
                "Confidence",
            ]:

                if column in table_df.columns:

                    values = pd.to_numeric(
                        table_df[column],
                        errors="coerce",
                    )

                    table_df[column] = (
                        values
                        .mul(100)
                        .round(2)
                        .map(
                            lambda x:
                            f"{x:.2f}%"
                            if pd.notna(x)
                            else "—"
                        )
                    )

            st.dataframe(
                table_df
                .tail(20)
                .iloc[::-1],
                use_container_width=True,
                hide_index=True,
            )

    # ============================================================
    # SELECTED COIN MLOPS HEALTH
    # ============================================================

    st.html(
        """
        <div class="section-row">

            <div class="section-title">
                MLOps health
            </div>

            <div class="section-helper">
                Production pipeline telemetry
            </div>

        </div>
        """
    )

    health_cols = st.columns(6)

    with health_cols[0]:

        st.metric(
            "Data freshness",
            str(
                pick(
                    monitoring_latest_row,
                    "data_freshness_status",
                    default="UNKNOWN",
                )
            ),
        )

    with health_cols[1]:

        st.metric(
            "Sequence",
            str(
                pick(
                    monitoring_latest_row,
                    "sequence_continuity_status",
                    default="UNKNOWN",
                )
            ),
        )

    with health_cols[2]:

        st.metric(
            "Predictions",
            str(
                pick(
                    monitoring_latest_row,
                    "prediction_completeness_status",
                    default="UNKNOWN",
                )
            ),
        )

    with health_cols[3]:

        st.metric(
            "Symbols",
            str(
                pick(
                    monitoring_latest_row,
                    "symbol_count",
                    default="—",
                )
            ),
        )

    with health_cols[4]:

        latest_latency = pick(
            monitoring_latest_row,
            "mean_inference_latency_ms",
            default=None,
        )

        st.metric(
            "Mean inference",
            (
                f"{float(latest_latency):.1f} ms"
                if latest_latency is not None
                else "—"
            ),
        )

    with health_cols[5]:

        latest_drift = pick(
            monitoring_latest_row,
            "feature_abs_z_gt3_pct",
            default=None,
        )

        st.metric(
            "|z| > 3",
            (
                f"{float(latest_drift):.2f}%"
                if latest_drift is not None
                else "—"
            ),
        )