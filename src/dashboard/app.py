from __future__ import annotations

import html
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = "http://127.0.0.1:8000"

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
]

TIMEFRAMES = [
    "1m",
    "5m",
    "1h",
    "4h",
]

HISTORY_LIMITS = [
    100,
    500,
    1000,
    5000,
]

MODEL_VERSION_FALLBACK = (
    "catboost_v1_gru_v1_ensemble_v1"
)


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
# HELPERS
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_price(value: Any) -> str:
    value = safe_float(value)

    if value >= 1000:
        return f"${value:,.2f}"

    if value >= 1:
        return f"${value:,.4f}"

    return f"${value:,.6f}"


def format_volume(value: Any) -> str:
    value = safe_float(value)

    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"

    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if value >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:,.0f}"


def signal_class(signal: Any) -> str:
    signal = str(signal).upper()

    if signal == "UP":
        return "signal-up"

    if signal == "DOWN":
        return "signal-down"

    return "signal-hold"


def signal_text(signal: Any) -> str:
    signal = str(signal).upper()

    if signal == "UP":
        return "↑ UP"

    if signal == "DOWN":
        return "↓ DOWN"

    return "— HOLD"


def to_dataframe(payload: Any) -> pd.DataFrame:

    if isinstance(payload, list):
        return pd.DataFrame(payload)

    if isinstance(payload, dict):

        if isinstance(
            payload.get("data"),
            list,
        ):
            return pd.DataFrame(
                payload["data"]
            )

        if isinstance(
            payload.get("results"),
            list,
        ):
            return pd.DataFrame(
                payload["results"]
            )

        return pd.DataFrame([payload])

    return pd.DataFrame()


def style_figure(
    fig: go.Figure,
    height: int = 420,
) -> go.Figure:

    fig.update_layout(
        height=height,
        margin=dict(
            l=20,
            r=20,
            t=35,
            b=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#cbd5e1",
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,.06)",
        zeroline=False,
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,.08)",
        zeroline=False,
    )

    return fig


def sync_symbol() -> None:
    st.query_params["symbol"] = (
        st.session_state["symbol_selector"]
    )


# ============================================================
# CSS
# ============================================================

st.html(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 90% 0%,
                rgba(59,130,246,.10),
                transparent 30%
            ),
            radial-gradient(
                circle at 0% 20%,
                rgba(139,92,246,.08),
                transparent 27%
            ),
            #080d18;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: #070c16;
        border-right: 1px solid rgba(148,163,184,.08);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* ======================================================
       HEADER
       ====================================================== */

    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1.7rem;
    }

    .app-title {
        font-size: 2.15rem;
        font-weight: 800;
        line-height: 1.05;
        color: #ffffff;
        letter-spacing: -0.035em;
    }

    .app-subtitle {
        margin-top: .5rem;
        color: #8ea0b8;
        font-size: .92rem;
    }

    .live-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 11px;
        border-radius: 999px;
        color: #86efac;
        background: rgba(34,197,94,.08);
        border: 1px solid rgba(34,197,94,.20);
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .04em;
    }

    .live-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow:
            0 0 12px rgba(34,197,94,.85);
    }

    /* ======================================================
       SECTION
       ====================================================== */

    .section {
        margin-top: 1.5rem;
        margin-bottom: .75rem;
    }

    .section-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 800;
        letter-spacing: -.01em;
    }

    .section-subtitle {
        color: #64748b;
        font-size: .74rem;
    }

    /* ======================================================
       HERO
       ====================================================== */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 1.55rem 1.65rem;
        border-radius: 20px;
        background:
            linear-gradient(
                135deg,
                rgba(21,31,49,.98),
                rgba(11,18,31,.98)
            );
        border: 1px solid rgba(148,163,184,.12);
        box-shadow:
            0 20px 60px rgba(0,0,0,.18);
    }

    .hero-glow {
        position: absolute;
        width: 280px;
        height: 280px;
        right: -110px;
        top: -140px;
        border-radius: 50%;
        background: rgba(59,130,246,.08);
        filter: blur(10px);
    }

    .hero-row {
        position: relative;
        z-index: 2;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }

    .hero-symbol {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 800;
    }

    .hero-market {
        color: #64748b;
        font-size: .74rem;
        margin-top: .25rem;
    }

    .hero-price {
        position: relative;
        z-index: 2;
        margin-top: 1.25rem;
        color: #ffffff;
        font-size: 2.9rem;
        font-weight: 850;
        letter-spacing: -.045em;
    }

    .hero-price-label {
        position: relative;
        z-index: 2;
        color: #64748b;
        font-size: .73rem;
        margin-top: .35rem;
    }

    .hero-confidence {
        position: relative;
        z-index: 2;
        margin-top: 1rem;
        color: #94a3b8;
        font-size: .82rem;
    }

    .hero-confidence strong {
        color: #f8fafc;
    }

    /* ======================================================
       SIGNALS
       ====================================================== */

    .signal {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 8px 12px;
        border-radius: 10px;
        font-size: .8rem;
        font-weight: 850;
        letter-spacing: .03em;
    }

    .signal-up {
        color: #86efac;
        background: rgba(34,197,94,.10);
        border: 1px solid rgba(34,197,94,.24);
    }

    .signal-down {
        color: #fca5a5;
        background: rgba(239,68,68,.10);
        border: 1px solid rgba(239,68,68,.24);
    }

    .signal-hold {
        color: #cbd5e1;
        background: rgba(148,163,184,.08);
        border: 1px solid rgba(148,163,184,.16);
    }

    /* ======================================================
       KPI CARDS
       ====================================================== */

    .card {
        min-height: 92px;
        padding: 1rem;
        border-radius: 16px;
        background: rgba(13,21,36,.86);
        border: 1px solid rgba(148,163,184,.09);
    }

    .card-label {
        color: #64748b;
        font-size: .68rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: .06em;
    }

    .card-value {
        margin-top: .35rem;
        color: #f8fafc;
        font-size: 1.18rem;
        font-weight: 800;
    }

    .card-small {
        margin-top: .2rem;
        color: #64748b;
        font-size: .67rem;
    }

    /* ======================================================
       MARKET OVERVIEW CARDS
       ====================================================== */

    .coin-card-wrap {
        text-decoration: none;
        color: inherit;
        display: block;
    }

    .coin-card {
        min-height: 148px;
        padding: 1.05rem 1.05rem 1rem;
        border-radius: 17px;
        background:
            linear-gradient(
                145deg,
                rgba(17,27,45,.98),
                rgba(11,18,31,.98)
            );
        border: 1px solid rgba(148,163,184,.10);
        box-shadow:
            0 10px 30px rgba(0,0,0,.10);
        transition:
            transform .15s ease,
            border-color .15s ease,
            background .15s ease;
    }

    .coin-card:hover {
        border-color: rgba(96,165,250,.35);
        transform: translateY(-2px);
        background:
            linear-gradient(
                145deg,
                rgba(22,34,56,1),
                rgba(12,21,36,1)
            );
    }

    .coin-card-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: .5rem;
    }

    .coin-card-symbol {
        color: #f1f5f9;
        font-weight: 800;
        font-size: .88rem;
        letter-spacing: .01em;
    }

    .coin-card-price {
        margin-top: .95rem;
        color: #ffffff;
        font-size: 1.34rem;
        font-weight: 800;
        letter-spacing: -.02em;
    }

    .coin-card-meta {
        margin-top: .45rem;
        color: #64748b;
        font-size: .73rem;
    }

    .coin-card-meta strong {
        color: #cbd5e1;
    }

    .coin-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 4px 8px;
        border-radius: 7px;
        font-size: .64rem;
        font-weight: 850;
        white-space: nowrap;
    }

    .confidence-track {
        margin-top: .7rem;
        width: 100%;
        height: 3px;
        border-radius: 999px;
        background: rgba(148,163,184,.08);
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 999px;
        background: #60a5fa;
    }

    /* ======================================================
       HEALTH
       ====================================================== */

    .health-card {
        min-height: 90px;
        padding: 1rem;
        border-radius: 15px;
        background: rgba(13,21,36,.82);
        border: 1px solid rgba(148,163,184,.09);
    }

    .health-label {
        color: #64748b;
        font-size: .68rem;
        text-transform: uppercase;
        letter-spacing: .05em;
    }

    .healthy {
        color: #86efac;
        font-weight: 800;
    }

    .warning {
        color: #fcd34d;
        font-weight: 800;
    }

    .failed {
        color: #fca5a5;
        font-weight: 800;
    }

    /* ======================================================
       SIDEBAR
       ====================================================== */

    .sidebar-brand {
        color: #f8fafc;
        font-weight: 850;
        font-size: 1.15rem;
    }

    .sidebar-sub {
        color: #64748b;
        font-size: .74rem;
        margin-top: .18rem;
        margin-bottom: 1.4rem;
    }

    .sidebar-info {
        margin-top: 1rem;
        padding: .9rem;
        border-radius: 12px;
        background: rgba(30,41,59,.42);
        border: 1px solid rgba(148,163,184,.08);
        color: #64748b;
        font-size: .69rem;
        line-height: 1.55;
    }

    .sidebar-info strong {
        color: #94a3b8;
    }

    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(148,163,184,.08);
        text-align: center;
        color: #475569;
        font-size: .69rem;
    }

    </style>
    """
)


# ============================================================
# API
# ============================================================

@st.cache_data(ttl=3)
def api_get(
    endpoint: str,
) -> Any:

    response = requests.get(
        f"{API_BASE_URL}{endpoint}",
        timeout=20,
    )

    response.raise_for_status()

    return response.json()


@st.cache_data(ttl=3)
def get_predictions() -> pd.DataFrame:

    return to_dataframe(
        api_get(
            "/predictions/latest"
        )
    )


@st.cache_data(ttl=3)
def get_market() -> pd.DataFrame:

    return to_dataframe(
        api_get(
            "/market/latest"
        )
    )


@st.cache_data(ttl=3)
def get_monitoring() -> dict:

    payload = api_get(
        "/monitoring/latest"
    )

    return (
        payload
        if isinstance(
            payload,
            dict,
        )
        else {}
    )


@st.cache_data(ttl=10)
def get_prediction_history(
    symbol: str,
) -> pd.DataFrame:

    payload = api_get(
        "/predictions/history"
        f"?symbol={symbol}"
        "&limit=5000"
    )

    df = to_dataframe(
        payload
    )

    if df.empty:
        return df

    if "candle_time" in df.columns:

        df["candle_time"] = pd.to_datetime(
            df["candle_time"],
            utc=True,
            errors="coerce",
        )

    for column in [
        "ensemble_probability_up",
        "ensemble_confidence",
        "inference_latency_ms",
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return (
        df
        .dropna(
            subset=["candle_time"]
        )
        .sort_values(
            "candle_time"
        )
    )


@st.cache_data(ttl=10)
def get_market_history(
    symbol: str,
    interval: str,
    limit: int,
) -> pd.DataFrame:

    payload = api_get(
        "/market/history"
        f"?symbol={symbol}"
        f"&interval={interval}"
        f"&limit={limit}"
    )

    df = to_dataframe(
        payload
    )

    if df.empty:
        return df

    if "open_time" in df.columns:

        df["open_time"] = pd.to_datetime(
            df["open_time"],
            utc=True,
            errors="coerce",
        )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return (
        df
        .dropna(
            subset=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
        .sort_values(
            "open_time"
        )
    )


# ============================================================
# SIDEBAR
# ============================================================

query_symbol = st.query_params.get(
    "symbol",
    "All",
)

if query_symbol not in (
    ["All"] + SYMBOLS
):
    query_symbol = "All"


with st.sidebar:

    st.html(
        """
        <div class="sidebar-brand">
            ₿ Crypto MLOps
        </div>

        <div class="sidebar-sub">
            Live prediction & monitoring
        </div>
        """
    )

    st.markdown("### Market")

    selected_symbol = st.selectbox(
        "Coin",
        ["All"] + SYMBOLS,
        index=(
            ["All"] + SYMBOLS
        ).index(
            query_symbol
        ),
        key="symbol_selector",
        on_change=sync_symbol,
        label_visibility="collapsed",
    )

    st.markdown("### Dashboard")

    if st.button(
        "↻  Refresh data",
        use_container_width=True,
    ):

        st.cache_data.clear()
        st.rerun()

    st.html(
        """
        <div class="sidebar-info">

            <strong>
                Data flow
            </strong><br>

            Binance → SQLite →
            Resident Inference →
            FastAPI → Streamlit

        </div>

        <div class="sidebar-info">

            <strong>
                Production model
            </strong><br>

            CatBoost + GRU<br>
            5-minute direction<br>
            Confidence threshold: 0.55

        </div>
        """
    )


# ============================================================
# LOAD DATA
# ============================================================

try:

    predictions = get_predictions()
    market = get_market()
    monitoring = get_monitoring()

except Exception as exc:

    st.error(
        "FastAPI connection failed."
    )

    st.code(
        str(exc)
    )

    st.stop()


# ============================================================
# NORMALIZE
# ============================================================

if "candle_time" in predictions.columns:

    predictions["candle_time"] = (
        pd.to_datetime(
            predictions["candle_time"],
            utc=True,
            errors="coerce",
        )
    )

if "open_time" in market.columns:

    market["open_time"] = (
        pd.to_datetime(
            market["open_time"],
            utc=True,
            errors="coerce",
        )
    )


all_predictions = predictions.copy()
all_market = market.copy()


# ============================================================
# FILTER CURRENT VIEW
# ============================================================

if selected_symbol != "All":

    predictions = predictions[
        predictions["symbol"]
        == selected_symbol
    ].copy()

    market = market[
        market["symbol"]
        == selected_symbol
    ].copy()


# ============================================================
# HEADER
# ============================================================

st.html(
    """
    <div class="app-header">

        <div>

            <div class="app-title">
                Crypto MLOps Control Center
            </div>

            <div class="app-subtitle">
                Live market intelligence, model inference
                and production health monitoring
            </div>

        </div>

        <div class="live-pill">
            <span class="live-dot"></span>
            LIVE PIPELINE
        </div>

    </div>
    """
)


# ============================================================
# SINGLE COIN VIEW
# ============================================================

if selected_symbol != "All":

    if predictions.empty:

        st.warning(
            "No prediction is currently available "
            "for this coin."
        )

        st.stop()

    prediction = predictions.iloc[0]

    market_row = (
        market.iloc[0]
        if not market.empty
        else None
    )

    signal = str(
        prediction.get(
            "signal",
            "HOLD",
        )
    ).upper()

    confidence = safe_float(
        prediction.get(
            "ensemble_confidence",
            0,
        )
    )

    probability_up = safe_float(
        prediction.get(
            "ensemble_probability_up",
            0,
        )
    )

    latency = safe_float(
        prediction.get(
            "inference_latency_ms",
            0,
        )
    )

    price = (
        safe_float(
            market_row.get(
                "close",
                0,
            )
        )
        if market_row is not None
        else safe_float(
            prediction.get(
                "close_price",
                0,
            )
        )
    )

    candle_time = prediction.get(
        "candle_time"
    )

    candle_text = (
        pd.Timestamp(
            candle_time
        ).strftime(
            "%H:%M:%S UTC"
        )
        if pd.notna(
            candle_time
        )
        else "—"
    )

    # ========================================================
    # HERO
    # ========================================================

    st.html(
        """
        <div class="section">

            <div class="section-row">

                <div class="section-title">
                    Market & model state
                </div>

                <div class="section-subtitle">
                    Selected asset
                </div>

            </div>

        </div>
        """
    )

    st.html(
        f"""
        <div class="hero">

            <div class="hero-glow"></div>

            <div class="hero-row">

                <div>

                    <div class="hero-symbol">
                        {html.escape(
                            selected_symbol
                        )}
                    </div>

                    <div class="hero-market">
                        Binance Spot • 1-minute candle
                    </div>

                </div>

                <div class="signal {
                    signal_class(signal)
                }">

                    {signal_text(signal)}

                </div>

            </div>

            <div class="hero-price">
                {format_price(price)}
            </div>

            <div class="hero-price-label">
                Latest closed market price
            </div>

            <div class="hero-confidence">

                Ensemble confidence:
                <strong>
                    {confidence:.2%}
                </strong>

                &nbsp; • &nbsp;

                UP probability:
                <strong>
                    {probability_up:.2%}
                </strong>

            </div>

        </div>
        """
    )

    st.html(
        "<div style='height:12px'></div>"
    )

    # ========================================================
    # KPI
    # ========================================================

    kpi_cols = st.columns(4)

    with kpi_cols[0]:

        st.html(
            f"""
            <div class="card">

                <div class="card-label">
                    Signal
                </div>

                <div class="card-value">
                    <span class="{
                        signal_class(signal)
                    }">
                        {signal_text(signal)}
                    </span>
                </div>

                <div class="card-small">
                    5-minute direction
                </div>

            </div>
            """
        )

    with kpi_cols[1]:

        st.html(
            f"""
            <div class="card">

                <div class="card-label">
                    Confidence
                </div>

                <div class="card-value">
                    {confidence:.2%}
                </div>

                <div class="card-small">
                    Threshold: 55%
                </div>

            </div>
            """
        )

    with kpi_cols[2]:

        st.html(
            f"""
            <div class="card">

                <div class="card-label">
                    Inference latency
                </div>

                <div class="card-value">
                    {latency:.1f} ms
                </div>

                <div class="card-small">
                    Resident inference service
                </div>

            </div>
            """
        )

    with kpi_cols[3]:

        st.html(
            f"""
            <div class="card">

                <div class="card-label">
                    Latest candle
                </div>

                <div class="card-value">
                    {candle_text}
                </div>

                <div class="card-small">
                    Most recent prediction
                </div>

            </div>
            """
        )

    # ========================================================
    # CANDLESTICK MARKET HISTORY
    # ========================================================

    st.html(
        """
        <div class="section">

            <div class="section-row">

                <div class="section-title">
                    Price history
                </div>

                <div class="section-subtitle">
                    OHLCV market data
                </div>

            </div>

        </div>
        """
    )

    chart_cols = st.columns(
        [1, 1]
    )

    with chart_cols[0]:

        selected_interval = st.radio(
            "Timeframe",
            TIMEFRAMES,
            horizontal=True,
            key="market_timeframe",
        )

    with chart_cols[1]:

        selected_limit = st.selectbox(
            "History",
            HISTORY_LIMITS,
            index=1,
            key="market_history_limit",
            format_func=lambda value:
                f"{value:,} candles",
        )

    market_history = get_market_history(
        selected_symbol,
        selected_interval,
        selected_limit,
    )

    if market_history.empty:

        st.info(
            "No market history is available "
            "for this asset."
        )

    else:

        # ----------------------------------------------------
        # EMA 20
        # ----------------------------------------------------

        market_history["ema_20"] = (
            market_history["close"]
            .ewm(
                span=20,
                adjust=False,
            )
            .mean()
        )

        # ----------------------------------------------------
        # BOLLINGER BANDS
        # ----------------------------------------------------

        rolling_mean = (
            market_history["close"]
            .rolling(
                20
            )
            .mean()
        )

        rolling_std = (
            market_history["close"]
            .rolling(
                20
            )
            .std()
        )

        market_history["bb_middle"] = (
            rolling_mean
        )

        market_history["bb_upper"] = (
            rolling_mean
            + 2 * rolling_std
        )

        market_history["bb_lower"] = (
            rolling_mean
            - 2 * rolling_std
        )

        # ----------------------------------------------------
        # CANDLESTICK
        # ----------------------------------------------------

        candle_fig = go.Figure()

        candle_fig.add_trace(
            go.Candlestick(
                x=market_history[
                    "open_time"
                ],
                open=market_history[
                    "open"
                ],
                high=market_history[
                    "high"
                ],
                low=market_history[
                    "low"
                ],
                close=market_history[
                    "close"
                ],
                name=selected_symbol,
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
                increasing_fillcolor="#22c55e",
                decreasing_fillcolor="#ef4444",
            )
        )

        candle_fig.add_trace(
            go.Scatter(
                x=market_history[
                    "open_time"
                ],
                y=market_history[
                    "ema_20"
                ],
                mode="lines",
                name="EMA 20",
                line=dict(
                    color="#60a5fa",
                    width=1.7,
                ),
            )
        )

        candle_fig.add_trace(
            go.Scatter(
                x=market_history[
                    "open_time"
                ],
                y=market_history[
                    "bb_upper"
                ],
                mode="lines",
                name="BB Upper",
                line=dict(
                    color="#64748b",
                    width=1,
                    dash="dot",
                ),
            )
        )

        candle_fig.add_trace(
            go.Scatter(
                x=market_history[
                    "open_time"
                ],
                y=market_history[
                    "bb_lower"
                ],
                mode="lines",
                name="BB Lower",
                line=dict(
                    color="#64748b",
                    width=1,
                    dash="dot",
                ),
            )
        )

        candle_fig.update_layout(
            height=600,
            margin=dict(
                l=15,
                r=15,
                t=35,
                b=15,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#cbd5e1"
            ),
            xaxis=dict(
                rangeslider=dict(
                    visible=True,
                    thickness=0.08,
                ),
                gridcolor=(
                    "rgba(148,163,184,.06)"
                ),
            ),
            yaxis=dict(
                title="Price",
                gridcolor=(
                    "rgba(148,163,184,.08)"
                ),
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )

        st.plotly_chart(
            candle_fig,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "scrollZoom": True,
            },
        )

        # ----------------------------------------------------
        # VOLUME
        # ----------------------------------------------------

        volume_colors = [
            (
                "#22c55e"
                if close_price >= open_price
                else "#ef4444"
            )
            for open_price, close_price
            in zip(
                market_history[
                    "open"
                ],
                market_history[
                    "close"
                ],
            )
        ]

        volume_fig = go.Figure()

        volume_fig.add_trace(
            go.Bar(
                x=market_history[
                    "open_time"
                ],
                y=market_history[
                    "volume"
                ],
                name="Volume",
                marker_color=volume_colors,
            )
        )

        volume_fig.update_layout(
            height=220,
            margin=dict(
                l=15,
                r=15,
                t=25,
                b=15,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#cbd5e1"
            ),
            xaxis=dict(
                gridcolor=(
                    "rgba(148,163,184,.06)"
                ),
            ),
            yaxis=dict(
                title="Volume",
                gridcolor=(
                    "rgba(148,163,184,.08)"
                ),
            ),
            showlegend=False,
        )

        st.plotly_chart(
            volume_fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

    # ========================================================
    # MARKET SNAPSHOT
    # ========================================================

    st.html(
        """
        <div class="section">

            <div class="section-row">

                <div class="section-title">
                    Market snapshot
                </div>

            </div>

        </div>
        """
    )

    if market_row is not None:

        market_values = [
            (
                "Open",
                format_price(
                    market_row.get(
                        "open",
                        0,
                    )
                ),
            ),
            (
                "High",
                format_price(
                    market_row.get(
                        "high",
                        0,
                    )
                ),
            ),
            (
                "Low",
                format_price(
                    market_row.get(
                        "low",
                        0,
                    )
                ),
            ),
            (
                "Volume",
                format_volume(
                    market_row.get(
                        "volume",
                        0,
                    )
                ),
            ),
        ]

        cols = st.columns(4)

        for col, (
            label,
            value,
        ) in zip(
            cols,
            market_values,
        ):

            with col:

                st.html(
                    f"""
                    <div class="card">

                        <div class="card-label">
                            {label}
                        </div>

                        <div class="card-value">
                            {value}
                        </div>

                    </div>
                    """
                )

    # ========================================================
    # MODEL CONSENSUS
    # ========================================================

    st.html(
        """
        <div class="section">

            <div class="section-row">

                <div class="section-title">
                    Model consensus
                </div>

                <div class="section-subtitle">
                    Probability of upward 5-minute direction
                </div>

            </div>

        </div>
        """
    )

    model_df = pd.DataFrame(
        {
            "Model": [
                "CatBoost",
                "GRU",
                "Ensemble",
            ],
            "Probability": [
                safe_float(
                    prediction.get(
                        "catboost_probability_up",
                        0,
                    )
                ),
                safe_float(
                    prediction.get(
                        "gru_probability_up",
                        0,
                    )
                ),
                safe_float(
                    prediction.get(
                        "ensemble_probability_up",
                        0,
                    )
                ),
            ],
        }
    )

    model_fig = px.bar(
        model_df,
        x="Model",
        y="Probability",
        text="Probability",
        range_y=[0, 1],
        template="plotly_dark",
    )

    model_fig.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside",
        marker_line_width=0,
    )

    model_fig.add_hline(
        y=0.55,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="55% threshold",
    )

    model_fig = style_figure(
        model_fig,
        380,
    )

    model_fig.update_yaxes(
        tickformat=".0%"
    )

    model_fig.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        model_fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    # ========================================================
    # PREDICTION HISTORY
    # ========================================================

    prediction_history = (
        get_prediction_history(
            selected_symbol
        )
    )

    if not prediction_history.empty:

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        st.html(
            """
            <div class="section">

                <div class="section-row">

                    <div class="section-title">
                        Ensemble confidence over time
                    </div>

                    <div class="section-subtitle">
                        Recent confidence behavior
                    </div>

                </div>

            </div>
            """
        )

        confidence_fig = go.Figure()

        confidence_fig.add_trace(
            go.Scatter(
                x=prediction_history[
                    "candle_time"
                ],
                y=prediction_history[
                    "ensemble_confidence"
                ],
                mode="lines+markers",
                name="Confidence",
                line=dict(
                    color="#60a5fa",
                    width=2.5,
                ),
                marker=dict(
                    size=5,
                ),
            )
        )

        confidence_fig.add_hline(
            y=0.55,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="55%",
        )

        confidence_fig.update_yaxes(
            range=[0, 1],
            tickformat=".0%",
            title="Confidence",
        )

        confidence_fig = style_figure(
            confidence_fig,
            420,
        )

        st.plotly_chart(
            confidence_fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

        # ----------------------------------------------------
        # UP PROBABILITY
        # ----------------------------------------------------

        st.html(
            """
            <div class="section">

                <div class="section-row">

                    <div class="section-title">
                        Historical ensemble UP probability
                    </div>

                    <div class="section-subtitle">
                        Directional probability over time
                    </div>

                </div>

            </div>
            """
        )

        probability_fig = go.Figure()

        probability_fig.add_trace(
            go.Scatter(
                x=prediction_history[
                    "candle_time"
                ],
                y=prediction_history[
                    "ensemble_probability_up"
                ],
                mode="lines+markers",
                name="UP probability",
                line=dict(
                    color="#a78bfa",
                    width=2.5,
                ),
                marker=dict(
                    size=5,
                ),
            )
        )

        probability_fig.add_hline(
            y=0.50,
            line_dash="dot",
            line_color="#64748b",
            annotation_text="50%",
        )

        probability_fig.add_hline(
            y=0.55,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="55%",
        )

        probability_fig.update_yaxes(
            range=[0, 1],
            tickformat=".0%",
            title="UP probability",
        )

        probability_fig = style_figure(
            probability_fig,
            420,
        )

        st.plotly_chart(
            probability_fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

        # ----------------------------------------------------
        # RECENT PREDICTIONS
        # ----------------------------------------------------

        st.html(
            """
            <div class="section">

                <div class="section-title">
                    Recent predictions
                </div>

            </div>
            """
        )

        table = prediction_history.copy()

        table[
            "candle_time"
        ] = (
            table[
                "candle_time"
            ]
            .dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        columns = [
            "candle_time",
            "symbol",
            "ensemble_probability_up",
            "ensemble_confidence",
            "signal",
            "inference_latency_ms",
        ]

        columns = [
            column
            for column in columns
            if column in table.columns
        ]

        table = (
            table[
                columns
            ]
            .sort_values(
                "candle_time",
                ascending=False,
            )
            .head(20)
        )

        table = table.rename(
            columns={
                "candle_time":
                    "Candle time",
                "symbol":
                    "Symbol",
                "ensemble_probability_up":
                    "UP probability",
                "ensemble_confidence":
                    "Confidence",
                "signal":
                    "Signal",
                "inference_latency_ms":
                    "Latency (ms)",
            }
        )

        if "UP probability" in table.columns:

            table[
                "UP probability"
            ] = (
                table[
                    "UP probability"
                ]
                .map(
                    lambda value:
                    f"{safe_float(value):.2%}"
                )
            )

        if "Confidence" in table.columns:

            table[
                "Confidence"
            ] = (
                table[
                    "Confidence"
                ]
                .map(
                    lambda value:
                    f"{safe_float(value):.2%}"
                )
            )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# ALL COINS VIEW
# ============================================================

else:

    st.html(
        """
        <div class="section">

            <div class="section-row">

                <div class="section-title">
                    Market overview
                </div>

                <div class="section-subtitle">
                    Click a card to open the full market view
                </div>

            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # MARKET CARDS
    # --------------------------------------------------------

    coin_cols = st.columns(
        5,
        gap="small",
    )

    for col, symbol in zip(
        coin_cols,
        SYMBOLS,
    ):

        pred = all_predictions[
            all_predictions["symbol"]
            == symbol
        ]

        mkt = all_market[
            all_market["symbol"]
            == symbol
        ]

        if pred.empty:

            signal = "HOLD"
            confidence = 0.0
            probability_up = 0.0
            price_text = "No data"

        else:

            row = pred.iloc[0]

            signal = str(
                row.get(
                    "signal",
                    "HOLD",
                )
            ).upper()

            confidence = safe_float(
                row.get(
                    "ensemble_confidence",
                    0,
                )
            )

            probability_up = safe_float(
                row.get(
                    "ensemble_probability_up",
                    0,
                )
            )

            price_text = (
                format_price(
                    mkt.iloc[0][
                        "close"
                    ]
                )
                if not mkt.empty
                else format_price(
                    row.get(
                        "close_price",
                        0,
                    )
                )
            )

        confidence_percent = min(
            max(
                confidence * 100,
                0,
            ),
            100,
        )

        card_html = f"""
        <a
            class="coin-card-wrap"
            href="?symbol={symbol}"
        >

            <div class="coin-card">

                <div class="coin-card-top">

                    <div class="coin-card-symbol">
                        {symbol}
                    </div>

                    <div
                        class="
                            coin-badge
                            {signal_class(signal)}
                        "
                    >
                        {signal_text(signal)}
                    </div>

                </div>

                <div class="coin-card-price">
                    {price_text}
                </div>

                <div class="coin-card-meta">
                    Confidence:
                    <strong>
                        {confidence:.2%}
                    </strong>

                    &nbsp;•&nbsp;

                    UP:
                    <strong>
                        {probability_up:.2%}
                    </strong>
                </div>

                <div class="confidence-track">

                    <div
                        class="confidence-fill"
                        style="
                            width:
                            {confidence_percent:.1f}%;
                        "
                    >
                    </div>

                </div>

            </div>

        </a>
        """

        with col:

            st.html(
                card_html
            )

    # --------------------------------------------------------
    # ENSEMBLE MARKET VIEW
    # --------------------------------------------------------

    st.html(
        """
        <div class="section">

            <div class="section-row">

                <div class="section-title">
                    Ensemble market view
                </div>

                <div class="section-subtitle">
                    Current UP probability by asset
                </div>

            </div>

        </div>
        """
    )

    if not all_predictions.empty:

        market_view = (
            all_predictions[
                [
                    "symbol",
                    "ensemble_probability_up",
                ]
            ]
            .copy()
        )

        market_view[
            "ensemble_probability_up"
        ] = pd.to_numeric(
            market_view[
                "ensemble_probability_up"
            ],
            errors="coerce",
        )

        market_fig = px.bar(
            market_view,
            x="symbol",
            y="ensemble_probability_up",
            text="ensemble_probability_up",
            range_y=[0, 1],
            template="plotly_dark",
        )

        market_fig.update_traces(
            texttemplate="%{text:.2%}",
            textposition="outside",
            marker_line_width=0,
        )

        market_fig.add_hline(
            y=0.55,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="55%",
        )

        market_fig.add_hline(
            y=0.50,
            line_dash="dot",
            line_color="#64748b",
            annotation_text="50%",
        )

        market_fig = style_figure(
            market_fig,
            400,
        )

        market_fig.update_yaxes(
            tickformat=".0%"
        )

        market_fig.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            market_fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

        # ----------------------------------------------------
        # ALL COIN HISTORY
        # ----------------------------------------------------

        all_history_frames = []

        for symbol in SYMBOLS:

            symbol_history = (
                get_prediction_history(
                    symbol
                )
            )

            if symbol_history.empty:
                continue

            all_history_frames.append(
                symbol_history
            )

        if all_history_frames:

            all_history = pd.concat(
                all_history_frames,
                ignore_index=True,
            )

            all_history = (
                all_history
                .dropna(
                    subset=[
                        "candle_time"
                    ]
                )
                .sort_values(
                    "candle_time"
                )
            )

            # ------------------------------------------------
            # CONFIDENCE HISTORY
            # ------------------------------------------------

            st.html(
                """
                <div class="section">

                    <div class="section-row">

                        <div class="section-title">
                            Ensemble confidence over time
                        </div>

                        <div class="section-subtitle">
                            All monitored assets
                        </div>

                    </div>

                </div>
                """
            )

            confidence_fig = go.Figure()

            for symbol in SYMBOLS:

                symbol_data = all_history[
                    all_history[
                        "symbol"
                    ] == symbol
                ]

                if symbol_data.empty:
                    continue

                confidence_fig.add_trace(
                    go.Scatter(
                        x=symbol_data[
                            "candle_time"
                        ],
                        y=symbol_data[
                            "ensemble_confidence"
                        ],
                        mode="lines",
                        name=symbol,
                        line=dict(
                            width=2,
                        ),
                    )
                )

            confidence_fig.add_hline(
                y=0.55,
                line_dash="dash",
                line_color="#f59e0b",
                annotation_text="55%",
            )

            confidence_fig.update_yaxes(
                range=[0, 1],
                tickformat=".0%",
                title="Confidence",
            )

            confidence_fig = style_figure(
                confidence_fig,
                440,
            )

            st.plotly_chart(
                confidence_fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                },
            )

            # ------------------------------------------------
            # UP PROBABILITY HISTORY
            # ------------------------------------------------

            st.html(
                """
                <div class="section">

                    <div class="section-row">

                        <div class="section-title">
                            Historical ensemble UP probability
                        </div>

                        <div class="section-subtitle">
                            All monitored assets
                        </div>

                    </div>

                </div>
                """
            )

            probability_fig = go.Figure()

            for symbol in SYMBOLS:

                symbol_data = all_history[
                    all_history[
                        "symbol"
                    ] == symbol
                ]

                if symbol_data.empty:
                    continue

                probability_fig.add_trace(
                    go.Scatter(
                        x=symbol_data[
                            "candle_time"
                        ],
                        y=symbol_data[
                            "ensemble_probability_up"
                        ],
                        mode="lines",
                        name=symbol,
                        line=dict(
                            width=2,
                        ),
                    )
                )

            probability_fig.add_hline(
                y=0.50,
                line_dash="dot",
                line_color="#64748b",
                annotation_text="50%",
            )

            probability_fig.add_hline(
                y=0.55,
                line_dash="dash",
                line_color="#f59e0b",
                annotation_text="55%",
            )

            probability_fig.update_yaxes(
                range=[0, 1],
                tickformat=".0%",
                title="UP probability",
            )

            probability_fig = style_figure(
                probability_fig,
                440,
            )

            st.plotly_chart(
                probability_fig,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                },
            )

    # --------------------------------------------------------
    # SIGNAL DISTRIBUTION
    # --------------------------------------------------------

    st.html(
        """
        <div class="section">

            <div class="section-row">

                <div class="section-title">
                    Signal distribution
                </div>

                <div class="section-subtitle">
                    Current ensemble decisions
                </div>

            </div>

        </div>
        """
    )

    if not all_predictions.empty:

        signal_counts = (
            all_predictions[
                "signal"
            ]
            .astype(str)
            .str.upper()
            .value_counts()
            .reset_index()
        )

        signal_counts.columns = [
            "Signal",
            "Count",
        ]

        signal_fig = px.pie(
            signal_counts,
            names="Signal",
            values="Count",
            hole=0.62,
            template="plotly_dark",
        )

        signal_fig.update_layout(
            height=380,
            margin=dict(
                l=15,
                r=15,
                t=15,
                b=15,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.05,
                xanchor="center",
                x=0.5,
            ),
        )

        st.plotly_chart(
            signal_fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )


# ============================================================
# MLOPS HEALTH
# ============================================================

st.html(
    """
    <div class="section">

        <div class="section-row">

            <div class="section-title">
                MLOps health
            </div>

            <div class="section-subtitle">
                Pipeline-wide monitoring
            </div>

        </div>

    </div>
    """
)


freshness = str(
    monitoring.get(
        "data_freshness_status",
        "UNKNOWN",
    )
).upper()

prediction_status = str(
    monitoring.get(
        "prediction_completeness_status",
        "UNKNOWN",
    )
).upper()

sequence_status = str(
    monitoring.get(
        "sequence_continuity_status",
        "UNKNOWN",
    )
).upper()

data_age = safe_float(
    monitoring.get(
        "data_age_seconds",
        0,
    )
)

mean_latency = safe_float(
    monitoring.get(
        "mean_inference_latency_ms",
        0,
    )
)

mean_confidence = safe_float(
    monitoring.get(
        "mean_confidence",
        0,
    )
)

feature_z_max = safe_float(
    monitoring.get(
        "feature_abs_z_max",
        0,
    )
)

feature_z_percent = safe_float(
    monitoring.get(
        "feature_abs_z_gt3_pct",
        0,
    )
)


def health_value(
    status: str,
) -> str:

    if status == "PASS":

        return (
            '<span class="healthy">'
            "● HEALTHY"
            "</span>"
        )

    if status in {
        "WARN",
        "WARNING",
    }:

        return (
            '<span class="warning">'
            "● WARNING"
            "</span>"
        )

    if status == "FAIL":

        return (
            '<span class="failed">'
            "● FAILED"
            "</span>"
        )

    return html.escape(
        status
    )


health_items = [
    (
        "Data freshness",
        health_value(
            freshness
        ),
    ),
    (
        "Predictions",
        health_value(
            prediction_status
        ),
    ),
    (
        "Sequence",
        health_value(
            sequence_status
        ),
    ),
    (
        "Feature drift",
        f"Max |z| {feature_z_max:.2f}",
    ),
    (
        "Inference",
        f"{mean_latency:.1f} ms",
    ),
]


health_cols = st.columns(5)

for col, (
    label,
    value,
) in zip(
    health_cols,
    health_items,
):

    with col:

        st.html(
            f"""
            <div class="health-card">

                <div class="health-label">
                    {label}
                </div>

                <div style="
                    margin-top:.35rem;
                    color:#e2e8f0;
                    font-size:.9rem;
                    font-weight:800;
                ">
                    {value}
                </div>

            </div>
            """
        )


# ============================================================
# HEALTH DETAILS
# ============================================================

detail_cols = st.columns(4)

with detail_cols[0]:

    st.metric(
        "Data age",
        f"{data_age:.1f}s",
    )

with detail_cols[1]:

    st.metric(
        "Mean confidence",
        f"{mean_confidence:.2%}",
    )

with detail_cols[2]:

    st.metric(
        "Feature |z| > 3",
        f"{feature_z_percent:.2f}%",
    )

with detail_cols[3]:

    model_version = monitoring.get(
        "model_version",
        MODEL_VERSION_FALLBACK,
    )

    st.metric(
        "Model version",
        str(
            model_version
        ).replace(
            "_",
            " ",
        ),
    )


# ============================================================
# FOOTER
# ============================================================

latest_timestamp = None

if (
    not all_predictions.empty
    and "candle_time"
    in all_predictions.columns
):

    latest_timestamp = (
        all_predictions[
            "candle_time"
        ].max()
    )


if pd.notna(
    latest_timestamp
):

    latest_text = (
        pd.Timestamp(
            latest_timestamp
        ).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

else:

    latest_text = "Unavailable"


st.html(
    f"""
    <div class="footer">

        Latest prediction:
        {latest_text}

        &nbsp;•&nbsp;

        Model:
        {html.escape(
            str(
                monitoring.get(
                    "model_version",
                    MODEL_VERSION_FALLBACK,
                )
            )
        )}

        &nbsp;•&nbsp;

        FastAPI connected

    </div>
    """
)