"""Presentation-ready Streamlit dashboard for Pearls AQI Predictor."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import pandas as pd
import requests
import streamlit as st

from pearls_aqi.domain.aqi_categories import get_us_aqi_category, requires_health_alert
from pearls_aqi.features.store import load_training_data
from pearls_aqi.models.explain import (
    explain_prediction,
    global_feature_importance,
    shap_feature_importance,
    shap_local_explanation,
)
from pearls_aqi.models.registry import load_champion
from pearls_aqi.settings import settings


st.set_page_config(page_title="Pearls AQI Predictor", layout="wide")

st.markdown("""<style>
    .stApp {background:#f4f7fb; color:#102a43;}
    .block-container {max-width:1440px; padding-top:2.4rem; padding-bottom:3rem;}
    .hero {position:relative; overflow:hidden; padding:2.5rem 2.7rem; border-radius:22px; color:#fff; background:radial-gradient(circle at 85% 18%,rgba(98,217,204,.24),transparent 24%), repeating-linear-gradient(135deg,rgba(255,255,255,.035) 0 2px,transparent 2px 18px),linear-gradient(118deg,#102a43 0%,#1b5966 58%,#197c74 100%); margin-bottom:1.5rem; box-shadow:0 18px 42px rgba(16,42,67,.18);}
    .hero:after {content:""; position:absolute; right:5%; bottom:-28px; width:260px; height:130px; border:1px solid rgba(255,255,255,.22); border-radius:50% 50% 0 0; box-shadow:-68px 30px 0 -1px rgba(255,255,255,.10),68px 44px 0 -1px rgba(255,255,255,.08);}
    .hero h1 {position:relative; margin:0; font-size:2.55rem; font-weight:760; letter-spacing:-.045em;} .hero p {position:relative; margin:.55rem 0 0; max-width:620px; opacity:.82; font-size:1rem; font-weight:450;}
    .section-kicker {font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; color:#587085; font-weight:750; margin-bottom:.3rem;}
    .forecast-card {min-height:158px; background:var(--tint); border:1px solid color-mix(in srgb,var(--aqi) 28%,#fff); border-left:5px solid var(--aqi); border-radius:15px; padding:1.05rem 1.15rem; box-shadow:0 6px 20px rgba(16,42,67,.07);}
    .forecast-label {font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; color:#526477; font-weight:700;}.forecast-value {font-size:2.1rem; line-height:1.15; font-weight:760; color:#102a43; margin:.3rem 0;}.category-badge {display:inline-block; border-radius:999px; padding:.22rem .55rem; color:#fff; font-size:.72rem; font-weight:700; background:var(--aqi);}.forecast-trend {font-size:.78rem; color:#526477; margin-top:.55rem;}
    .aqi-card {background:#fff; border-radius:15px; padding:1.2rem 1.3rem; border:1px solid #e0e8f0; box-shadow:0 5px 18px rgba(16,42,67,.07);}.eyebrow {font-size:.73rem; text-transform:uppercase; letter-spacing:.1em; color:#627489; font-weight:700;}
    .notice {border-radius:12px; padding:.9rem 1.05rem; margin:.6rem 0 1rem; border-left:4px solid; font-size:.94rem;}.notice-stale {background:#e8f0f7; border-color:#56758f; color:#29475e;}.notice-alert {background:#fce9ea; border-color:#ba2633; color:#7c1821; font-weight:600;}
    [data-testid="stTabs"] [data-baseweb="tab-list"] {gap:.4rem; border-bottom:1px solid #d9e2ec;} [data-testid="stTabs"] button {height:42px; padding:0 1rem; color:#526477; font-weight:600;} [data-testid="stTabs"] button[aria-selected="true"] {color:#126b66; border-bottom:3px solid #126b66;}
    [data-testid="stDataFrame"] {border:1px solid #e0e8f0; border-radius:14px; overflow:hidden;}
    [data-testid="stExpander"] {background:#fff; border:1px solid #e0e8f0; border-radius:12px;}
    [data-testid="stChatMessage"] {background:#fff; border:1px solid #e0e8f0; border-radius:14px; padding:.5rem .8rem;}
    [data-testid="stTextInput"] input {border-radius:12px; border:1px solid #b7c8d7; background:#fff; min-height:48px; box-shadow:0 4px 14px rgba(16,42,67,.05);}
    button[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] button {background:#126b66 !important; color:#fff !important; border:0; border-radius:8px; font-weight:600; padding:.55rem 1.1rem;}
    button[data-testid="stBaseButton-primary"] *, [data-testid="stBaseButton-primary"] button * {color:#fff !important;}
    button[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primary"] button:hover {background:#0e5652 !important; color:#fff !important;}
    [data-testid="stFormSubmitButton"] button {background:#126b66 !important; color:#fff !important; border:0 !important; font-weight:700 !important;}
    [data-testid="stFormSubmitButton"] button p, [data-testid="stFormSubmitButton"] button span, [data-testid="stFormSubmitButton"] button div {color:#fff !important; font-weight:700 !important; opacity:1 !important;}
    [data-testid="stBaseButton-secondary"] {background:transparent; color:#526477; border:1px solid #d4e0e9; border-radius:10px; padding:.38rem .6rem; font-weight:500; text-align:left;}
    [data-testid="stBaseButton-secondary"]:hover {background:#e8f3f2; color:#0e5652; border-color:#8dc5bf;}
</style>""", unsafe_allow_html=True)


WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
]


@st.cache_data(ttl=120, show_spinner=False)
def prediction(city: str) -> dict:
    response = requests.get(f"{settings.API_BASE_URL}/predict/{city}", timeout=35)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def recent_history(city: str) -> pd.DataFrame:
    df = load_training_data(city).tail(72)
    return df[["event_time_utc", "aqi", "pm2_5_ug_m3", "pm10_ug_m3", "wind_speed_10m_kph"]].copy()


@st.cache_data(ttl=300, show_spinner=False)
def forecast_weather_features(city_slug: str, observed_at_iso: str) -> dict[str, float]:
    city = next(item for item in cities if item["slug"] == city_slug)
    observed_at = pd.Timestamp(observed_at_iso).to_pydatetime()

    response = requests.get(
        WEATHER_FORECAST_URL,
        params={
            "latitude": city["latitude"],
            "longitude": city["longitude"],
            "hourly": ",".join(FORECAST_WEATHER_VARIABLES),
            "forecast_days": 4,
            "timezone": "UTC",
        },
        timeout=12,
    )
    response.raise_for_status()

    hourly = response.json().get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []), utc=True)
    if times.empty:
        raise ValueError("Open-Meteo forecast response did not contain hourly timestamps.")

    features = {}
    for horizon in (24, 48, 72):
        target_time = observed_at + timedelta(hours=horizon)
        nearest_idx = int(abs(times - target_time).argmin())
        features[f"forecast_temperature_2m_c_{horizon}h"] = float(hourly["temperature_2m"][nearest_idx])
        features[f"forecast_relative_humidity_2m_pct_{horizon}h"] = float(hourly["relative_humidity_2m"][nearest_idx])
        features[f"forecast_surface_pressure_hpa_{horizon}h"] = float(hourly["surface_pressure"][nearest_idx])
        features[f"forecast_wind_speed_10m_kph_{horizon}h"] = float(hourly["wind_speed_10m"][nearest_idx])
        features[f"forecast_precipitation_mm_{horizon}h"] = float(hourly["precipitation"][nearest_idx])

    return features


def enrich_latest_for_analytics(city_slug: str, latest: pd.DataFrame) -> pd.DataFrame:
    enriched = latest.copy()
    observed_at = enriched.iloc[0]["event_time_utc"]
    if not isinstance(observed_at, str):
        observed_at = pd.Timestamp(observed_at).isoformat()

    for column, value in forecast_weather_features(city_slug, observed_at).items():
        enriched[column] = value

    return enriched


def trend_label(current: float, horizon: float) -> str:
    delta = horizon - current
    return "Worsening" if delta > 1 else "Improving" if delta < -1 else "Stable"


AQI_TONES = {
    "Good": ("#2E7D32", "#EAF6EC"),
    "Moderate": ("#B57D00", "#FFF6DB"),
    "Unhealthy for Sensitive Groups": ("#D86D00", "#FFF0E3"),
    "Unhealthy": ("#C7363F", "#FDEBED"),
    "Very Unhealthy": ("#7D3C98", "#F4EAF8"),
    "Hazardous": ("#6D1B2A", "#F8E7EA"),
}


def category_for(aqi: float) -> str:
    return get_us_aqi_category(float(aqi)).category


def forecast_card(label: str, aqi: float, current: float, category: str | None = None) -> str:
    category = category or category_for(aqi)
    color, tint = AQI_TONES[category]
    delta = aqi - current
    trend = "Current observation" if label == "Current" else f"{'↑ Worsening' if delta > 1 else '↓ Improving' if delta < -1 else '→ Stable'} {delta:+.1f} AQI vs. now"
    return f"<div class='forecast-card' style='--aqi:{color};--tint:{tint}'><div class='forecast-label'>{label}</div><div class='forecast-value'>{aqi:.0f}</div><span class='category-badge'>{category}</span><div class='forecast-trend'>{trend}</div></div>"


cities = [city for city in settings.load_cities_config()["cities"] if city.get("enabled", True)]
labels = {city["name"]: city["slug"] for city in cities}

st.markdown("""<div class="hero"><h1>Pearls AQI Predictor</h1><p>Three-day air-quality intelligence for Pakistan's major cities</p></div>""", unsafe_allow_html=True)

selected_page = st.radio(
    "Navigation",
    ["Overview", "City comparison", "Model analytics", "AQI Copilot"],
    horizontal=True,
    label_visibility="collapsed",
)

if selected_page == "Overview":
    st.markdown("<div class='section-kicker'>Forecast overview</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 3])
    with left:
        city_name = st.selectbox("Select city", list(labels), label_visibility="collapsed")
    city_slug = labels[city_name]

    try:
        with st.spinner("Preparing the latest city forecast..."):
            data = prediction(city_slug)
            history = recent_history(city_slug)
    except (requests.RequestException, ValueError) as exc:
        st.error(f"Forecast unavailable: {exc}")
        st.stop()

    forecasts = data["forecasts"]
    current = float(data["current_aqi"])
    category = category_for(current)

    if data["is_stale"]:
        st.markdown("<div class='notice notice-stale'><strong>Stored-data notice.</strong> Observations are older than 30 hours; run the feature pipeline before using this operationally.</div>", unsafe_allow_html=True)

    if any(requires_health_alert(item["aqi"]) for item in forecasts):
        st.markdown("<div class='notice notice-alert'><strong>Health alert.</strong> At least one forecast reaches Unhealthy or worse. Follow local public-health guidance.</div>", unsafe_allow_html=True)

    metric_cols = st.columns(4)
    metric_cols[0].markdown(forecast_card("Current", current, current, category), unsafe_allow_html=True)

    for column, item in zip(metric_cols[1:], forecasts):
        column.markdown(
            forecast_card(f"+{item['horizon_hours']} hours", item["aqi"], current, item["category"]),
            unsafe_allow_html=True,
        )

    color, _ = AQI_TONES[category]
    st.markdown(f"<div class='section-kicker' style='margin-top:1rem'>Selected city status <span style='color:{color}'>| {city_name}: {category}</span></div>", unsafe_allow_html=True)

    chart_col, insight_col = st.columns([2, 1])
    with chart_col:
        st.subheader("AQI history and forecast")
        historical = history.rename(columns={"event_time_utc": "time", "aqi": "AQI"})[["time", "AQI"]]
        forecast_frame = pd.DataFrame([{"time": item["valid_at_utc"], "AQI": item["aqi"]} for item in forecasts])
        chart_data = pd.concat(
            [
                historical,
                pd.DataFrame([{"time": data["latest_observation_at_utc"], "AQI": current}]),
                forecast_frame,
            ],
            ignore_index=True,
        )
        chart_data["time"] = pd.to_datetime(chart_data["time"], utc=True)
        st.line_chart(
            chart_data.drop_duplicates("time", keep="last").set_index("time"),
            color="#126b66",
            height=310,
        )

    with insight_col:
        st.subheader("Forecast signal")
        st.markdown(f"<div class='aqi-card'><div class='eyebrow'>72-hour direction</div><h2>{trend_label(current, forecasts[-1]['aqi'])}</h2><p>{forecasts[-1]['aqi'] - current:+.1f} AQI vs. now</p></div>", unsafe_allow_html=True)
        st.progress(min(int(current), 300) / 300, text=f"AQI intensity: {current:.0f} / 300")
        st.caption("Scale capped at 300 for display. Forecasts are model estimates, not health advice.")

    st.subheader("Current environmental snapshot")
    latest = history.iloc[-1]
    env = st.columns(4)
    env[0].metric("PM2.5", f"{latest.pm2_5_ug_m3:.1f} µg/m³")
    env[1].metric("PM10", f"{latest.pm10_ug_m3:.1f} µg/m³")
    env[2].metric("Wind", f"{latest.wind_speed_10m_kph:.1f} km/h")
    env[3].metric("Model", "Per-horizon champion")
    st.caption(f"Latest observation: {data['latest_observation_at_utc']} · Open-Meteo/CAMS modeled data · {data['model_name']} v{data['model_version']}")

@st.cache_data(ttl=120, show_spinner=False)
def comparison_predictions(city_pairs: tuple[tuple[str, str], ...]) -> tuple[list[dict], list[str]]:
    """Fetch city forecasts concurrently without overwhelming the API service."""
    rows: list[dict] = []
    unavailable: list[str] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(prediction, slug): name for name, slug in city_pairs}
        for future in as_completed(futures):
            name = futures[future]
            try:
                payload = future.result()
                values = [item["aqi"] for item in payload["forecasts"]]
                rows.append(
                    {
                        "City": name,
                        "Status": category_for(payload["current_aqi"]),
                        "Current": payload["current_aqi"],
                        "24h": values[0],
                        "48h": values[1],
                        "72h": values[2],
                        "Direction": trend_label(payload["current_aqi"], values[2]),
                    }
                )
            except (requests.RequestException, IndexError, KeyError, ValueError):
                unavailable.append(name)
    return rows, sorted(unavailable)


if selected_page == "City comparison":
    st.markdown("<div class='section-kicker'>Cross-city outlook</div>", unsafe_allow_html=True)
    with st.spinner("Loading city forecasts..."):
        rows, unavailable = comparison_predictions(tuple(labels.items()))

    if unavailable:
        st.warning("Forecasts are temporarily unavailable for: " + ", ".join(unavailable) + ".")

    if rows:
        comparison = pd.DataFrame(rows).sort_values("24h")

        def color_aqi(value):
            color, tint = AQI_TONES[category_for(value)]
            return f"background-color: {tint}; color: {color}; font-weight: 700"

        st.dataframe(
            comparison.style.map(color_aqi, subset=["Current", "24h", "48h", "72h"]),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Forecast comparison")
        st.bar_chart(comparison.set_index("City")[["24h", "48h", "72h"]], height=340)
    else:
        st.info("No city forecasts are currently available.")

if selected_page == "Model analytics":
    st.markdown("<div class='section-kicker'>Model quality and explanations</div>", unsafe_allow_html=True)
    analytics_city = st.selectbox("Analytics city", list(labels), key="analytics_city")
    analytics_slug = labels[analytics_city]

    try:
        model, metadata = load_champion(analytics_slug)
        training_data = load_training_data(analytics_slug).sort_values("event_time_utc")
        latest_features = enrich_latest_for_analytics(analytics_slug, training_data.iloc[[-1]])

        metrics = metadata["metrics"].get("per_horizon", metadata["metrics"])
        st.dataframe(
            pd.DataFrame(
                [
                    {"Horizon": target.replace("target_aqi_", "+"), **values}
                    for target, values in metrics.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        selections = metadata["metrics"].get("selection", {})
        if selections:
            st.caption(
                " · ".join(
                    f"{target.replace('target_aqi_', '+')}: {item['model']}"
                    for target, item in selections.items()
                )
            )

        horizon = st.selectbox("Explanation horizon", [24, 48, 72])

        if st.button("Generate SHAP importance"):
            try:
                importance = shap_feature_importance(model, training_data, horizon)
                method = "SHAP"
            except Exception:
                importance = global_feature_importance(model, training_data, horizon)
                method = "Permutation importance fallback"

            st.caption(f"Method: {method}")
            st.bar_chart(pd.DataFrame(importance[:12]).set_index("feature"))

        try:
            local = shap_local_explanation(model, latest_features, horizon)
        except Exception:
            local = explain_prediction(model, latest_features, horizon)

        st.caption(f"Latest local prediction (+{horizon}h): {local['prediction']:.1f} AQI")
        st.dataframe(
            pd.DataFrame(local["contributions"][:10]),
            use_container_width=True,
            hide_index=True,
        )

    except (FileNotFoundError, KeyError, ValueError, requests.RequestException) as exc:
        st.info(f"Model analytics unavailable: {exc}")

if selected_page == "AQI Copilot":
    if "copilot_history" not in st.session_state:
        st.session_state.copilot_history = []

    st.markdown("<div class='section-kicker'>Grounded assistant</div>", unsafe_allow_html=True)
    st.subheader("AQI Copilot")
    st.caption("Ask about current AQI, forecasts, pollutants, comparisons, or model explanations for supported cities.")

    def render_copilot_answer(payload):
        answer, marker, freshness = payload["answer"].partition("\n\nData freshness:")
        st.markdown(answer)
        if marker:
            st.warning(f"Data freshness:{freshness}")
        if settings.SHOW_COPILOT_DEBUG:
            st.caption(
                f"Source: {payload['provider']} | Tools: "
                f"{', '.join(payload['tools_used']) or 'No tools required'}"
            )

    for turn in st.session_state.copilot_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            render_copilot_answer(turn["payload"])
            if settings.SHOW_COPILOT_DEBUG:
                events = ", ".join(
                    f"{item['tool']} ({item['outcome']}, {item['latency_ms']} ms)"
                    for item in turn["payload"].get("tool_events", [])
                ) or "No tools required"
                st.caption(
                    f"Data generated: {turn['payload'].get('generated_at_utc', 'unavailable')} | "
                    f"{events} | Correlation ID: {turn['payload'].get('correlation_id', 'unavailable')}"
                )

    example_prompts = [
        "What is Lahore's AQI forecast for the next three days?",
        "Why is Lahore's AQI predicted to be high tomorrow?",
        "Compare Lahore, Karachi, and Islamabad tomorrow.",
        "Which supported city is expected to have the cleanest air tomorrow?",
        "Which city is forecast to improve the most over three days?",
    ]

    example_question = None
    with st.expander("Try an example question", expanded=False):
        for index, prompt in enumerate(example_prompts):
            if st.button(prompt, key=f"copilot_example_{index}", type="secondary", use_container_width=True):
                example_question = prompt

    if example_question:
        st.session_state.copilot_draft = example_question
        st.rerun()

    if "copilot_draft" not in st.session_state:
        st.session_state.copilot_draft = ""

    with st.form("copilot_message_form", clear_on_submit=True):
        st.text_input(
            "Ask the AQI Copilot",
            placeholder="Ask about AQI in Lahore, Karachi...",
            key="copilot_draft",
            label_visibility="collapsed",
        )
        ask_col, _ = st.columns([1, 7])
        with ask_col:
            ask_requested = st.form_submit_button("Ask Copilot", type="primary", use_container_width=True)

    question = st.session_state.copilot_draft.strip()

    if ask_requested and question:
        try:
            with st.spinner("Checking grounded forecast data..."):
                history = [turn["question"] for turn in st.session_state.copilot_history[-6:]]
                reply = requests.post(
                    f"{settings.API_BASE_URL}/api/v1/copilot/chat",
                    json={"message": question, "history": history},
                    timeout=60,
                )
                reply.raise_for_status()

            payload = reply.json()
            st.session_state.copilot_history.append({"question": question, "payload": payload})
            st.rerun()

        except requests.RequestException as exc:
            st.error(f"Copilot unavailable: {exc}")
