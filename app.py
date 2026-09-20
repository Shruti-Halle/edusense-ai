"""EduSense AI — Streamlit early-warning and study coach."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_utils import FEATURE_COLUMNS, FEATURE_LABELS, generate_synthetic_data, load_student_data
from fairness import build_fairness_table
from granite_client import build_prompt, generate_study_plan
from model import train_models


st.set_page_config(
    page_title="EduSense AI",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLORS = {
    "navy": "#17253d",
    "navy_2": "#243957",
    "coral": "#f27b4a",
    "mint": "#d9f0e8",
    "gold": "#f3d58b",
    "ink": "#1e2b3f",
    "muted": "#67748a",
    "paper": "#faf9f6",
    "line": "#e5e7eb",
    "risk": "#d95d56",
    "safe": "#2d806f",
    "purple": "#8e82c9",
}


def inject_styles() -> None:
    """Add the custom HTML/CSS layer that makes the Streamlit UI feel designed."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --navy: #17253d;
            --navy-2: #243957;
            --coral: #f27b4a;
            --mint: #d9f0e8;
            --gold: #f3d58b;
            --ink: #1e2b3f;
            --muted: #67748a;
            --paper: #faf9f6;
            --line: #e5e7eb;
            --shadow: 0 18px 55px rgba(23, 37, 61, .08);
        }

        html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stAppViewContainer"] { background:
            radial-gradient(circle at 92% 2%, rgba(242,123,74,.08), transparent 24rem),
            var(--paper); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { visibility: hidden; height: 0; }
        section[data-testid="stSidebar"] { background: var(--navy); border-right: 0; }
        section[data-testid="stSidebar"] > div { padding-top: 2rem; }
        section[data-testid="stSidebar"] * { color: #f7f6f2; }
        section[data-testid="stSidebar"] .stRadio label { color: rgba(247,246,242,.72); font-size: .84rem; }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap: .36rem; }
        section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
            background: var(--coral); border-radius: 12px; color: white; font-weight: 700;
        }
        section[data-testid="stSidebar"] .stRadio label { padding: .65rem .75rem; }
        section[data-testid="stSidebar"] .stFileUploader { background: rgba(255,255,255,.06); border-radius: 12px; padding: .4rem; }
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { border: 1px dashed rgba(255,255,255,.24); }
        .block-container { max-width: 1400px; padding: 2rem 3.2rem 4rem; }
        .brand { display: flex; align-items: center; gap: .75rem; margin: 0 0 2.5rem; }
        .brand-mark { width: 2.6rem; height: 2.6rem; border-radius: 12px; background: var(--coral); display: grid; place-items: center; color: white; font-family: 'Space Grotesk'; font-weight: 700; font-size: 1.25rem; }
        .brand-name { font-family: 'Space Grotesk'; font-size: 1.22rem; font-weight: 700; letter-spacing: -.04em; }
        .brand-caption { color: rgba(247,246,242,.5); font-size: .64rem; letter-spacing: .1em; text-transform: uppercase; margin-top: .12rem; }
        .sidebar-kicker { color: rgba(247,246,242,.42); font-family: 'DM Mono'; font-size: .62rem; letter-spacing: .13em; text-transform: uppercase; margin: 1.5rem 0 .6rem; }
        .sidebar-note { color: rgba(247,246,242,.6); font-size: .71rem; line-height: 1.55; margin-top: 1.7rem; padding: .85rem; border: 1px solid rgba(255,255,255,.11); border-radius: 12px; }
        .eyebrow { color: var(--coral); font-family: 'DM Mono'; font-size: .7rem; letter-spacing: .14em; text-transform: uppercase; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink); letter-spacing: -.06em; }
        h1 { font-size: clamp(2.5rem, 5vw, 4.8rem) !important; line-height: .98 !important; margin: .45rem 0 1rem !important; }
        h2 { font-size: 2rem !important; }
        h3 { font-size: 1.05rem !important; letter-spacing: -.04em; }
        .lede { max-width: 610px; color: var(--muted); font-size: 1rem; line-height: 1.7; }
        .hero { background: linear-gradient(135deg, #17253d 0%, #29425e 72%, #31516a 100%); border-radius: 24px; padding: clamp(1.6rem, 4vw, 3.5rem); position: relative; overflow: hidden; color: white; box-shadow: 0 18px 60px rgba(23,37,61,.18); }
        .hero:after { content: ''; position: absolute; width: 18rem; height: 18rem; right: -5rem; top: -7rem; border: 1px solid rgba(255,255,255,.16); border-radius: 50%; box-shadow: 0 0 0 2.8rem rgba(255,255,255,.03), 0 0 0 5.6rem rgba(255,255,255,.025); }
        .hero h1 { color: white !important; max-width: 710px; }
        .hero .eyebrow { color: #f8c4a7; }
        .hero .lede { color: rgba(255,255,255,.72); }
        .hero-art { position: absolute; right: 6%; bottom: 0; display: flex; gap: .55rem; align-items: end; opacity: .85; }
        .hero-art span { display: block; width: 1.3rem; border-radius: 9px 9px 0 0; background: var(--coral); }
        .hero-art span:nth-child(1) { height: 3.5rem; opacity: .55; }
        .hero-art span:nth-child(2) { height: 5.8rem; opacity: .7; }
        .hero-art span:nth-child(3) { height: 8.5rem; }
        .hero-art span:nth-child(4) { height: 11rem; background: var(--mint); }
        .card { background: rgba(255,255,255,.84); border: 1px solid rgba(229,231,235,.9); border-radius: 18px; box-shadow: var(--shadow); }
        .metric-card { padding: 1.2rem 1.3rem; min-height: 8.1rem; position: relative; overflow: hidden; }
        .metric-card:after { content: ''; position: absolute; width: 5rem; height: 5rem; right: -1.6rem; bottom: -1.8rem; border-radius: 50%; background: var(--mint); opacity: .7; }
        .metric-card:nth-child(1):after { background: #fae2d5; }
        .metric-card:nth-child(3):after { background: var(--gold); opacity: .45; }
        .metric-card:nth-child(4):after { background: #e3dff6; }
        .metric-label { color: var(--muted); font-size: .72rem; }
        .metric-value { color: var(--ink); font-family: 'Space Grotesk'; font-size: 2.2rem; font-weight: 700; letter-spacing: -.08em; margin-top: .7rem; }
        .metric-detail { color: var(--muted); font-size: .68rem; }
        .trend { color: var(--safe); font-weight: 700; }
        .section-card { padding: 1.35rem; }
        .section-title { color: var(--ink); font-family: 'Space Grotesk'; font-size: 1.05rem; font-weight: 700; letter-spacing: -.04em; }
        .section-copy { color: var(--muted); font-size: .72rem; margin-top: .25rem; }
        .plain-callout { background: #edf5f2; border-left: 4px solid var(--safe); border-radius: 0 12px 12px 0; padding: .85rem 1rem; color: var(--ink); font-size: .79rem; line-height: 1.55; }
        .plain-callout strong { display: block; color: var(--safe); font-size: .65rem; font-family: 'DM Mono'; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .25rem; }
        .risk-pill { display: inline-flex; align-items: center; border-radius: 999px; padding: .26rem .62rem; font-size: .63rem; font-weight: 700; }
        .risk-high { background: #fae0dd; color: #a84842; }
        .risk-watch { background: #fff2ce; color: #8e6a15; }
        .risk-safe { background: #d9f0e8; color: #27705f; }
        .info-panel { background: var(--navy); border-radius: 18px; padding: 1.25rem; color: white; }
        .info-panel h3 { color: white !important; margin-top: 0 !important; }
        .info-panel p { color: rgba(255,255,255,.7); font-size: .78rem; line-height: 1.6; }
        .profile-chip { display: inline-flex; background: var(--mint); color: #27705f; border-radius: 999px; padding: .35rem .7rem; font-size: .7rem; font-weight: 700; margin: .25rem .25rem 0 0; }
        .stButton > button, .stDownloadButton > button { border-radius: 10px; border: 0; background: var(--coral); color: white; font-weight: 700; padding: .65rem 1rem; box-shadow: 0 7px 18px rgba(242,123,74,.2); }
        .stButton > button:hover, .stDownloadButton > button:hover { background: #e86635; color: white; border: 0; }
        .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div { border-radius: 10px; border-color: var(--line); }
        .stTabs [data-baseweb="tab-list"] { gap: .35rem; }
        .stTabs [data-baseweb="tab"] { background: #eef1f4; border-radius: 999px; padding: .4rem .8rem; height: auto; color: var(--muted); }
        .stTabs [aria-selected="true"] { background: var(--navy) !important; color: white !important; }
        .dataframe { border-radius: 12px; overflow: hidden; }
        [data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 14px; padding: 1rem; }
        [data-testid="stMetricValue"] { font-family: 'Space Grotesk'; color: var(--ink); }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        .footer-note { color: var(--muted); font-size: .68rem; text-align: center; margin-top: 3rem; }
        @media (max-width: 850px) {
            .block-container { padding: 1.2rem 1rem 3rem; }
            .hero-art { display: none; }
            h1 { font-size: 3rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card_metric(label: str, value: str, detail: str, trend: str) -> None:
    st.markdown(
        f"""<div class="card metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail"><span class="trend">{trend}</span> {detail}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def page_intro(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""<div style="margin: .6rem 0 1.5rem">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <div class="lede">{description}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def plain_callout(text: str) -> None:
    st.markdown(f'<div class="plain-callout"><strong>In plain English</strong>{text}</div>', unsafe_allow_html=True)


def risk_pill(label: str) -> str:
    color = {"High": "risk-high", "Medium": "risk-watch", "Low": "risk-safe"}.get(label, "risk-watch")
    return f'<span class="risk-pill {color}">{label} risk</span>'


def make_chart_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans", "size": 11, "color": COLORS["muted"]},
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    fig.update_xaxes(showgrid=False, linecolor=COLORS["line"])
    fig.update_yaxes(showgrid=True, gridcolor="#edf0f2", zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def cached_data(uploaded_bytes: bytes | None, filename: str | None) -> pd.DataFrame:
    if uploaded_bytes is not None:
        return load_student_data(io.BytesIO(uploaded_bytes), filename or "student-data.csv")
    return generate_synthetic_data()


@st.cache_resource(show_spinner=False)
def cached_models(data: pd.DataFrame) -> dict:
    return train_models(data)


def sidebar() -> tuple[str, pd.DataFrame, str]:
    st.sidebar.markdown(
        """<div class="brand">
            <div class="brand-mark">e.</div>
            <div><div class="brand-name">EduSense</div><div class="brand-caption">teaching companion</div></div>
        </div>
        <div class="sidebar-kicker">Workspace</div>""",
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Explore data", "Check a student", "Study plan", "Fair & safe AI"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown('<div class="sidebar-kicker">Your data</div>', unsafe_allow_html=True)
    upload = st.sidebar.file_uploader("Upload student-mat.csv", type=["csv", "txt"], help="Semicolon-separated UCI files are detected automatically.")
    source = "Uploaded file" if upload else "Synthetic demo data"
    try:
        data = cached_data(upload.getvalue() if upload else None, upload.name if upload else None)
    except ValueError as exc:
        st.sidebar.error(f"Could not use this upload: {exc}")
        st.stop()
    st.sidebar.markdown(
        f"""<div class="sidebar-note"><b>{source}</b><br>{len(data):,} students in view.<br>Nothing is sent anywhere or saved.</div>""",
        unsafe_allow_html=True,
    )
    return page, data, source


def home_page(data: pd.DataFrame, source: str, bundle: dict) -> None:
    page_intro("A calmer way to notice", "Make the next step feel possible.", "EduSense turns everyday learning signals into supportive next moves. It helps teachers spot patterns early without turning a student into a label.")
    st.markdown(
        """<div class="hero">
            <div class="eyebrow">Early support for real classrooms</div>
            <h1>Spot students who need help before they fall behind.</h1>
            <div class="lede">Start with the signal, add your human context, and leave with a small plan that can fit into a real week.</div>
            <div class="hero-art"><span></span><span></span><span></span><span></span></div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.write("")
    columns = st.columns(4)
    at_risk = int(data["at_risk"].sum())
    with columns[0]:
        card_metric("Students analysed", f"{len(data):,}", "in the current view", "+12")
    with columns[1]:
        card_metric("Flagged for support", f"{at_risk}", f"{at_risk / len(data):.0%} of students", "early signal")
    with columns[2]:
        card_metric("Best model F1", f"{bundle['metrics'][bundle['best_model_name']]['f1']:.2f}", "balanced model quality", "validated")
    with columns[3]:
        card_metric("Data source", "Demo", "safe, local sample", "private")

    st.write("")
    st.markdown('<div class="section-title">Try it in 3 easy steps</div><div class="section-copy">A useful flow for a busy morning.</div>', unsafe_allow_html=True)
    steps = st.columns(3)
    step_content = [
        ("01", "Explore the data", "See class patterns and understand what changed."),
        ("02", "Check a student", "Combine model signals with what you already know."),
        ("03", "Make a study plan", "Turn one conversation into a small next step."),
    ]
    for col, (number, title, body) in zip(steps, step_content):
        with col:
            st.markdown(f'<div class="card section-card" style="margin-top:.8rem"><div class="eyebrow">{number}</div><h3>{title}</h3><div class="section-copy">{body}</div></div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.2, .8])
    with left:
        st.markdown('<div class="card section-card"><div class="section-title">Why it matters</div><div class="section-copy">A risk level is a nudge, not a label.</div><br><div class="plain-callout"><strong>SDG 4 · Quality education</strong>Earlier support can make learning more equitable. EduSense keeps the teacher, tutor, or parent in the loop and leaves the decision with them.</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="info-panel"><div class="eyebrow" style="color:#f8c4a7">Built for trust</div><h3>No names collected. Nothing saved.</h3><p>This demo runs on a local synthetic dataset. Uploads stay in the current session and the model is explainable enough to question.</p></div>', unsafe_allow_html=True)


def explore_page(data: pd.DataFrame) -> None:
    page_intro("Explore data", "See the shape of learning.", "Move between class patterns and the people behind them. Every signal is a prompt for curiosity, never a student label.")
    with st.expander("View raw data and data quality"):
        st.write(f"**Shape:** {data.shape[0]:,} rows × {data.shape[1]} columns")
        st.write(f"**Missing values:** {int(data.isna().sum().sum())}")
        st.dataframe(data.head(20), use_container_width=True, hide_index=True)

    at_risk_label = data["at_risk"].map({0: "Not at risk", 1: "At risk"})
    count_data = at_risk_label.value_counts().rename_axis("Status").reset_index(name="Students")
    fig1 = px.bar(count_data, x="Status", y="Students", color="Status", color_discrete_map={"At risk": COLORS["risk"], "Not at risk": COLORS["navy_2"]})
    st.plotly_chart(make_chart_layout(fig1), use_container_width=True)
    plain_callout(f"{int(data['at_risk'].sum())} students are below the early-support threshold in this view. That is {data['at_risk'].mean():.0%} of the class, so the first move should be a conversation, not a conclusion.")

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.histogram(data, x="G3", nbins=11, color=at_risk_label, color_discrete_map={"At risk": COLORS["risk"], "Not at risk": COLORS["navy_2"]}, labels={"G3": "Final grade"})
        st.plotly_chart(make_chart_layout(fig2), use_container_width=True)
        plain_callout(f"The typical final grade is {data['G3'].median():.0f}. Grades under 10 are the group to understand sooner, while the full spread shows where support can be targeted.")
    with col2:
        fig3 = px.scatter(data.sample(min(len(data), 260), random_state=42), x="absences", y="G3", color=at_risk_label.sample(min(len(data), 260), random_state=42), color_discrete_map={"At risk": COLORS["risk"], "Not at risk": COLORS["navy_2"]}, opacity=.72, labels={"absences": "Absences", "G3": "Final grade"})
        st.plotly_chart(make_chart_layout(fig3), use_container_width=True)
        corr = data[["absences", "G3"]].corr().iloc[0, 1]
        plain_callout(f"Absences and final grade have a {corr:+.2f} relationship in this sample. It is a useful clue, but attendance alone should never decide a student's story.")

    col3, col4 = st.columns(2)
    with col3:
        fig4 = px.box(data, x="studytime", y="G3", color=at_risk_label, color_discrete_map={"At risk": COLORS["risk"], "Not at risk": COLORS["navy_2"]}, labels={"studytime": "Study time band", "G3": "Final grade"})
        st.plotly_chart(make_chart_layout(fig4), use_container_width=True)
        plain_callout("Study time bands run from 1 (less than 2 hours) to 4 (more than 10 hours). More time can help, but it works best when paired with the right kind of practice.")
    with col4:
        corr_cols = ["G1", "G2", "G3", "absences", "failures", "studytime"]
        corr_matrix = data[corr_cols].corr()
        fig5 = px.imshow(corr_matrix, color_continuous_scale=["#f7eee8", "#f27b4a", "#17253d"], aspect="auto", labels={"color": "Relationship"})
        fig5.update_layout(height=340)
        st.plotly_chart(make_chart_layout(fig5), use_container_width=True)
        plain_callout("The earlier grades G1 and G2 usually carry the clearest relationship with the final grade. That is why early check-ins are more useful than waiting for the final result.")

    insight_cols = st.columns(4)
    insights = [
        ("Class average", f"{data['G3'].mean():.1f}", "final grade"),
        ("Median absences", f"{data['absences'].median():.0f}", "days"),
        ("Early support", f"{data['at_risk'].mean():.0%}", "of students"),
        ("Most common", str(data["studytime"].mode().iloc[0]), "study band"),
    ]
    for col, (label, value, detail) in zip(insight_cols, insights):
        with col:
            st.markdown(f'<div class="card section-card"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:1.75rem">{value}</div><div class="metric-detail">{detail}</div></div>', unsafe_allow_html=True)


def default_profile() -> dict:
    return {
        "studytime": 2, "failures": 1, "absences": 8, "G1": 10, "G2": 11,
        "health": 3, "freetime": 3, "goout": 3, "Medu": 2, "Fedu": 2,
        "internet": "yes", "schoolsup": "no", "famsup": "yes", "higher": "yes",
        "age": 16, "sex": "F",
    }


def check_page(data: pd.DataFrame, bundle: dict) -> None:
    page_intro("Check a student", "Start with a question, not a conclusion.", "Choose a profile, add the human context, and let the model surface a few caring next steps. Sex is collected for fairness review only, never used in the model.")
    example = st.radio("Start with an example", ["Typical student", "Needs support", "Doing well"], horizontal=True)
    presets = {
        "Typical student": default_profile(),
        "Needs support": {**default_profile(), "failures": 2, "absences": 18, "G1": 7, "G2": 8, "studytime": 1},
        "Doing well": {**default_profile(), "failures": 0, "absences": 2, "G1": 15, "G2": 16, "studytime": 3},
    }
    profile = presets[example]
    with st.form("student_check_form"):
        left, mid, right = st.columns(3)
        with left:
            studytime = st.slider("Study time band", 1, 4, profile["studytime"], help="1 = less than 2 hours, 4 = more than 10 hours")
            failures = st.number_input("Past class failures", 0, 4, profile["failures"])
            absences = st.number_input("Absences", 0, 80, profile["absences"])
            G1 = st.number_input("First period grade (G1)", 0, 20, profile["G1"])
            G2 = st.number_input("Second period grade (G2)", 0, 20, profile["G2"])
        with mid:
            health = st.slider("Health", 1, 5, profile["health"])
            freetime = st.slider("Free time", 1, 5, profile["freetime"])
            goout = st.slider("Going out", 1, 5, profile["goout"])
            medu = st.slider("Mother's education", 0, 4, profile["Medu"])
            fedu = st.slider("Father's education", 0, 4, profile["Fedu"])
        with right:
            internet = st.selectbox("Internet at home", ["yes", "no"], index=0 if profile["internet"] == "yes" else 1)
            schoolsup = st.selectbox("School support", ["yes", "no"], index=0 if profile["schoolsup"] == "yes" else 1)
            famsup = st.selectbox("Family support", ["yes", "no"], index=0 if profile["famsup"] == "yes" else 1)
            higher = st.selectbox("Plans for higher education", ["yes", "no"], index=0 if profile["higher"] == "yes" else 1)
            age = st.number_input("Age", 14, 22, profile["age"])
            sex = st.selectbox("Sex (fairness review only)", ["F", "M"], index=0 if profile["sex"] == "F" else 1)
        submitted = st.form_submit_button("Check risk level")

    if submitted:
        input_profile = {"studytime": studytime, "failures": failures, "absences": absences, "G1": G1, "G2": G2, "health": health, "freetime": freetime, "goout": goout, "Medu": medu, "Fedu": fedu, "internet": internet, "schoolsup": schoolsup, "famsup": famsup, "higher": higher, "age": age, "sex": sex}
        features = pd.DataFrame([{key: input_profile[key] for key in FEATURE_COLUMNS}])
        probability = float(bundle["best_model"].predict_proba(features)[:, 1][0])
        risk = "High" if probability >= .62 else "Medium" if probability >= .35 else "Low"
        importances = bundle["feature_importances"]
        reasons = sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        st.session_state["prediction"] = {"profile": input_profile, "probability": probability, "risk": risk, "reasons": reasons}

    prediction = st.session_state.get("prediction")
    if not prediction:
        st.info("Choose an example or adjust the inputs, then click Check risk level.")
        return

    left, right = st.columns([0.9, 1.1])
    with left:
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=prediction["probability"] * 100, number={"suffix": "%", "font": {"family": "Space Grotesk", "size": 46, "color": COLORS["ink"]}}, title={"text": "Chance of falling behind", "font": {"family": "Plus Jakarta Sans", "size": 13, "color": COLORS["muted"]}}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS["risk"] if prediction["risk"] == "High" else COLORS["coral"]}, "steps": [{"range": [0, 35], "color": "#d9f0e8"}, {"range": [35, 62], "color": "#fff2ce"}, {"range": [62, 100], "color": "#fae0dd"}]}))
        gauge.update_layout(height=300, margin={"l": 25, "r": 25, "t": 45, "b": 0}, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True)
    with right:
        st.markdown(f'<div class="card section-card"><div class="eyebrow">What this means</div><h2 style="margin-top:.5rem">{risk_pill(prediction["risk"])}</h2><p style="color:{COLORS["muted"]};line-height:1.7">This is a pattern to explore, not a prediction. Start with what you know about the person in front of you.</p><div class="section-title">Main reasons to talk about</div>', unsafe_allow_html=True)
        for feature, value in prediction["reasons"]:
            st.markdown(f'<div class="profile-chip">{FEATURE_LABELS.get(feature, feature)} · {value:+.2f}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Build a small study plan", type="primary"):
            st.session_state["page_override"] = "Study plan"
            st.rerun()


def plan_page() -> None:
    page_intro("Study plan", "A plan with room to breathe.", "Small, specific practice that fits a real week. Adapt it as the student’s confidence and capacity change.")
    prediction = st.session_state.get("prediction")
    if not prediction:
        st.warning("Check a student first. EduSense will use the risk pattern and the top reasons to shape the plan.")
        return
    factors = [FEATURE_LABELS.get(name, name) for name, _ in prediction["reasons"]]
    prompt = build_prompt(prediction["profile"], factors)
    plan = generate_study_plan(prediction["profile"], factors, prompt)
    left, right = st.columns([.85, 1.15])
    with left:
        st.markdown('<div class="card section-card"><div class="section-title">Input</div><div class="section-copy">The plan is grounded in this check.</div><br>', unsafe_allow_html=True)
        st.markdown(risk_pill(prediction["risk"]), unsafe_allow_html=True)
        st.metric("Risk pattern", f"{prediction['probability']:.0%}")
        st.markdown("**Top factors**")
        for factor in factors:
            st.markdown(f"- {factor}")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Generated output</div><div class="section-copy">Generated with AI · demo placeholder</div><br>', unsafe_allow_html=True)
        st.markdown(plan)
        st.download_button("Download plan (.txt)", data=plan, file_name="edusense-study-plan.txt", mime="text/plain")
        st.markdown("</div>", unsafe_allow_html=True)
    with st.expander("Show exactly what the AI is asked"):
        st.code(prompt, language="text")


def safe_page(data: pd.DataFrame, bundle: dict) -> None:
    page_intro("Fair & safe AI", "The model stays humble.", "EduSense is designed to make care more specific, not to make decisions about young people. Here is what that means in practice.")
    table = build_fairness_table(data, bundle["best_model"])
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card section-card"><div class="section-title">Accuracy by group</div><div class="section-copy">How often the model gets the overall class signal right.</div>', unsafe_allow_html=True)
        fig = px.bar(table, x="group", y="accuracy", color="attribute", barmode="group", color_discrete_sequence=[COLORS["coral"], COLORS["navy_2"]])
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(make_chart_layout(fig), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card section-card"><div class="section-title">Recall by group</div><div class="section-copy">How often the model notices students who truly need support.</div>', unsafe_allow_html=True)
        fig2 = px.bar(table, x="group", y="recall", color="attribute", barmode="group", color_discrete_sequence=[COLORS["safe"], COLORS["purple"]])
        fig2.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(make_chart_layout(fig2), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.dataframe(table.style.format({"accuracy": "{:.0%}", "recall": "{:.0%}"}), use_container_width=True, hide_index=True)
    plain_callout("A fairness check is a regular habit, not a one-time approval. Gaps can come from the data, the threshold, or missing context, so every model result still needs a human review.")
    cols = st.columns(3)
    for col, title, body in zip(cols, ["How the score works", "What shapes it", "Privacy by design"], ["Two simple models are compared by F1, accuracy, recall, and precision.", "Grades, attendance, study habits, and support signals shape the demo score. Sex is excluded from training.", "The demo uses local data. No names are collected and nothing is saved."]):
        with col:
            st.markdown(f'<div class="card section-card"><div class="eyebrow">Principle</div><h3>{title}</h3><div class="section-copy" style="line-height:1.6">{body}</div></div>', unsafe_allow_html=True)


def main() -> None:
    inject_styles()
    page, data, source = sidebar()
    try:
        bundle = cached_models(data)
    except ValueError as exc:
        st.error(f"Could not train the models on this file: {exc}")
        st.stop()

    if st.session_state.get("page_override"):
        page = st.session_state.pop("page_override")
    if page == "Home":
        home_page(data, source, bundle)
    elif page == "Explore data":
        explore_page(data)
    elif page == "Check a student":
        check_page(data, bundle)
    elif page == "Study plan":
        plan_page()
    else:
        safe_page(data, bundle)
    st.markdown('<div class="footer-note">EduSense AI · A supportive demo for SDG 4 quality education · Model scores are prompts, not verdicts.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
