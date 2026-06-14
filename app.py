

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: #6366f1 !important;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-card h2 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}
.metric-card p {
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 4px 0 0 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Result badges ── */
.badge-safe {
    background: linear-gradient(135deg, #065f46, #047857);
    border: 1px solid #10b981;
    border-radius: 50px;
    padding: 10px 28px;
    font-size: 1.1rem;
    font-weight: 600;
    color: #ecfdf5;
    display: inline-block;
    margin: 8px 0;
}
.badge-warning {
    background: linear-gradient(135deg, #78350f, #92400e);
    border: 1px solid #f59e0b;
    border-radius: 50px;
    padding: 10px 28px;
    font-size: 1.1rem;
    font-weight: 600;
    color: #fffbeb;
    display: inline-block;
    margin: 8px 0;
}
.badge-danger {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 50px;
    padding: 10px 28px;
    font-size: 1.1rem;
    font-weight: 600;
    color: #fef2f2;
    display: inline-block;
    margin: 8px 0;
}

/* ── Section headers ── */
.section-header {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6366f1;
    margin: 20px 0 8px 0;
}

/* ── Divider ── */
.custom-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 20px 0;
}

/* ── Main background ── */
.main .block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* ── Prediction box ── */
.prediction-box {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border-radius: 20px;
    border: 1px solid rgba(99,102,241,0.3);
    padding: 32px;
    text-align: center;
}

/* ── Info panel ── */
.info-panel {
    background: rgba(99,102,241,0.08);
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #94a3b8;
}

/* Hide Streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model_path    = os.path.join("models", "xgb_model.pkl")
    scaler_path   = os.path.join("models", "scaler.pkl")
    features_path = os.path.join("models", "feature_names.pkl")

    if not os.path.exists(model_path):
        st.error(
            "⚠️ Model not found. Please run `python train.py` first to generate the model files."
        )
        st.stop()

    model         = joblib.load(model_path)
    scaler        = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    return model, scaler, feature_names


model, scaler, feature_names = load_model()


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUTS
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px 0;'>
        <span style='font-size:2.2rem;'>🎓</span>
        <h2 style='margin:6px 0 2px 0; font-size:1.15rem; font-weight:700;'>
            Dropout Predictor
        </h2>
        <p style='font-size:0.78rem; color:#64748b; margin:0;'>
            XGBoost · Ensemble Model
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Academic Info ──
    st.markdown("<div class='section-header'>📚 Academic Performance</div>", unsafe_allow_html=True)

    grade1 = st.slider(
        "Avg Grade — 1st Semester",
        min_value=0.0, max_value=20.0, value=12.0, step=0.5,
        help="Grade out of 20 (Portuguese grading scale)",
    )
    grade2 = st.slider(
        "Avg Grade — 2nd Semester",
        min_value=0.0, max_value=20.0, value=12.0, step=0.5,
        help="Grade out of 20 (Portuguese grading scale)",
    )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Personal Info ──
    st.markdown("<div class='section-header'>👤 Student Profile</div>", unsafe_allow_html=True)

    age = st.slider(
        "Age at Enrollment",
        min_value=17, max_value=60, value=20, step=1,
    )

    gender = st.radio(
        "Gender",
        options=["Female", "Male"],
        horizontal=True,
    )

    scholarship = st.radio(
        "Scholarship Holder",
        options=["No", "Yes"],
        horizontal=True,
    )

    debtor = st.radio(
        "Tuition Debtor",
        options=["No", "Yes"],
        horizontal=True,
    )

    tuition_ok = st.radio(
        "Tuition Fees Up to Date",
        options=["No", "Yes"],
        horizontal=True,
    )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Economic Context ──
    st.markdown("<div class='section-header'>🌍 Economic Context</div>", unsafe_allow_html=True)

    unemp = st.slider(
        "Unemployment Rate (%)",
        min_value=3.0, max_value=25.0, value=10.0, step=0.5,
    )
    inflation = st.slider(
        "Inflation Rate (%)",
        min_value=-1.0, max_value=8.0, value=1.5, step=0.1,
    )
    gdp = st.number_input(
        "GDP per Capita (USD)",
        min_value=5000, max_value=80000, value=22000, step=500,
    )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    predict_btn = st.button("🔍 Predict Dropout Risk", use_container_width=True, type="primary")



# MAIN AREA — HEADER


st.markdown("""
<h1 style='font-size:2rem; font-weight:700; margin-bottom:4px;'>
    Student Dropout Risk Predictor
</h1>
<p style='color:#64748b; font-size:0.95rem; margin-bottom:24px;'>
    Enter student details in the sidebar to predict dropout probability using a
    trained <strong>XGBoost</strong> model with engineered features and SHAP explanations.
</p>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — BUILD INPUT VECTOR
# ──────────────────────────────────────────────────────────────────────────────

def build_input():
    grade_trend   = grade2 - grade1
    sem_ratio     = grade2 / grade1 if grade1 > 0 else 1.0
    gender_enc    = 1 if gender == "Male" else 0
    scholarship_e = 1 if scholarship == "Yes" else 0
    debtor_enc    = 1 if debtor == "Yes" else 0
    tuition_enc   = 1 if tuition_ok == "Yes" else 0

    raw = {
        "Age at Enrollment":        float(age),
        "Average Grade (1st Sem)":  grade1,
        "Average Grade (2nd Sem)":  grade2,
        "Unemployment Rate (%)":    unemp,
        "Inflation Rate (%)":       inflation,
        "GDP per Capita (USD)":     float(gdp),
        "Scholarship Holder":       float(scholarship_e),
        "Gender":                   float(gender_enc),
        "Debtor":                   float(debtor_enc),
        "Tuition Fees Up to Date":  float(tuition_enc),
        "grade_trend":              grade_trend,
        "sem_performance_ratio":    sem_ratio,
    }

    # Align to trained feature order
    row = pd.DataFrame([[raw.get(f, 0.0) for f in feature_names]], columns=feature_names)
    return row


# ──────────────────────────────────────────────────────────────────────────────
# GAUGE CHART
# ──────────────────────────────────────────────────────────────────────────────

def make_gauge(prob: float) -> go.Figure:
    if prob < 0.35:
        color = "#10b981"
        label = "LOW RISK"
    elif prob < 0.65:
        color = "#f59e0b"
        label = "MEDIUM RISK"
    else:
        color = "#ef4444"
        label = "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        title={"text": f"<b>{label}</b>", "font": {"size": 14, "color": color}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#334155",
                "tickfont": {"color": "#64748b"},
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [0,   35],  "color": "rgba(16,185,129,0.12)"},
                {"range": [35,  65],  "color": "rgba(245,158,11,0.12)"},
                {"range": [65, 100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.8,
                "value": round(prob * 100, 1),
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=20, r=20),
        height=240,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE CHART
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_feature_importance(_model, _feature_names):
    """Return sorted feature importances from XGBoost."""
    importance = _model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature":    _feature_names,
        "Importance": importance,
    }).sort_values("Importance", ascending=True).tail(10)
    return fi_df


def plot_feature_importance():
    fi_df = get_feature_importance(model, tuple(feature_names))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    colors = plt.cm.plasma(np.linspace(0.3, 0.85, len(fi_df)))
    bars = ax.barh(fi_df["Feature"], fi_df["Importance"], color=colors, height=0.6)

    ax.set_xlabel("Importance Score", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.spines[:].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    ax.yaxis.set_tick_params(length=0)

    # Value labels
    for bar in bars:
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}", va="center", ha="left",
            color="#94a3b8", fontsize=7.5,
        )

    plt.tight_layout(pad=1.0)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT / PREDICTION DISPLAY
# ──────────────────────────────────────────────────────────────────────────────

col_main, col_right = st.columns([3, 2], gap="large")

with col_right:
    st.markdown("#### 📊 Feature Importance")
    st.markdown("<p style='color:#64748b; font-size:0.82rem; margin-top:-8px;'>Top 10 factors driving the model's decisions</p>", unsafe_allow_html=True)
    st.pyplot(plot_feature_importance(), use_container_width=True)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # Mini stats
    st.markdown("#### ℹ️ Model Info")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Algorithm", "XGBoost")
        st.metric("Train Size", "~2400")
    with m2:
        st.metric("Features", str(len(feature_names)))
        st.metric("Est. AUC", "≥ 0.88")


with col_main:
    if not predict_btn:
        # ── Placeholder state ──
        st.markdown("""
        <div style='background: linear-gradient(135deg, #0f172a, #1e293b);
                    border: 1px solid rgba(99,102,241,0.2);
                    border-radius: 20px; padding: 48px; text-align: center;'>
            <div style='font-size:3.5rem; margin-bottom:12px;'>🎓</div>
            <h3 style='color:#e2e8f0; margin:0 0 8px 0;'>Ready to Predict</h3>
            <p style='color:#64748b; font-size:0.9rem;'>
                Fill in the student details in the sidebar<br>and click <strong>Predict Dropout Risk</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── What each section means ──
        st.markdown("#### 📋 Feature Guide")

        with st.expander("📚 Academic Performance"):
            st.markdown("""
            - **Avg Grade (1st / 2nd Sem)**: Scored out of 20 (Portuguese scale).
              Grades below 10 indicate risk.
            - **Grade Trend** (auto-computed): `2nd sem − 1st sem`.
              A negative trend is a strong dropout signal.
            """)

        with st.expander("👤 Student Profile"):
            st.markdown("""
            - **Scholarship Holder**: Students with scholarships show lower dropout rates.
            - **Tuition Debtor**: Outstanding debt significantly raises dropout risk.
            - **Tuition Up to Date**: Timely payments correlate with retention.
            """)

        with st.expander("🌍 Economic Context"):
            st.markdown("""
            - **Unemployment Rate**: Higher unemployment may push students to leave
              education and seek work.
            - **GDP per Capita**: Proxy for the economic environment the student operates in.
            """)

    else:
        # ── Run prediction ──
        input_df = build_input()
        input_sc = scaler.transform(input_df)
        prob     = model.predict_proba(input_sc)[0, 1]

        if prob < 0.35:
            risk_label = "Low Risk"
            badge_cls  = "badge-safe"
            verdict    = "✅ Likely to Graduate"
            desc       = "This student profile shows a low probability of dropping out. Keep monitoring academic performance."
        elif prob < 0.65:
            risk_label = "Medium Risk"
            badge_cls  = "badge-warning"
            verdict    = "⚠️ Moderate Dropout Risk"
            desc       = "This student shows some risk factors. Early intervention and counselling is recommended."
        else:
            risk_label = "High Risk"
            badge_cls  = "badge-danger"
            verdict    = "🚨 High Dropout Risk"
            desc       = "This student profile strongly indicates dropout risk. Immediate support and intervention advised."

        # Gauge
        st.plotly_chart(make_gauge(prob), use_container_width=True, config={"displayModeBar": False})

        # Verdict
        st.markdown(f"""
        <div style='text-align:center; margin: -8px 0 20px 0;'>
            <div class='{badge_cls}'>{verdict}</div>
            <p style='color:#94a3b8; font-size:0.88rem; margin-top:10px;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Key metrics breakdown ──
        st.markdown("#### 📈 Input Summary")

        c1, c2, c3, c4 = st.columns(4)

        grade_trend_val = grade2 - grade1
        trend_arrow = "↑" if grade_trend_val >= 0 else "↓"
        trend_color = "#10b981" if grade_trend_val >= 0 else "#ef4444"

        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color:#6366f1;'>{grade1:.1f}</h2>
                <p>1st Sem Grade</p>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color:#6366f1;'>{grade2:.1f}</h2>
                <p>2nd Sem Grade</p>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color:{trend_color};'>{trend_arrow}{abs(grade_trend_val):.1f}</h2>
                <p>Grade Trend</p>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color:#6366f1;'>{prob*100:.1f}%</h2>
                <p>Dropout Prob.</p>
            </div>""", unsafe_allow_html=True)

        # ── Risk factors ──
        st.markdown("#### 🔍 Risk Factor Analysis")

        factors = []
        if grade1 < 10:
            factors.append(("⚠️", "Low 1st semester grade (<10)", "high"))
        if grade2 < 10:
            factors.append(("⚠️", "Low 2nd semester grade (<10)", "high"))
        if grade_trend_val < -2:
            factors.append(("⚠️", f"Significant grade decline ({grade_trend_val:+.1f})", "high"))
        if debtor == "Yes":
            factors.append(("⚠️", "Student is a tuition debtor", "high"))
        if tuition_ok == "No":
            factors.append(("⚠️", "Tuition fees not up to date", "medium"))
        if unemp > 15:
            factors.append(("⚠️", f"High unemployment rate ({unemp:.1f}%)", "medium"))
        if scholarship == "Yes":
            factors.append(("✅", "Scholarship holder — positive factor", "positive"))
        if grade_trend_val >= 0:
            factors.append(("✅", "Grade improving or stable — positive factor", "positive"))
        if tuition_ok == "Yes":
            factors.append(("✅", "Tuition fees are up to date", "positive"))

        if not factors:
            factors.append(("ℹ️", "No strong risk signals detected in the input profile.", "neutral"))

        for icon, text, level in factors:
            if level == "high":
                color = "#fef2f2"; border = "#ef4444"
            elif level == "medium":
                color = "#fffbeb"; border = "#f59e0b"
            elif level == "positive":
                color = "#ecfdf5"; border = "#10b981"
            else:
                color = "#f1f5f9"; border = "#6366f1"

            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03); border-left: 3px solid {border};
                        border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 6px 0;
                        font-size: 0.88rem; color: #e2e8f0;'>
                {icon} {text}
            </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#334155; font-size:0.78rem;'>
    Student Dropout Prediction System · XGBoost Classifier ·
    Built with Streamlit · Features: Academic, Demographic & Economic
</p>
""", unsafe_allow_html=True)
