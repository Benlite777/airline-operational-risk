import streamlit as st

# Load environment variables from .env if present
from dotenv import load_dotenv
load_dotenv()
import joblib
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airline Operational Risk",
    page_icon="✈️",
    layout="centered",
)

# ── Corporate CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:          #f5f6f8;
    --white:       #ffffff;
    --border:      #e2e5ea;
    --text:        #1a1f2e;
    --muted:       #6b7280;
    --accent:      #1d4ed8;
    --accent-light:#eff4ff;
    --radius:      8px;
}

html, body, .stApp {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

.main .block-container {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2.5rem 3rem 3rem !important;
    max-width: 760px !important;
    margin-top: 2.5rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}

h1 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.65rem !important;
    font-weight: 600 !important;
    letter-spacing: -.02em !important;
    color: var(--text) !important;
    margin-bottom: .15rem !important;
    border-bottom: 2px solid var(--accent) !important;
    padding-bottom: .6rem !important;
    display: inline-block !important;
}

.stMarkdown p {
    color: var(--muted) !important;
    font-size: .88rem !important;
    margin-top: .35rem !important;
}

.stNumberInput label,
.stSelectbox label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: .78rem !important;
    font-weight: 500 !important;
    letter-spacing: .02em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

.stNumberInput input {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .9rem !important;
    transition: border-color .15s, box-shadow .15s;
}
.stNumberInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(29,78,216,.1) !important;
    outline: none !important;
}

.stSelectbox > div > div {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-size: .9rem !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(29,78,216,.1) !important;
}

[data-baseweb="popover"] ul {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,.08) !important;
}
[data-baseweb="popover"] li { color: var(--text) !important; font-size: .88rem !important; }
[data-baseweb="popover"] li:hover { background: var(--accent-light) !important; }

hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: .88rem !important;
    border-radius: var(--radius) !important;
    padding: .5rem 1.4rem !important;
    transition: all .15s ease !important;
}

.stButton:last-of-type > button {
    background: var(--accent) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 1px 3px rgba(29,78,216,.3) !important;
}
.stButton:last-of-type > button:hover {
    background: #1e40af !important;
    box-shadow: 0 2px 6px rgba(29,78,216,.35) !important;
}

.stButton:first-of-type > button {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
}
.stButton:first-of-type > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-light) !important;
}

.stMarkdown h3 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    margin-bottom: .75rem !important;
}
.stMarkdown h4 {
    font-size: .85rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: .04em !important;
}
.stMarkdown strong {
    font-weight: 500 !important;
    color: var(--muted) !important;
    font-size: .85rem !important;
    text-transform: uppercase !important;
    letter-spacing: .03em !important;
}

.stSuccess {
    background: #f0fdf4 !important;
    border: 1px solid #bbf7d0 !important;
    border-radius: var(--radius) !important;
    color: #166534 !important;
    font-size: .88rem !important;
}
.stWarning {
    background: #fffbeb !important;
    border: 1px solid #fde68a !important;
    border-radius: var(--radius) !important;
    color: #92400e !important;
}

.stSpinner > div { border-top-color: var(--accent) !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
model, feature_cols = joblib.load("xgboost_model10_with_features.joblib")
feature_cols = [col if col != 'TAXI_OUT' else 'EXPECTED_TAXI_TIME' for col in feature_cols]
if 'EXPECTED_TAXI_TIME' not in feature_cols:
    feature_cols.append('EXPECTED_TAXI_TIME')

# ── Header ────────────────────────────────────────────────────────────────────
st.title("✈ Airline Operational Risk Prediction")
st.write("Enter the required features to predict operational risk.")

# ── Data ──────────────────────────────────────────────────────────────────────
dep_period_options = ['Morning', 'Afternoon', 'Evening', 'Night']

carrier_map = {
    'AA': 'American Airlines',
    'WN': 'Southwest Airlines',
    'UA': 'United Airlines',
    'B6': 'JetBlue Airways',
    'AS': 'Alaska Airlines',
    'DL': 'Delta Air Lines',
    'HA': 'Hawaiian Airlines',
    'G4': 'Allegiant Air',
    'NK': 'Spirit Airlines',
    'F9': 'Frontier Airlines',
}
carrier_options = list(carrier_map.keys())
carrier_names   = [carrier_map[c] for c in carrier_options]

feature_labels = {
    'QUARTER':             'Quarter (1-4)',
    'MONTH':               'Month (1-12)',
    'DAY_OF_WEEK':         'Day of Week (1=Mon, 7=Sun)',
    'DEP_HOUR':            'Departure Hour (0-23)',
    'ORIGIN_AIRPORT_ID':   'Origin Airport ID',
    'DEST_AIRPORT_ID':     'Destination Airport ID',
    'DISTANCE':            'Flight Distance (miles)',
    'EXPECTED_TAXI_TIME':  'Expected Taxi Time (minutes)',
    'DEP_PERIOD':          'Departure Period',
    'MKT_UNIQUE_CARRIER':  'Marketing Carrier',
}

example = {
    'QUARTER': 1,
    'MONTH': 5,
    'DAY_OF_WEEK': 3,
    'DEP_HOUR': 14,
    'ORIGIN_AIRPORT_ID': 12478,
    'DEST_AIRPORT_ID': 12892,
    'DISTANCE': 800,
    'TAXI_OUT': 15,
    'DEP_PERIOD': 'Morning',
    'MKT_UNIQUE_CARRIER': 'B6',
}

# ── Input form ────────────────────────────────────────────────────────────────
def user_input_features():
    input_data = {}
    col1, col2 = st.columns(2)
    with col1:
        input_data['QUARTER']            = st.number_input(feature_labels['QUARTER'],           value=example['QUARTER'])
        input_data['MONTH']              = st.number_input(feature_labels['MONTH'],             value=example['MONTH'])
        input_data['DAY_OF_WEEK']        = st.number_input(feature_labels['DAY_OF_WEEK'],       value=example['DAY_OF_WEEK'])
        input_data['DEP_HOUR']           = st.number_input(feature_labels['DEP_HOUR'],          value=example['DEP_HOUR'])
    with col2:
        input_data['ORIGIN_AIRPORT_ID']  = st.number_input(feature_labels['ORIGIN_AIRPORT_ID'], value=example['ORIGIN_AIRPORT_ID'])
        input_data['DEST_AIRPORT_ID']    = st.number_input(feature_labels['DEST_AIRPORT_ID'],   value=example['DEST_AIRPORT_ID'])
        input_data['DISTANCE']           = st.number_input(feature_labels['DISTANCE'],          value=example['DISTANCE'])
        input_data['EXPECTED_TAXI_TIME'] = st.number_input(feature_labels['EXPECTED_TAXI_TIME'],value=example['TAXI_OUT'])

    col3, col4 = st.columns(2)
    with col3:
        dep_period   = st.selectbox(feature_labels['DEP_PERIOD'],         dep_period_options,
                                    index=dep_period_options.index(example['DEP_PERIOD']))
    with col4:
        carrier_full = st.selectbox(feature_labels['MKT_UNIQUE_CARRIER'], carrier_names,
                                    index=carrier_names.index(carrier_map[example['MKT_UNIQUE_CARRIER']]))
    carrier = [k for k, v in carrier_map.items() if v == carrier_full][0]

    for period in dep_period_options:
        colname = f'DEP_PERIOD_{period}'
        if colname in feature_cols:
            input_data[colname] = 1 if dep_period == period else 0
    for c in carrier_options:
        colname = f'MKT_UNIQUE_CARRIER_{c}'
        if colname in feature_cols:
            input_data[colname] = 1 if carrier == c else 0

    return pd.DataFrame([input_data]), dep_period, carrier

# ── Session state ─────────────────────────────────────────────────────────────
if 'autofill' not in st.session_state:
    st.session_state['autofill'] = False
if st.button('Auto-fill Example'):
    st.session_state['autofill'] = True

if st.session_state['autofill']:
    input_df, dep_period, carrier = user_input_features()
    st.session_state['autofill'] = False
else:
    input_df, dep_period, carrier = user_input_features()

input_df = input_df.astype(float)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict"):
    X     = input_df[feature_cols]
    proba = model.predict_proba(X)[:, 1][0]
    pred  = int(proba > 0.45)

    if proba >= 0.75:
        risk_level = 'High Risk'
        color      = 'red'
        advice     = 'Immediate attention required. Consider delaying or reviewing operational plans.'
    elif proba >= 0.5:
        risk_level = 'Moderate Risk'
        color      = 'orange'
        advice     = 'Monitor closely. Some risk factors present.'
    else:
        risk_level = 'Low Risk'
        color      = 'green'
        advice     = 'No significant operational risk detected.'

    st.markdown("### Prediction Result")
    st.markdown(
        f"**Risk Level:** <span style='color:{color}; font-size:1.2em'><b>{risk_level}</b></span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Interpretation:** {advice}")
    st.markdown(f"**Predicted Class:** {'Operational Risk' if pred == 1 else 'No Risk'}")

    st.markdown("---")
    try:
        from gemini_rag import generate_rag_explanation
        feature_context = "\n".join([
            f"Quarter: {input_df['QUARTER'][0]}",
            f"Month: {input_df['MONTH'][0]}",
            f"Day of Week: {input_df['DAY_OF_WEEK'][0]}",
            f"Departure Hour: {input_df['DEP_HOUR'][0]}",
            f"Origin Airport ID: {input_df['ORIGIN_AIRPORT_ID'][0]}",
            f"Destination Airport ID: {input_df['DEST_AIRPORT_ID'][0]}",
            f"Distance: {input_df['DISTANCE'][0]}",
            f"Expected Taxi Time: {input_df['EXPECTED_TAXI_TIME'][0]}",
            f"Departure Period: {dep_period}",
            f"Marketing Carrier: {carrier}",
        ])
        user_query = (
            f"Given the above input and prediction probability {proba:.2%}, "
            "what are the main operational risk factors and recommendations?"
        )
        with st.spinner("Generating operational recommendation..."):
            if risk_level == 'Low Risk':
                explanation = "Most Likely Delay: None. Justification: There is no problem, risk is low and it is good that operations are running smoothly."
            else:
                explanation = generate_rag_explanation(user_query, feature_context)
        st.markdown("#### Operational Recommendation")
        st.success(explanation)
    except Exception as e:
        st.warning(f"Operational recommendation unavailable: {e}")