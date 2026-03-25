import streamlit as st
import joblib
import pandas as pd


# Load model and feature columns
model, feature_cols = joblib.load("xgboost_model10_with_features.joblib")
feature_cols = [col if col != 'TAXI_OUT' else 'EXPECTED_TAXI_TIME' for col in feature_cols]
if 'EXPECTED_TAXI_TIME' not in feature_cols:
    feature_cols.append('EXPECTED_TAXI_TIME')

st.title("Airline Operational Risk Prediction")
st.write("Enter the required features to predict operational risk.")

# Create input fields for each feature


# Define original categorical options
dep_period_options = ['Morning', 'Afternoon', 'Evening', 'Night']
carrier_options = ['AS', 'B6', 'DL', 'F9', 'G4', 'HA', 'NK', 'UA', 'WN']


# User-friendly labels for input fields
feature_labels = {
    'QUARTER': 'Quarter (1-4)',
    'MONTH': 'Month (1-12)',
    'DAY_OF_WEEK': 'Day of Week (1=Mon, 7=Sun)',
    'DEP_HOUR': 'Departure Hour (0-23)',
    'ORIGIN_AIRPORT_ID': 'Origin Airport ID',
    'DEST_AIRPORT_ID': 'Destination Airport ID',
    'DISTANCE': 'Flight Distance (miles)',
    'EXPECTED_TAXI_TIME': 'Expected Taxi Time (minutes)',
    'DEP_PERIOD': 'Departure Period',
    'MKT_UNIQUE_CARRIER': 'Marketing Carrier',
}

# Carrier code to full name mapping
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
    'F9': 'Frontier Airlines'
}
carrier_options = list(carrier_map.keys())
carrier_names = [carrier_map[c] for c in carrier_options]

# Example values for auto-fill
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



def user_input_features():
    # Main numeric features
    input_data = {}
    input_data['QUARTER'] = st.number_input(feature_labels['QUARTER'], value=example['QUARTER'])
    input_data['MONTH'] = st.number_input(feature_labels['MONTH'], value=example['MONTH'])
    input_data['DAY_OF_WEEK'] = st.number_input(feature_labels['DAY_OF_WEEK'], value=example['DAY_OF_WEEK'])
    input_data['DEP_HOUR'] = st.number_input(feature_labels['DEP_HOUR'], value=example['DEP_HOUR'])
    input_data['ORIGIN_AIRPORT_ID'] = st.number_input(feature_labels['ORIGIN_AIRPORT_ID'], value=example['ORIGIN_AIRPORT_ID'])
    input_data['DEST_AIRPORT_ID'] = st.number_input(feature_labels['DEST_AIRPORT_ID'], value=example['DEST_AIRPORT_ID'])
    input_data['DISTANCE'] = st.number_input(feature_labels['DISTANCE'], value=example['DISTANCE'])
    input_data['EXPECTED_TAXI_TIME'] = st.number_input(feature_labels['EXPECTED_TAXI_TIME'], value=example['TAXI_OUT'])

    # Dropdowns for categorical features
    dep_period = st.selectbox(feature_labels['DEP_PERIOD'], dep_period_options, index=dep_period_options.index(example['DEP_PERIOD']))
    carrier_full = st.selectbox(feature_labels['MKT_UNIQUE_CARRIER'], carrier_names, index=carrier_names.index(carrier_map[example['MKT_UNIQUE_CARRIER']]))
    # Map back to code
    carrier = [k for k, v in carrier_map.items() if v == carrier_full][0]

    # One-hot encoding for dummies
    for period in dep_period_options:
        colname = f'DEP_PERIOD_{period}'
        if colname in feature_cols:
            input_data[colname] = 1 if dep_period == period else 0
    for c in carrier_options:
        colname = f'MKT_UNIQUE_CARRIER_{c}'
        if colname in feature_cols:
            input_data[colname] = 1 if carrier == c else 0

    return pd.DataFrame([input_data]), dep_period, carrier

# Auto-fill button
if 'autofill' not in st.session_state:
    st.session_state['autofill'] = False
if st.button('Auto-fill Example'):
    st.session_state['autofill'] = True


if st.session_state['autofill']:
    # Use example values
    input_df, dep_period, carrier = user_input_features()
    st.session_state['autofill'] = False
else:
    input_df, dep_period, carrier = user_input_features()
input_df = input_df.astype(float)

if st.button("Predict"):
    X = input_df[feature_cols]
    proba = model.predict_proba(X)[:, 1][0]
    pred = int(proba > 0.45)
    
    # Interpret the risk
    if proba >= 0.75:
        risk_level = 'High Risk'
        color = 'red'
        advice = 'Immediate attention required. Consider delaying or reviewing operational plans.'
    elif proba >= 0.5:
        risk_level = 'Moderate Risk'
        color = 'orange'
        advice = 'Monitor closely. Some risk factors present.'
    else:
        risk_level = 'Low Risk'
        color = 'green'
        advice = 'No significant operational risk detected.'

    st.markdown(f"### Prediction Result")
    st.markdown(f"**Risk Level:** <span style='color:{color}; font-size:1.2em'><b>{risk_level}</b></span>", unsafe_allow_html=True)
    st.markdown(f"**Probability of Operational Risk:** {proba:.2%}")
    st.markdown(f"**Interpretation:** {advice}")
    st.markdown(f"**Predicted Class:** {'Operational Risk' if pred == 1 else 'No Risk'}")

    # --- RAG Integration ---
    if risk_level != 'Low Risk':
        st.markdown("---")
        try:
            from gemini_rag import generate_rag_explanation
            # Only send flight features to LLM
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
                f"Marketing Carrier: {carrier}"
            ])
            user_query = f"Given the above input and prediction probability {proba:.2%}, what are the main operational risk factors and recommendations?"
            with st.spinner("Generating operational recommendation..."):
                explanation = generate_rag_explanation(user_query, feature_context)
            st.markdown("#### Operational Recommendation")
            st.success(explanation)
        except Exception as e:
            st.warning(f"Operational recommendation unavailable: {e}")
