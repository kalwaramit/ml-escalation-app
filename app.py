# ============================================================
# SECTION 6: STREAMLIT DEPLOYMENT DEMO
# H9MLAI Machine Learning Project
# Save this as: app.py
# Run with: streamlit run app.py
# ============================================================
# SETUP (run in Colab first to save models, then locally):
#   pip install streamlit joblib xgboost tensorflow shap
#   streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import warnings
warnings.filterwarnings('ignore')
import os

 

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Escalation Predictor",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        color: #1a1a2e; margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 0.95rem; color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1.2rem; border-left: 4px solid #4CAF50;
        margin-bottom: 0.75rem;
    }
    .metric-card.danger { border-left-color: #F44336; background: #fff5f5; }
    .metric-card.warning { border-left-color: #FF9800; background: #fffbf0; }
    .metric-card.success { border-left-color: #4CAF50; background: #f0fff4; }
    .risk-high   { color: #F44336; font-size: 2.5rem; font-weight: 800; }
    .risk-medium { color: #FF9800; font-size: 2.5rem; font-weight: 800; }
    .risk-low    { color: #4CAF50; font-size: 2.5rem; font-weight: 800; }
    .feature-tag {
        display: inline-block; padding: 2px 10px;
        border-radius: 12px; font-size: 0.75rem;
        margin: 2px; font-weight: 500;
    }
    .stButton > button {
        width: 100%; background: #1a1a2e;
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 1rem; font-size: 1rem;
        font-weight: 600; cursor: pointer;
    }
    .stButton > button:hover { background: #2d2d5e; }
    .section-divider {
        border: none; border-top: 1px solid #e0e0e0;
        margin: 1.5rem 0;
    }
    .badge-escalate {
        background: #F44336; color: white; padding: 6px 16px;
        border-radius: 20px; font-weight: 700; font-size: 1rem;
    }
    .badge-safe {
        background: #4CAF50; color: white; padding: 6px 16px;
        border-radius: 20px; font-weight: 700; font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── MODEL LOADING ────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all trained models. Cached so only loads once."""
    models = {}
    model_files = {
        'XGBoost':             'xgboost_best.pkl',
        'Random Forest':       'random_forest_best.pkl',
        'Logistic Regression': 'logistic_regression_best.pkl',
    }
    for name, path in model_files.items():
        try:
            models[name] = joblib.load(path)
        except FileNotFoundError:
            models[name] = None

    # Try loading Keras MLP
    try:
        from tensorflow import keras
        models['MLP'] = keras.models.load_model('mlp_config1_best.h5')
    except Exception:
        models['MLP'] = None

    # Load scaler
    try:
        scaler = joblib.load('standard_scaler.pkl')
    except FileNotFoundError:
        scaler = None

    return models, scaler


@st.cache_resource
def load_feature_names():
    """Load feature names from saved CSV or define manually."""
    try:
        df = pd.read_csv('unified_dataset.csv', nrows=1)
        # Match Section 2 feature engineering
        feat_cols = [c for c in df.columns
                     if c not in ['customer_id','case_id','escalation',
                                  'sla_breach','priority','emotion_label',
                                  'priority_num','sla_deadline_hrs']]
        return feat_cols
    except Exception:
        return None


# ── FEATURE ENGINEERING (mirrors Section 2) ──────────────────
def engineer_features(raw_input: dict) -> pd.DataFrame:
    """
    Replicates the Section 2 preprocessing pipeline on a single case.
    raw_input keys match the sidebar widget names.
    """
    priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
    sla_map      = {'Low': 48, 'Medium': 24, 'High': 8, 'Critical': 4}
    channel_opts = ['Email', 'Live Chat', 'Phone', 'Social Media']

    p_enc    = priority_map[raw_input['priority']]
    sla_dead = sla_map[raw_input['priority']]

    # One-hot channel
    ch_ohe = {f"ch_{c}": int(raw_input['channel'] == c) for c in channel_opts}

    # Engineered features
    interaction_rate   = raw_input['num_interactions'] / (raw_input['response_time_hrs'] + 0.01)
    urgency_score      = p_enc * (raw_input['transfer_count'] + 1)
    engagement_deficit = raw_input['days_since_last_login'] / (raw_input['web_sessions'] + 1)
    frustration_index  = (
        -raw_input['sentiment_score'] * 0.5
        + raw_input['exclamation_cnt'] * 0.3
        + (raw_input['word_count'] / 100) * 0.2
    )
    sla_risk_ratio = raw_input['response_time_hrs'] / (sla_dead + 0.01)

    # Emotion encoding
    emotion_map = {
        'Positive': 4, 'Neutral-Positive': 3, 'Neutral': 2,
        'Neutral-Negative': 1, 'Negative': 0
    }
    emotion_enc = emotion_map.get(raw_input['emotion_label'], 2)

    row = {
        'response_time_hrs':     raw_input['response_time_hrs'],
        'num_interactions':      raw_input['num_interactions'],
        'transfer_count':        raw_input['transfer_count'],
        'agent_experience':      raw_input['agent_experience'],
        'priority_encoded':      p_enc,
        **ch_ohe,
        'sentiment_score':       raw_input['sentiment_score'],
        'emotion_encoded':       emotion_enc,
        'word_count':            raw_input['word_count'],
        'exclamation_cnt':       raw_input['exclamation_cnt'],
        'question_cnt':          raw_input['question_cnt'],
        'web_sessions':          raw_input['web_sessions'],
        'login_frequency':       raw_input['login_frequency'],
        'product_interactions':  raw_input['product_interactions'],
        'days_since_last_login': raw_input['days_since_last_login'],
        'pages_per_session':     raw_input['pages_per_session'],
        'interaction_rate':      interaction_rate,
        'urgency_score':         urgency_score,
        'engagement_deficit':    engagement_deficit,
        'frustration_index':     frustration_index,
        'sla_risk_ratio':        sla_risk_ratio,
    }
    return pd.DataFrame([row])


def predict_all(feature_df: pd.DataFrame, models: dict, scaler) -> dict:
    """Run all available models and return probability dict."""
    predictions = {}

    # Scale features
    if scaler is not None:
        try:
            X = scaler.transform(feature_df)
            X_df = pd.DataFrame(X, columns=feature_df.columns)
        except Exception:
            X_df = feature_df.copy()
    else:
        X_df = feature_df.copy()

    for name, model in models.items():
        if model is None:
            continue
        try:
            t0 = time.time()
            if name == 'MLP':
                prob = float(model.predict(X_df.values.astype(np.float32),
                                           verbose=0).flatten()[0])
            else:
                prob = float(model.predict_proba(X_df)[:, 1][0])
            latency_ms = round((time.time() - t0) * 1000, 2)
            predictions[name] = {'probability': prob, 'latency_ms': latency_ms}
        except Exception as e:
            predictions[name] = {'probability': None, 'error': str(e)}

    return predictions


# ── RISK LEVEL HELPER ────────────────────────────────────────
def risk_level(prob: float):
    if prob >= 0.70:
        return "HIGH RISK", "risk-high", "danger", "🔴"
    elif prob >= 0.40:
        return "MEDIUM RISK", "risk-medium", "warning", "🟡"
    else:
        return "LOW RISK", "risk-low", "success", "🟢"


def prob_bar(prob: float) -> str:
    """Simple HTML progress bar."""
    pct   = int(prob * 100)
    color = "#F44336" if prob >= 0.7 else ("#FF9800" if prob >= 0.4 else "#4CAF50")
    return (f'<div style="background:#eee;border-radius:6px;height:12px;margin:6px 0;">'
            f'<div style="width:{pct}%;background:{color};height:12px;'
            f'border-radius:6px;transition:width 0.4s;"></div></div>'
            f'<small style="color:{color};font-weight:600;">{pct}% escalation probability</small>')


# ── MAIN APP ─────────────────────────────────────────────────
def main():
    # Header
    st.markdown('<div class="main-header">🚨 Customer Escalation Predictor</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">H9MLAI Project — Real-time escalation risk '
        'scoring using ML & Deep Learning</div>',
        unsafe_allow_html=True
    )

    # Load models
    with st.spinner("Loading models..."):
        models, scaler = load_models()

    loaded = [k for k, v in models.items() if v is not None]
    if loaded:
        st.success(f"✅ Models loaded: {', '.join(loaded)}")
    else:
        st.warning("⚠️ No model files found. Run Sections 1–4 first, "
                   "then save models to the same directory as app.py")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── SIDEBAR — INPUT FORM ──────────────────────────────────
    with st.sidebar:
        st.header("📋 Case Details")
        st.caption("Fill in the service case information below")

        st.subheader("Service Case")
        channel     = st.selectbox("Communication Channel",
                                   ['Email', 'Phone', 'Live Chat', 'Social Media'])
        priority    = st.selectbox("Case Priority",
                                   ['Low', 'Medium', 'High', 'Critical'])
        response_time = st.slider("Response Time (hours)", 0.5, 72.0, 6.0, 0.5)
        num_interactions = st.slider("Number of Interactions", 1, 20, 4)
        transfer_count   = st.slider("Transfer Count", 0, 10, 1)
        agent_experience = st.slider("Agent Experience (years)", 1, 10, 5)

        st.subheader("Customer Activity")
        web_sessions      = st.slider("Web Sessions (30d)", 0, 60, 12)
        login_frequency   = st.slider("Login Frequency (30d)", 0, 30, 8)
        product_interactions = st.slider("Product Interactions", 0, 50, 20)
        days_since_login  = st.slider("Days Since Last Login", 0, 90, 7)
        pages_per_session = st.slider("Pages per Session", 1.0, 20.0, 4.5, 0.5)

        st.subheader("Interaction Sentiment")
        sentiment_score = st.slider("Sentiment Score", -1.0, 1.0, 0.1, 0.05)
        emotion_label   = st.selectbox("Emotion",
                                       ['Positive', 'Neutral-Positive', 'Neutral',
                                        'Neutral-Negative', 'Negative'])
        word_count      = st.slider("Message Word Count", 10, 250, 60)
        exclamation_cnt = st.slider("Exclamation Count", 0, 15, 1)
        question_cnt    = st.slider("Question Count", 0, 10, 2)

        st.markdown("---")
        predict_btn = st.button("🔍 Predict Escalation Risk")

    # ── MAIN PANEL — DEMO CASES ───────────────────────────────
    col_main, col_demo = st.columns([2, 1])

    with col_demo:
        st.subheader("📌 Demo Cases")
        st.caption("Click to auto-fill a scenario")

        if st.button("🔴 High-risk case"):
            st.session_state['demo'] = 'high'
            st.rerun()
        if st.button("🟡 Medium-risk case"):
            st.session_state['demo'] = 'medium'
            st.rerun()
        if st.button("🟢 Low-risk case"):
            st.session_state['demo'] = 'low'
            st.rerun()

        st.markdown("---")
        st.subheader("ℹ️ Model Info")
        for name, m in models.items():
            status = "✅" if m is not None else "❌"
            st.markdown(f"{status} **{name}**")

    # Handle demo case population
    demo_inputs = {
        'high': {
            'channel': 'Social Media', 'priority': 'Critical',
            'response_time_hrs': 18.0, 'num_interactions': 9,
            'transfer_count': 5, 'agent_experience': 2,
            'web_sessions': 2, 'login_frequency': 2,
            'product_interactions': 3, 'days_since_last_login': 28.0,
            'pages_per_session': 2.0, 'sentiment_score': -0.85,
            'emotion_label': 'Negative', 'word_count': 180,
            'exclamation_cnt': 8, 'question_cnt': 5,
        },
        'medium': {
            'channel': 'Email', 'priority': 'High',
            'response_time_hrs': 9.0, 'num_interactions': 5,
            'transfer_count': 2, 'agent_experience': 5,
            'web_sessions': 8, 'login_frequency': 6,
            'product_interactions': 12, 'days_since_last_login': 10.0,
            'pages_per_session': 3.5, 'sentiment_score': -0.2,
            'emotion_label': 'Neutral-Negative', 'word_count': 90,
            'exclamation_cnt': 2, 'question_cnt': 3,
        },
        'low': {
            'channel': 'Phone', 'priority': 'Low',
            'response_time_hrs': 2.0, 'num_interactions': 2,
            'transfer_count': 0, 'agent_experience': 8,
            'web_sessions': 20, 'login_frequency': 15,
            'product_interactions': 35, 'days_since_last_login': 1.0,
            'pages_per_session': 6.0, 'sentiment_score': 0.6,
            'emotion_label': 'Positive', 'word_count': 30,
            'exclamation_cnt': 0, 'question_cnt': 1,
        },
    }

    # Build raw_input from sidebar or demo
    if 'demo' in st.session_state and st.session_state['demo'] in demo_inputs:
        raw = demo_inputs[st.session_state['demo']]
        st.session_state.pop('demo')
    else:
        raw = {
            'channel': channel, 'priority': priority,
            'response_time_hrs': response_time,
            'num_interactions': num_interactions,
            'transfer_count': transfer_count,
            'agent_experience': agent_experience,
            'web_sessions': web_sessions,
            'login_frequency': login_frequency,
            'product_interactions': product_interactions,
            'days_since_last_login': days_since_login,
            'pages_per_session': pages_per_session,
            'sentiment_score': sentiment_score,
            'emotion_label': emotion_label,
            'word_count': word_count,
            'exclamation_cnt': exclamation_cnt,
            'question_cnt': question_cnt,
        }

    # ── PREDICTION OUTPUT ─────────────────────────────────────
    with col_main:
        if predict_btn or st.session_state.get('auto_predict'):
            st.subheader("🎯 Prediction Results")

            # Feature engineering
            feature_df = engineer_features(raw)

            # Run all models
            with st.spinner("Running models..."):
                preds = predict_all(feature_df, models, scaler)

            if not preds:
                st.error("No models available. Please load model files first.")
                return

            # Primary model result (XGBoost preferred, then first available)
            primary = next(
                (n for n in ['XGBoost','Random Forest','Logistic Regression','MLP']
                 if n in preds and preds[n]['probability'] is not None),
                None
            )

            if primary:
                prob_primary = preds[primary]['probability']
                label, css_class, card_type, emoji = risk_level(prob_primary)

                # Big risk indicator
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown(
                        f'<div class="metric-card {card_type}">'
                        f'<div style="font-size:0.8rem;color:#666;margin-bottom:4px;">'
                        f'Primary Model ({primary})</div>'
                        f'<div class="{css_class}">{emoji} {label}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(prob_bar(prob_primary), unsafe_allow_html=True)

                    # Prediction badge
                    pred_class = "escalate" if prob_primary >= 0.5 else "safe"
                    badge_text = "⚠️ WILL ESCALATE" if pred_class == "escalate" else "✅ SAFE"
                    badge_cls  = f"badge-{pred_class}"
                    st.markdown(
                        f'<div style="margin-top:12px;">'
                        f'<span class="{badge_cls}">{badge_text}</span></div>',
                        unsafe_allow_html=True
                    )

                with col_b:
                    # All model probabilities
                    st.markdown("**All model predictions:**")
                    for model_name, result in preds.items():
                        if result.get('probability') is None:
                            continue
                        p = result['probability']
                        lat = result.get('latency_ms', '—')
                        _, _, ct, em = risk_level(p)
                        st.markdown(
                            f'<div class="metric-card {ct}" style="padding:0.7rem 1rem;">'
                            f'<strong>{model_name}</strong> &nbsp;'
                            f'<span style="font-size:0.8rem;color:#666;">({lat} ms)</span><br>'
                            f'{em} <strong>{p:.1%}</strong> escalation probability'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            # ── INPUT SUMMARY ─────────────────────────────────
            st.subheader("📊 Input Feature Summary")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Service Case**")
                st.write(f"Channel: `{raw['channel']}`")
                st.write(f"Priority: `{raw['priority']}`")
                st.write(f"Response Time: `{raw['response_time_hrs']}h`")
                st.write(f"Interactions: `{raw['num_interactions']}`")
                st.write(f"Transfers: `{raw['transfer_count']}`")
                st.write(f"Agent Exp.: `{raw['agent_experience']} yrs`")

            with col2:
                st.markdown("**Customer Activity**")
                st.write(f"Web Sessions: `{raw['web_sessions']}`")
                st.write(f"Login Freq.: `{raw['login_frequency']}`")
                st.write(f"Product Inter.: `{raw['product_interactions']}`")
                st.write(f"Days Since Login: `{raw['days_since_last_login']}`")
                st.write(f"Pages/Session: `{raw['pages_per_session']}`")

            with col3:
                st.markdown("**Sentiment Signals**")
                st.write(f"Score: `{raw['sentiment_score']}`")
                st.write(f"Emotion: `{raw['emotion_label']}`")
                st.write(f"Word Count: `{raw['word_count']}`")
                st.write(f"Exclamations: `{raw['exclamation_cnt']}`")
                st.write(f"Questions: `{raw['question_cnt']}`")

            # Engineered features preview
            with st.expander("🔧 Engineered Features (from preprocessing pipeline)"):
                feat_df = engineer_features(raw)
                eng_cols = ['interaction_rate', 'urgency_score',
                            'engagement_deficit', 'frustration_index', 'sla_risk_ratio']
                eng_preview = feat_df[eng_cols].T.rename(columns={0: 'Value'}).round(4)
                st.dataframe(eng_preview, use_container_width=True)

            # ── RISK INTERPRETATION ───────────────────────────
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.subheader("💡 Risk Interpretation")

            if primary and prob_primary >= 0.70:
                st.error(
                    "**High escalation risk detected.** Recommend immediate proactive "
                    "outreach. Assign an experienced agent and consider priority upgrade. "
                    "Key risk factors: high transfer count, negative sentiment, "
                    "or extended response time."
                )
            elif primary and prob_primary >= 0.40:
                st.warning(
                    "**Moderate escalation risk.** Monitor this case closely. "
                    "Consider a proactive check-in call to address customer concerns "
                    "before the situation deteriorates."
                )
            else:
                st.success(
                    "**Low escalation risk.** Case is on track. Standard handling "
                    "procedures are sufficient. Continue monitoring response time "
                    "against SLA deadline."
                )

        else:
            # Landing state — instructions
            st.subheader("👈 How to use this tool")
            st.markdown("""
            1. **Fill in case details** in the sidebar (or click a demo case →)
            2. **Click "Predict Escalation Risk"** to run all models
            3. **Review the risk score** and model consensus
            4. **Expand "Engineered Features"** to see the preprocessing pipeline in action

            ---

            **Models available:**
            - **XGBoost** — primary model (best F1 in evaluation)
            - **Random Forest** — ensemble baseline
            - **Logistic Regression** — interpretable baseline
            - **MLP** — deep learning model (if .h5 file present)

            ---
            **Note:** Model files must be in the same folder as `app.py`.
            Run the Colab notebook (Sections 1–4) to generate them.
            """)

    # ── FOOTER ───────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.caption(
        "H9MLAI Machine Learning Project | MSc Artificial Intelligence 2026 | "
        "Predicting Customer Service Escalation Using ML & Deep Learning"
    )


if __name__ == "__main__":
    main()
