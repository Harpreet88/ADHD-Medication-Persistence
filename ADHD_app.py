# streamlit run app.py
import warnings
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import joblib

# Suppress ScriptRunContext warning from lifelines/streamlit
warnings.filterwarnings("ignore", message="missing ScriptRunContext")
warnings.filterwarnings("ignore", message="ScriptRunContext")

st.set_page_config(page_title="ADHD Persistence", layout="wide")

# =======================
# Artifacts
# =======================
ARTIFACTS = {
    "Adjusted (include SES)": {
        "model": "cph_a.pkl",
        "pre": "columns_a.pkl",
        "feats": "feats_a.pkl",
    },
    "Unadjusted (omit SES)": {
        "model": "cph_u.pkl",
        "pre": "columns_u.pkl",
        "feats": "feats_u.pkl",
    },
}

# =======================
# Raw input schema
# =======================
SES_LEVELS = ["Low", "Medium", "High"]
ADHD_CLASS_LEVELS = [
    "non_stimulant",
    "methylphenidate_first",
    "methylphenidate_second",
    "amphetamine_second",
    # add more if you trained them (e.g., "amphetamine_first")
]

COMORBID_MEDS = [
    "Fluoxetine","Sertraline","Citalopram","Paroxetine","Fluvoxamine","Bupropion","Venlafaxine",
    "Desvenlafaxine","Vortioxetine","Duloxetine","Vilazodone",
    "Intuniv XR","Risperidone","Aripiprazole","Olanzapine","Quetiapine","Ziprasidone","Paliperidone",
    "Clonidine","Guanfacine XR",
    "Lithium","Lamotrigine","Carbamazepine","Divalproex","Valproic acid",
    "Asenapine","Cariprazine",
    "Ativan","Clonazepam",
]

# =======================
# Loaders & helpers
# =======================
@st.cache_resource
def load_artifacts(choice: str):
    cfg = ARTIFACTS[choice]
    cph = joblib.load(cfg["model"])          # lifelines.CoxPHFitter
    pre = joblib.load(cfg["pre"])            # sklearn ColumnTransformer
    feat_names = joblib.load(cfg["feats"])   # list[str] after transform
    return cph, pre, feat_names

def build_raw_row(include_ses: bool) -> pd.DataFrame:
    st.sidebar.header("Patient inputs")

    age = st.sidebar.slider("Age (years)", min_value=5, max_value=80, value=12, step=1)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    adhd_med_class = st.sidebar.selectbox("ADHD medication class", ADHD_CLASS_LEVELS)

    ses_val = None
    if include_ses:
        ses_val = st.sidebar.selectbox("Socioeconomic Status (SES)", SES_LEVELS)

    st.sidebar.divider()
    meds_selected = st.sidebar.multiselect(
        "Comorbid medications (select all that apply)", COMORBID_MEDS, default=[]
    )

    data = {
        "age": age,
        "sex": sex,
        "adhd_med_class": adhd_med_class,
    }
    if include_ses:
        data["SES"] = ses_val

    # 0/1 flags for each med
    for m in COMORBID_MEDS:
        data[m] = 1 if m in meds_selected else 0

    return pd.DataFrame([data])

def preprocess_raw_row(raw_row: pd.DataFrame, feature_names: list, include_ses: bool) -> pd.DataFrame:
    
    processed_row = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Copy age directly
    if 'age' in raw_row.columns and 'age' in feature_names:
        processed_row['age'] = raw_row['age'].values[0]
    
    # Handle sex encoding (Male = 1, Female = 0)
    if 'sex_Male' in feature_names:
        processed_row['sex_Male'] = 1 if raw_row['sex'].values[0] == 'Male' else 0
    
    # Handle ADHD medication class one-hot encoding
    adhd_class = raw_row['adhd_med_class'].values[0]
    for feat in feature_names:
        if feat.startswith('adhd_med_class_'):
            processed_row[feat] = 1 if feat.endswith(adhd_class) else 0
    
    # Handle SES encoding 
    if include_ses:
        ses_val = raw_row['SES'].values[0]
        for feat in feature_names:
            if feat.startswith('SES_'):
                processed_row[feat] = 1 if feat.endswith(ses_val) else 0
    
    # Handle comorbid medications (already 0/1 encoded in raw input)
    for med in COMORBID_MEDS:
        if med in raw_row.columns and med in feature_names:
            processed_row[med] = raw_row[med].values[0]
    
    # Interaction terms
    if 'days_on_first_med_interactions' in feature_names:
        processed_row['days_on_first_med_interactions'] = 0
    
    
    for feat in ['has_anxiety_depression', 'has_odd', 'has_tourette', 'has_bipolar']:
        if feat in feature_names:
            processed_row[feat] = 0
    
    return processed_row

def predict_partial_hazard(cph, X_df: pd.DataFrame) -> float:
    # lifelines expects these columns present (not used in score, but keeps API consistent)
    dfp = X_df.copy()
    dfp["days_on_first_med"] = 0.0
    dfp["switched_med"] = 0
    ph = cph.predict_partial_hazard(dfp).values.ravel()[0]
    return float(ph)

def predict_survival_function(cph, X_df: pd.DataFrame) -> pd.DataFrame:
    dfp = X_df.copy()
    dfp["days_on_first_med"] = 0.0
    dfp["switched_med"] = 0
    sf_df = cph.predict_survival_function(dfp)  # index: time, column: single individual
    return sf_df

@st.cache_resource
def make_explainer(_cph, bg_X: pd.DataFrame):
    # SHAP KernelExplainer for CoxPH
    def f(X):
        dfp = pd.DataFrame(X, columns=bg_X.columns).copy()
        dfp["days_on_first_med"] = 0.0
        dfp["switched_med"] = 0
        ph = _cph.predict_partial_hazard(dfp).values.ravel()
        return np.log(ph + 1e-12)
    return shap.KernelExplainer(f, bg_X.values)

def shap_waterfall(explainer, X_df: pd.DataFrame, title: str, max_display=12):
    try:
        sv = explainer.shap_values(X_df.values, nsamples="auto")
        if len(sv) == 0 or sv[0] is None:
            st.warning("SHAP explanation could not be generated for this prediction")
            return
            
        exp = shap.Explanation(
            values=sv[0],
            base_values=explainer.expected_value,
            data=X_df.values[0],
            feature_names=list(X_df.columns),
        )
        shap.plots.waterfall(exp, max_display=min(max_display, len(X_df.columns)), show=False)
        plt.title(title)
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.warning(f"SHAP explanation could not be generated: {str(e)}")

# =======================
# UI
# =======================
st.title("ADHD Medication Persistence — Survival & Explainability (CoxPH)")

model_variant = st.sidebar.radio("Model variant", list(ARTIFACTS.keys()))
include_ses = (model_variant == "Adjusted (include SES)")

# Load model + feature names (pre is now just a list, not a transformer)
try:
    cph, pre, feat_names = load_artifacts(model_variant)
except Exception as e:
    st.error(f"Could not load artifacts for '{model_variant}'. Ensure pkl files exist.\n{e}")
    st.stop()

# Collect raw inputs
raw_row = build_raw_row(include_ses)
# Preprocess to get model input
try:
    X_df = preprocess_raw_row(raw_row, feat_names, include_ses)
except Exception as e:
    st.error(
        "Preprocessing failed. Could not transform raw inputs to expected format."
        f"\nRaw columns: {list(raw_row.columns)}\nError: {e}"
    )
    st.stop()

# Layout
left, right = st.columns([1.15, 1])

with left:
    st.subheader("Predicted survival")
    sf = predict_survival_function(cph, X_df)
    t = sf.index.values
    s = sf.iloc[:, 0].values

    fig, ax = plt.subplots()
    ax.plot(t, s, linewidth=2)
    ax.set_xlabel("Days on first medication")
    ax.set_ylabel("Survival probability (no switch yet)")
    
    if np.min(s) > 0.95:
        ax.set_ylim(0.9, 1.01)  # Zoom in on high survival probabilities
        ax.text(0.5, 0.95, "Excellent persistence predicted", 
                transform=ax.transAxes, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    else:
        ax.set_ylim(0, 1.0)
    
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    risk = predict_partial_hazard(cph, X_df)
    
    # Format risk value appropriately for display
    if risk < 0.001:
        risk_display = f"{risk:.2e}"  # Scientific notation for very small values
    else:
        risk_display = f"{risk:.3f}" 
    
    st.metric("Relative risk (partial hazard)", risk_display)
    
    # interpretation
    if risk < 0.1:
        st.success("Very low risk of switching medications - excellent persistence predicted!")
    elif risk < 1.0:
        st.info("Below average risk of switching - good persistence predicted")
    elif risk == 1.0:
        st.warning("Average risk of switching")
    else:
        st.error("Higher than average risk of switching medications")

with right:
    st.subheader("Per-patient explanation (SHAP)")
    # SHAP explanation
    bg = pd.DataFrame(np.repeat(X_df.values, repeats=80, axis=0), columns=X_df.columns)
    explainer = make_explainer(cph, bg)
    shap_waterfall(explainer, X_df, title=f"Log-risk contributors — {model_variant}", max_display=12)

def generate_clinical_interpretation(model_variant, raw_row, risk, include_ses):
    
    interpretation = f"## Clinical Interpretation\n\n"
    
    # Model variant context
    if model_variant == "Adjusted (include SES)":
        interpretation += "**Model Context:** This analysis uses the adjusted model that includes socioeconomic status (SES) as a predictor. This model controls for potential confounding effects of socioeconomic factors, providing more reliable estimates of medication-specific effects.\n\n"
    else:
        interpretation += "**Model Context:** This analysis uses the unadjusted model that excludes socioeconomic status. Note that medication effects may be confounded by socioeconomic factors in this model.\n\n"
    
    # Patient characteristics summary
    interpretation += f"**Patient Profile:** {raw_row['age'].values[0]} year old {raw_row['sex'].values[0].lower()} patient prescribed {raw_row['adhd_med_class'].values[0].replace('_', ' ')} as first ADHD medication.\n\n"
    
    if include_ses and 'SES' in raw_row.columns:
        interpretation += f"**Socioeconomic Status:** {raw_row['SES'].values[0]}\n\n"
    
    # Risk interpretation
    if risk < 0.1:
        interpretation += "**Risk Assessment:** VERY LOW risk of medication switching. This patient demonstrates excellent predicted persistence with their first prescribed ADHD medication.\n\n"
    elif risk < 1.0:
        interpretation += "**Risk Assessment:** BELOW AVERAGE risk of medication switching. Good predicted persistence expected.\n\n"
    elif risk == 1.0:
        interpretation += "**Risk Assessment:** AVERAGE risk of medication switching. Typical persistence pattern expected.\n\n"
    else:
        interpretation += "**Risk Assessment:** HIGHER THAN AVERAGE risk of medication switching. This patient may benefit from closer monitoring and support for medication adherence.\n\n"
    
    # Clinical recommendations based on risk
    if risk < 0.3:
        interpretation += "**Clinical Recommendation:** Continue current treatment plan. The patient shows excellent response to first-line therapy.\n\n"
    elif risk < 1.5:
        interpretation += "**Clinical Recommendation:** Standard monitoring recommended. Consider patient education on medication adherence.\n\n"
    else:
        interpretation += "**Clinical Recommendation:** Consider enhanced monitoring and support. Evaluate potential barriers to adherence and consider early intervention strategies.\n\n"
    
    # Key factors influencing prediction
    interpretation += "**Key Predictive Factors:** The SHAP analysis (right panel) shows which factors most influence this prediction. Positive values increase switching risk, negative values decrease risk.\n\n"
    
    # Model limitations note
    interpretation += "**Note:** This prediction is based on statistical modeling of historical patterns. Individual patient factors and clinical judgment should always guide treatment decisions."
    
    return interpretation

# Enhanced visual design with better spacing and organization
st.markdown("---")
st.subheader("📊 Additional Insights & Model Understanding")

# Create a more organized layout with tabs
tab1, tab2, tab3 = st.tabs(["📈 Risk Analysis", "🔍 Model Insights", "💾 Export Results"])

with tab1:
    # Create two columns for risk analysis graphs
    col1, col2 = st.columns(2)

with col1:
    # Age vs Risk analysis
    st.write("**Age vs Predicted Risk Trend**")
    ages_to_test = [5, 10, 15, 20, 30, 40, 50, 60]
    risks_by_age = []
    
    for test_age in ages_to_test:
        test_data = raw_row.copy()
        test_data['age'] = test_age
        X_test = preprocess_raw_row(test_data, feat_names, include_ses)
        test_risk = predict_partial_hazard(cph, X_test)
        risks_by_age.append(test_risk)
    
    fig_age, ax_age = plt.subplots(figsize=(8, 4))
    ax_age.plot(ages_to_test, risks_by_age, 'o-', linewidth=2, markersize=6)
    ax_age.set_xlabel("Age (years)")
    ax_age.set_ylabel("Relative Risk")
    ax_age.set_title("Risk Trend by Age")
    ax_age.grid(alpha=0.3)
    st.pyplot(fig_age)

with col2:
    # Medication class comparison
    st.write("**Risk by Medication Class**")
    med_classes = ADHD_CLASS_LEVELS
    risks_by_med = []
    
    for med_class in med_classes:
        test_data = raw_row.copy()
        test_data['adhd_med_class'] = med_class
        X_test = preprocess_raw_row(test_data, feat_names, include_ses)
        test_risk = predict_partial_hazard(cph, X_test)
        risks_by_med.append(test_risk)
    
    fig_med, ax_med = plt.subplots(figsize=(8, 4))
    bars = ax_med.bar(range(len(med_classes)), risks_by_med)
    ax_med.set_xlabel("Medication Class")
    ax_med.set_ylabel("Relative Risk")
    ax_med.set_title("Risk by Medication Type")
    ax_med.set_xticks(range(len(med_classes)))
    ax_med.set_xticklabels([m.replace('_', ' ') for m in med_classes], rotation=45, ha='right')
    ax_med.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax_med.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
    
    st.pyplot(fig_med)

    # Risk distribution context with better styling
    st.markdown("#### 📋 Risk Interpretation Guide")
    st.success(f"""
    **Current Prediction: {risk:.3f}**
    
    **Risk Categories:**
    - 🟢 **Very Low Risk** (< 0.3): Excellent persistence expected
    - 🟡 **Low Risk** (0.3 - 0.8): Good persistence expected  
    - 🟠 **Average Risk** (0.8 - 1.2): Typical persistence pattern
    - 🔴 **High Risk** (> 1.2): May need additional support
    
    *Note: Most patients in the training data showed good medication persistence, which is why the model tends to predict lower risks.*
    """)

with tab2:
    # Global feature importance from CoxPH model
    st.markdown("#### 🌍 Global Feature Importance")
    
    try:
        # Get feature importance from CoxPH model (coefficients)
        if hasattr(cph, 'params_'):
            feature_importance = pd.DataFrame({
                'feature': cph.params_.index,
                'importance': abs(cph.params_.values)  # Absolute value of coefficients
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
            
            fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
            bars = ax_importance.barh(range(len(feature_importance)), feature_importance['importance'], color=colors)
            
            # Add labels and formatting
            ax_importance.set_yticks(range(len(feature_importance)))
            ax_importance.set_yticklabels(feature_importance['feature'])
            ax_importance.set_xlabel('Absolute Coefficient Value')
            ax_importance.set_title('Top 10 Most Important Features')
            ax_importance.grid(alpha=0.3)
            
            st.pyplot(fig_importance)
            
    except Exception as e:
        st.warning(f"Could not generate global feature importance: {e}")
            

# Add clinical interpretation section
st.markdown("---")
clinical_text = generate_clinical_interpretation(model_variant, raw_row, risk, include_ses)
st.markdown(clinical_text)

st.info(
    "Tip: Use the sidebar to switch model variant (Adjusted vs Unadjusted), set SES, medication class, "
    "and add comorbid meds. All visualizations and interpretations update in real-time."
)
