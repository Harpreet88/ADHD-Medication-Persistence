# ADHD Medication Persistence Project  
[![ADHD Illustration](https://mind.help/wp-content/uploads/2022/08/Attention-Deficit-Hyperactivity-Disorder-ADHD.jpg)](https://adhd-medicine-persistence.streamlit.app)  
*(Click the image to open the live Streamlit app)*

##  Overview  
This project explores **medication persistence in ADHD patients** using **survival analysis** and interpretable machine learning techniques. It simulates electronic medical record (EMR) data and evaluates factors that influence how long patients remain on their first prescribed medication.  

Two modeling approaches are compared:  
- **Cox Proportional Hazards (CoxPH)**  
- **DeepSurv (Neural Network Survival Model)**  

The project also includes a **Streamlit dashboard** for interactive exploration of survival predictions and feature importance (via SHAP values).  

---

##  Objectives  
- Simulate a realistic ADHD medication dataset.  
- Assess how demographic, clinical, and socio-economic factors affect treatment persistence.  
- Compare traditional statistical models (CoxPH) with deep learning approaches (DeepSurv).  
- Visualize confounding effects (e.g., missing socio-economic status variables).  
- Provide an interactive dashboard for clinicians and researchers.  

---

##  Tech Stack  
- **Python**: pandas, numpy, scikit-learn, lifelines, PyTorch, SHAP  
- **Dashboard**: Streamlit, Plotly  
- **Deployment/Tools**: Streamlit, AWS (optional), joblib for model persistence  

---

##  Features  
- **Survival Modeling**: CoxPH vs DeepSurv comparison with Concordance Index evaluation.  
- **Confounding Experiments**: SES adjustment vs unadjusted models.  
- **Explainability**: SHAP plots for feature contributions.  
- **Interactive Dashboard**:  
  - Upload patient-level data  
  - View survival curves and hazard ratios  
  - Explore feature importance dynamically  

---

##  Getting Started  

### 1. Clone Repository  
```bash
git clone https://github.com/Harpreet88/ADHD-Medication-Persistence.git
cd ADHD-Medication-Persistence
```
---

### 2. Install requirements
```bash
pip install -r requirements.txt
```
---
### 3. Run

```bash
streamlit run ADHD_app.py
```
---

## Dashboard
**Try the live dashboard here:** [ADHD Medicine Persistence App](https://adhd-medicine-persistence.streamlit.app)  
