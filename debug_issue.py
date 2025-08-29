import warnings
import pandas as pd
import numpy as np
import joblib

# Suppress ScriptRunContext warning from lifelines
warnings.filterwarnings("ignore", message="missing ScriptRunContext")
warnings.filterwarnings("ignore", message="ScriptRunContext")

# Load the model to test what happens with age=58
try:
    cph = joblib.load("cph_a.pkl")
    print("Model loaded successfully")
    
    # Create a test case with age=58
    test_data = pd.DataFrame({
        'age': [58],
        'sex_Male': [1],  # Assuming male
        'adhd_med_class_non_stimulant': [0],
        'adhd_med_class_methylphenidate_second': [0],
        'adhd_med_class_amphetamine_second': [0],
        # Add other features with default values
    })
    
    # Add all expected columns with 0 values
    expected_features = joblib.load("feats_a.pkl")
    for feat in expected_features:
        if feat not in test_data.columns:
            test_data[feat] = 0
    
    # Reorder columns to match model expectation
    test_data = test_data[expected_features]
    
    # Add the required columns for prediction
    test_data["days_on_first_med"] = 0.0
    test_data["switched_med"] = 0
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Test data columns: {test_data.columns.tolist()}")
    
    # Test partial hazard prediction
    ph = cph.predict_partial_hazard(test_data)
    print(f"Partial hazard raw: {ph.values}")
    print(f"Partial hazard value: {ph.values.ravel()[0]}")
    
    # Test survival function
    sf = cph.predict_survival_function(test_data)
    print(f"Survival function shape: {sf.shape}")
    print(f"Survival values: {sf.iloc[:, 0].values[:10]}")  # First 10 values
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
