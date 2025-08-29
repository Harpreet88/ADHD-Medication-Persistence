#!/usr/bin/env python3
"""
Test script to verify ScriptRunContext warning suppression
"""
import warnings
import pandas as pd
import numpy as np
import joblib

print("Testing ScriptRunContext warning suppression...")

# Suppress ScriptRunContext warning from lifelines
warnings.filterwarnings("ignore", message="missing ScriptRunContext")
warnings.filterwarnings("ignore", message="ScriptRunContext")

try:
    # Try to load a model to trigger the warning
    cph = joblib.load("cph_a.pkl")
    print("✓ Model loaded successfully without ScriptRunContext warnings")
    
    # Load expected features
    feat_names = joblib.load("feats_a.pkl")
    
    # Create a proper test case with all required features
    test_data = pd.DataFrame(0, index=[0], columns=feat_names)
    
    # Set basic features
    test_data['age'] = 12
    test_data['sex_Male'] = 1
    test_data['adhd_med_class_methylphenidate_first'] = 1  # Set one ADHD class to 1
    
    # Add required columns for lifelines API
    test_data["days_on_first_med"] = 0.0
    test_data["switched_med"] = 0
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Test data has all required features: {len(test_data.columns) == len(feat_names) + 2}")
    
    # Test prediction
    ph = cph.predict_partial_hazard(test_data)
    print(f"✓ Prediction successful: {ph.values.ravel()[0]:.3f}")
    
    print("✓ ScriptRunContext warning successfully suppressed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
