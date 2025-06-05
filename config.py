"""
Configuration File for Grasso et al. (2023) Reproduction Study
=============================================================

CONFIGURATION MANAGEMENT:
Centralizes all configurable parameters for the reproduction study,
enabling easy modification of analysis parameters without changing core code.

PARAMETERS:
- Data processing parameters (file paths, quality filters)
- Model training parameters (Random Forest hyperparameters)
- Analysis parameters (evaluation metrics, visualization settings)
- Preprocessing parameters (scaling, transformation options)

AUTHOR: Mehak Wadhwa
INSTITUTION: Fordham University
RESEARCH MENTOR: Dr. Joshua Schrier
"""

import os

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    # Primary data file (place in same directory as scripts)
    'default_data_files': [
        'sb2c00328_si_011.csv'       # Primary dataset file
    ],
    
    # Alternative locations (if organized in subdirectory)
    'alternative_data_paths': [
        'data/sb2c00328_si_011.csv'
    ],
    
    # Excel file settings (if using .xlsx version)
    'excel_sheet_name': 'Library_w_Bins_and_WA',
    
    # Grasso quality control criteria
    'min_signal_peptide_length': 10,    # Minimum functional SP length (amino acids)
    'max_signal_peptide_length': 40,    # Maximum functional SP length (amino acids)
    'min_wa_value': 1.0,                # Minimum valid WA score
    'max_wa_value': 10.0,               # Maximum valid WA score
    
    # Data handling
    'missing_value_strategy': 'fill_zero',  # Fill missing feature values with 0
    'random_state': 42                       # Random seed for reproducibility
}

# ============================================================================
# MODEL CONFIGURATION - EXACT GRASSO PARAMETERS
# ============================================================================

MODEL_CONFIG = {
    # Random Forest hyperparameters (exact values from Grasso et al. study)
    'n_estimators': 75,              # Number of trees in ensemble
    'max_depth': 25,                 # Maximum depth of trees
    'min_samples_split': 0.001,      # Minimum samples to split internal node
    'min_samples_leaf': 0.0001,      # Minimum samples in leaf node
    'max_features': 156,             # Maximum features per split (all features)
    'random_state': 42,              # Random seed for model training
    'n_jobs': -1,                    # Use all available CPU cores
    
    # Cross-validation settings
    'cross_validation_folds': 5      # Number of CV folds for stability assessment
}

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

PREPROCESSING_CONFIG = {
    # Feature scaling parameters
    'scaling_method': 'standard',        # StandardScaler (Z-score normalization)
    'scaling_threshold': 100.0,          # Apply scaling if max/min range ratio > threshold
    'auto_scaling': True,                # Automatically determine need for scaling
    
    # Target transformation parameters
    'target_transformation': 'auto',    # Auto-detect need for transformation
    'skewness_threshold': 1.0,          # Apply transformation if |skewness| > threshold
    
    # Feature selection (disabled for exact reproduction)
    'feature_selection': False
}

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

ANALYSIS_CONFIG = {
    # Evaluation metrics
    'primary_metric': 'mse',            # Primary metric for model comparison
    'secondary_metrics': ['mae', 'r2'], # Additional metrics to calculate
    'calculate_cross_validation': True, # Whether to run cross-validation
    
    # Performance benchmarks (Grasso et al. original results)
    'target_train_mse': 1.75,          # Original training MSE
    'target_test_mse': 1.22,           # Original test MSE (primary benchmark)
    
    # Feature importance analysis
    'top_features_to_show': 15,        # Number of top features to display
    'biological_validation': True      # Perform biological relevance assessment
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

VISUALIZATION_CONFIG = {
    # Figure settings
    'figure_size': (14, 10),           # Default figure size (width, height)
    'figure_dpi': 300,                 # Resolution for saved figures
    'figure_format': 'png',            # Output format: 'png', 'pdf', 'svg'
    
    # Output files
    'main_results_filename': 'grasso_reproduction_results.png',
    'save_figures': True,              # Whether to save figures automatically
    'show_figures': True               # Whether to display figures
}

# ============================================================================
# BIOLOGICAL VALIDATION CONFIGURATION
# ============================================================================

BIOLOGICAL_CONFIG = {
    # Feature categories for biological validation
    'biological_feature_keywords': {
        'hydrophobicity': ['gravy', 'BomanInd'],
        'cleavage_specificity': ['-1_A', '-3_A', 'A_C', '-1_S', '-3_S'],
        'length_optimization': ['Length_SP', 'Length_N', 'Length_H', 'Length_C'],
        'charge_effects': ['Charge_SP', 'pI_C', 'pI_Ac'],
        'structural_properties': ['Helix_SP', 'Turn_SP', 'Sheet_C', 'flexibility_SP'],
        'energy_stability': ['mfe_SP', 'mfe_N', 'mfe_H', 'mfe_C']
    },
    
    # Validation thresholds
    'min_biological_features_in_top10': 5,  # Minimum meaningful features in top 10
    'biological_relevance_threshold': 0.7   # Minimum fraction for good validation
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_config():
    """
    Validate configuration parameters for consistency and correctness.
    
    RETURNS:
    list: Validation results (empty if all checks pass)
    """
    validation_results = []
    
    # Validate data parameters
    if DATA_CONFIG['min_wa_value'] >= DATA_CONFIG['max_wa_value']:
        validation_results.append("Error: min_wa_value must be less than max_wa_value")
    
    if DATA_CONFIG['min_signal_peptide_length'] >= DATA_CONFIG['max_signal_peptide_length']:
        validation_results.append("Error: min_signal_peptide_length must be less than max_signal_peptide_length")
    
    # Validate model parameters
    if MODEL_CONFIG['n_estimators'] <= 0:
        validation_results.append("Error: n_estimators must be positive")
    
    if MODEL_CONFIG['max_depth'] <= 0:
        validation_results.append("Error: max_depth must be positive")
    
    # Validate preprocessing parameters
    if PREPROCESSING_CONFIG['scaling_threshold'] <= 1:
        validation_results.append("Warning: scaling_threshold <= 1 may cause excessive scaling")
    
    return validation_results

def print_config_summary():
    """
    Print a summary of current configuration settings.
    """
    print("CONFIGURATION SUMMARY")
    print("=" * 50)
    
    print("Data Configuration:")
    print(f"  Primary data file: {DATA_CONFIG['default_data_files'][0]}")
    print(f"  WA range: {DATA_CONFIG['min_wa_value']:.1f} - {DATA_CONFIG['max_wa_value']:.1f}")
    print(f"  SP length range: {DATA_CONFIG['min_signal_peptide_length']}-{DATA_CONFIG['max_signal_peptide_length']} aa")
    print()
    
    print("Model Configuration:")
    print(f"  Random Forest: {MODEL_CONFIG['n_estimators']} estimators, depth {MODEL_CONFIG['max_depth']}")
    print(f"  Random state: {MODEL_CONFIG['random_state']}")
    print()
    
    print("Analysis Configuration:")
    print(f"  Primary metric: {ANALYSIS_CONFIG['primary_metric'].upper()}")
    print(f"  Target test MSE: {ANALYSIS_CONFIG['target_test_mse']:.2f}")
    print(f"  Biological validation: {BIOLOGICAL_CONFIG['min_biological_features_in_top10']} min features")
    print()
    
    print("Output Configuration:")
    print(f"  Results figure: {VISUALIZATION_CONFIG['main_results_filename']}")
    print(f"  Figure DPI: {VISUALIZATION_CONFIG['figure_dpi']}")
    print(f"  Save figures: {VISUALIZATION_CONFIG['save_figures']}")
    print()

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Configuration validation and summary when run directly.
    """
    print("GRASSO REPRODUCTION STUDY - CONFIGURATION")
    print("=" * 50)
    print()
    
    # Validate configuration
    validation_results = validate_config()
    
    if validation_results:
        print("Configuration Issues:")
        for result in validation_results:
            print(f"  {result}")
        print()
    else:
        print("✓ Configuration validation passed")
        print()
    
    # Print configuration summary
    print_config_summary()
    
    print("Configuration ready for reproduction study execution")
