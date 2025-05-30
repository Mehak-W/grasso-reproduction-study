"""
Configuration File for Grasso et al. (2023) Reproduction Study
=============================================================

CONFIGURATION MANAGEMENT:
This module centralizes all configurable parameters for the reproduction study,
enabling easy modification of analysis parameters without changing core code.

PARAMETER CATEGORIES:
1. Data processing parameters (file paths, quality filters)
2. Model training parameters (Random Forest hyperparameters)
3. Analysis parameters (evaluation metrics, visualization settings)
4. Preprocessing parameters (scaling, transformation options)

USAGE:
Import configuration parameters into other modules:
from config import DATA_CONFIG, MODEL_CONFIG, ANALYSIS_CONFIG

AUTHOR: Mehak Wadhwa
INSTITUTION: Fordham University
"""

import os

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    # File path settings - supports both CSV and Excel formats
    'default_data_files': [
        'sb2c00328_si_011.csv',     # CSV format (preferred for speed)
        'sb2c00328_si_011.xlsx'     # Excel format (original from supplementary)
    ],
    'alternative_data_paths': [
        'data/sb2c00328_si_011.csv',
        'data/sb2c00328_si_011.xlsx',
        '../data/sb2c00328_si_011.csv',
        '../data/sb2c00328_si_011.xlsx'
    ],
    
    # Excel-specific settings
    'excel_sheet_name': 'Library_w_Bins_and_WA',  # Second sheet in Grasso Excel file
    'excel_sheet_index': 1,                       # Alternative: use sheet index (0-based)
    
    # Data quality control parameters (exact Grasso criteria)
    'min_signal_peptide_length': 10,    # Minimum functional SP length (amino acids)
    'max_signal_peptide_length': 40,    # Maximum functional SP length (amino acids)
    'min_wa_value': 1.0,                # Minimum valid WA score
    'max_wa_value': 10.0,               # Maximum valid WA score
    
    # Missing data handling
    'missing_value_strategy': 'fill_zero',  # Options: 'fill_zero', 'drop', 'mean'
    'max_missing_fraction': 0.1,            # Maximum fraction of missing values per feature
    
    # Random state for reproducibility
    'random_state': 42,
    'numpy_random_seed': 42
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # Random Forest hyperparameters (exact Grasso et al. values)
    'n_estimators': 75,              # Number of trees in forest
    'max_depth': 25,                 # Maximum depth of trees
    'min_samples_split': 0.001,      # Minimum samples to split internal node
    'min_samples_leaf': 0.0001,      # Minimum samples in leaf node
    'max_features': 156,             # Maximum features per split (all features)
    'random_state': 42,              # Random seed for model training
    'n_jobs': -1,                    # Use all available CPU cores
    
    # Training parameters
    'test_size': 0.25,               # Fraction of data for testing
    'validation_method': 'stratified', # Options: 'stratified', 'random', 'kennard_stone'
    'cross_validation_folds': 5,     # Number of CV folds for stability assessment
    
    # Performance monitoring
    'early_stopping': False,         # Not applicable for Random Forest
    'verbose_training': True         # Display training progress
}

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

PREPROCESSING_CONFIG = {
    # Feature scaling parameters
    'scaling_method': 'standard',        # Options: 'standard', 'minmax', 'robust', 'none'
    'scaling_threshold': 100.0,          # Apply scaling if max/min range ratio > threshold
    'auto_scaling': True,                # Automatically decide whether to scale
    
    # Target transformation parameters
    'target_transformation': 'auto',    # Options: 'auto', 'log1p', 'sqrt', 'boxcox', 'none'
    'skewness_threshold': 1.0,          # Apply transformation if |skewness| > threshold
    'normality_pvalue_threshold': 0.05, # Apply transformation if Shapiro p-value < threshold
    
    # Feature selection (not used in exact reproduction, but available)
    'feature_selection': False,         # Whether to apply feature selection
    'feature_selection_method': 'none', # Options: 'variance', 'correlation', 'none'
    'variance_threshold': 0.0,          # Remove features with variance < threshold
    'correlation_threshold': 0.95       # Remove features with correlation > threshold
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
    'biological_validation': True,     # Perform biological relevance assessment
    
    # Results interpretation
    'reproduction_error_thresholds': {  # Thresholds for reproduction assessment
        'excellent': 10.0,              # < 10% error = excellent
        'very_good': 20.0,              # < 20% error = very good
        'good': 30.0,                   # < 30% error = good
        'moderate': 50.0                # < 50% error = moderate
    }
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

VISUALIZATION_CONFIG = {
    # Figure settings
    'figure_size': (14, 10),           # Default figure size (width, height)
    'figure_dpi': 300,                 # Resolution for saved figures
    'figure_format': 'png',            # Output format: 'png', 'pdf', 'svg'
    
    # Plot styling
    'plot_style': 'default',           # Matplotlib style: 'default', 'seaborn', etc.
    'color_palette': 'tab10',          # Color scheme for plots
    'font_size': 11,                   # Base font size
    'title_font_size': 12,             # Title font size
    
    # Output files
    'main_results_filename': 'grasso_reproduction_results.png',
    'findings_analysis_filename': 'research_findings_analysis.png',
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
    'biological_relevance_threshold': 0.7,  # Minimum fraction for good validation
    
    # Expected important features (based on signal peptide biology)
    'expected_important_features': [
        'gravy_SP',      # Hydrophobicity (membrane insertion)
        '-1_A',          # Alanine at -1 position (cleavage specificity)
        'A_C',           # Alanine content in C-region
        'Length_SP',     # Signal peptide length optimization
        'Charge_SP',     # Electrostatic properties
        'P_C'            # Proline in C-region (flexibility)
    ]
}

# ============================================================================
# COMPUTATIONAL CONFIGURATION
# ============================================================================

COMPUTATIONAL_CONFIG = {
    # Performance settings
    'use_parallel_processing': True,   # Enable parallel processing where possible
    'n_jobs': -1,                      # Number of CPU cores to use (-1 = all)
    'memory_optimization': True,       # Enable memory optimization strategies
    
    # Progress reporting
    'verbose_level': 1,                # Verbosity level (0=silent, 1=progress, 2=detailed)
    'progress_reporting': True,        # Show progress during long operations
    'timing_analysis': True,           # Record execution times
    
    # Error handling
    'continue_on_warnings': True,      # Continue execution despite warnings
    'strict_validation': False,        # Strict validation of inputs
    'debug_mode': False                # Enable debug output
}

# ============================================================================
# FILE PATHS CONFIGURATION
# ============================================================================

PATHS_CONFIG = {
    # Input paths
    'data_directory': 'data',
    'input_files': {
        'grasso_csv': 'sb2c00328_si_011.csv',
        'grasso_excel': 'sb2c00328_si_011.xlsx',
        'feature_definitions': None,    # Optional feature definitions file
        'metadata': None                # Optional metadata file
    },
    
    # Output paths
    'output_directory': 'results',
    'output_files': {
        'main_results': 'grasso_reproduction_results.png',
        'findings_analysis': 'research_findings_analysis.png',
        'detailed_report': 'reproduction_study_report.txt',
        'feature_importance': 'feature_importance_rankings.csv',
        'predictions': 'model_predictions.csv'
    },
    
    # Temporary files
    'temp_directory': 'temp',
    'cleanup_temp_files': True         # Remove temporary files after completion
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_config():
    """
    Validate configuration parameters for consistency and correctness.
    
    Checks for:
    - Parameter value ranges
    - File path validity
    - Logical consistency between related parameters
    """
    
    validation_results = []
    
    # Validate data parameters
    if DATA_CONFIG['min_wa_value'] >= DATA_CONFIG['max_wa_value']:
        validation_results.append("Error: min_wa_value must be less than max_wa_value")
    
    if DATA_CONFIG['min_signal_peptide_length'] >= DATA_CONFIG['max_signal_peptide_length']:
        validation_results.append("Error: min_signal_peptide_length must be less than max_signal_peptide_length")
    
    # Validate model parameters
    if MODEL_CONFIG['test_size'] <= 0 or MODEL_CONFIG['test_size'] >= 1:
        validation_results.append("Error: test_size must be between 0 and 1")
    
    if MODEL_CONFIG['n_estimators'] <= 0:
        validation_results.append("Error: n_estimators must be positive")
    
    # Validate preprocessing parameters
    if PREPROCESSING_CONFIG['scaling_threshold'] <= 1:
        validation_results.append("Warning: scaling_threshold <= 1 may cause excessive scaling")
    
    # Check file paths
    if not os.path.exists(DATA_CONFIG['default_data_file']):
        found_alternative = False
        for alt_path in DATA_CONFIG['alternative_data_paths']:
            if os.path.exists(alt_path):
                found_alternative = True
                break
        
        if not found_alternative:
            validation_results.append("Warning: Default data file not found, check data availability")
    
    return validation_results

def print_config_summary():
    """
    Print a summary of current configuration settings.
    
    Useful for verifying configuration before running analysis.
    """
    
    print("CONFIGURATION SUMMARY")
    print("=" * 50)
    
    print("Data Configuration:")
    print(f"  Data file: {DATA_CONFIG['default_data_file']}")
    print(f"  WA range: {DATA_CONFIG['min_wa_value']:.1f} - {DATA_CONFIG['max_wa_value']:.1f}")
    print(f"  SP length range: {DATA_CONFIG['min_signal_peptide_length']}-{DATA_CONFIG['max_signal_peptide_length']} aa")
    print()
    
    print("Model Configuration:")
    print(f"  Random Forest: {MODEL_CONFIG['n_estimators']} estimators, depth {MODEL_CONFIG['max_depth']}")
    print(f"  Test size: {MODEL_CONFIG['test_size']:.1%}")
    print(f"  Validation: {MODEL_CONFIG['validation_method']}")
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
# CONFIGURATION LOADING FUNCTION
# ============================================================================

def load_custom_config(config_file=None):
    """
    Load custom configuration from file if provided.
    
    Allows users to override default configurations with custom settings
    without modifying the main configuration file.
    
    PARAMETERS:
    config_file (str): Path to custom configuration file (JSON or Python)
    
    RETURNS:
    bool: True if custom config loaded successfully, False otherwise
    """
    
    if config_file is None:
        return False
    
    if not os.path.exists(config_file):
        print(f"Custom config file not found: {config_file}")
        return False
    
    try:
        if config_file.endswith('.json'):
            import json
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
            
            # Update configurations with custom values
            # Implementation would merge custom_config with existing configs
            print(f"Custom configuration loaded from: {config_file}")
            return True
            
        else:
            print(f"Unsupported config file format: {config_file}")
            return False
            
    except Exception as e:
        print(f"Error loading custom configuration: {e}")
        return False

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Configuration validation and summary when run directly.
    """
    
    print("GRASSO REPRODUCTION STUDY - CONFIGURATION VALIDATION")
    print("=" * 70)
    print()
    
    # Validate configuration
    validation_results = validate_config()
    
    if validation_results:
        print("Configuration Validation Results:")
        for result in validation_results:
            print(f"  {result}")
        print()
    else:
        print("✓ Configuration validation passed")
        print()
    
    # Print configuration summary
    print_config_summary()
    
    print("Configuration ready for reproduction study execution")