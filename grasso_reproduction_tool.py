"""
Grasso et al. (2023) Reproduction Study - Signal Peptide Efficiency Prediction
============================================================================

RESEARCH OBJECTIVE:
Computational reproduction of machine learning methodology from:
Grasso et al. (2023) "Signal Peptide Efficiency: From High-Throughput Data to 
Prediction and Explanation" published in ACS Synthetic Biology.

METHODOLOGY:
Reproduces the exact Random Forest approach using:
- Original Grasso train/test data partitions (Set column)
- 156 validated physicochemical features from Table S2
- Identical hyperparameters and preprocessing pipeline
- Comprehensive biological validation of learned relationships

RESULTS:
Achieved 97.6% reproduction accuracy (Test MSE: 1.191 vs Target: 1.22 WA²)
with biologically meaningful feature importance rankings.

AUTHOR: Mehak Wadhwa
INSTITUTION: Fordham University
RESEARCH MENTOR: Dr. Joshua Schrier
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Import configuration parameters
from config import (
    DATA_CONFIG, MODEL_CONFIG, ANALYSIS_CONFIG, 
    PREPROCESSING_CONFIG, VISUALIZATION_CONFIG, BIOLOGICAL_CONFIG
)

warnings.filterwarnings('ignore')

# ============================================================================
# GRASSO VALIDATED FEATURE SET (156 features from Table S2)
# ============================================================================

GRASSO_VALIDATED_FEATURES = [
    # N-region physicochemical features (amino-terminal region)
    'Turn_N', 'A_N', 'C_N', 'D_N', 'E_N', 'F_N', 'G_N', 'H_N', 'I_N', 
    'L_N', 'N_N', 'P_N', 'Q_N', 'S_N', 'T_N', 'V_N', 'W_N', 'Y_N',
    'Length_N', 'InstabilityInd_N', 'Aromaticity_N', 'flexibility_N', 
    'kytedoolittle_N', 'mfe_N',
    
    # H-region physicochemical features (hydrophobic core region)
    'Turn_H', 'G_H', 'M_H', 'N_H', 'P_H', 'Q_H', 'S_H', 'T_H', 'W_H', 'Y_H',
    'Length_H', 'InstabilityInd_H', 'BomanInd_H', 'mfe_H',
    
    # C-region physicochemical features (carboxy-terminal region)
    'Helix_C', 'Turn_C', 'Sheet_C', 'A_C', 'C_C', 'D_C', 'E_C', 'G_C', 
    'I_C', 'L_C', 'M_C', 'N_C', 'P_C', 'Q_C', 'R_C', 'S_C', 'T_C', 
    'V_C', 'W_C', 'Y_C', 'Length_C', 'pI_C', 'InstabilityInd_C', 
    'AliphaticInd_C', 'ez_C', 'gravy_C', 'mfe_C', 'CAI_RSCU_C',
    
    # Ac-region physicochemical features (post-cleavage amino acids)
    'Turn_Ac', 'Sheet_Ac', 'A_Ac', 'D_Ac', 'E_Ac', 'F_Ac', 'G_Ac', 
    'H_Ac', 'I_Ac', 'L_Ac', 'M_Ac', 'N_Ac', 'P_Ac', 'Q_Ac', 'R_Ac', 
    'S_Ac', 'T_Ac', 'V_Ac', 'MW_Ac', 'pI_Ac', 'InstabilityInd_Ac', 
    'BomanInd_Ac', 'ez_Ac', 'mfe_Ac', 'CAI_RSCU_Ac',
    
    # SP-region global features (overall signal peptide properties)
    'Helix_SP', 'Turn_SP', 'D_SP', 'E_SP', 'F_SP', 'G_SP', 'H_SP', 
    'L_SP', 'M_SP', 'N_SP', 'P_SP', 'Q_SP', 'S_SP', 'T_SP', 'W_SP', 'Y_SP',
    'Length_SP', 'Charge_SP', 'InstabilityInd_SP', 'flexibility_SP', 
    'gravy_SP', 'mfe_SP', '-35_mfe_SP', 'amyQ_mfe_SP', 'CAI_RSCU_SP',
    
    # Cleavage site specificity features (signal peptidase recognition)
    '-3_A', '-3_C', '-3_D', '-3_E', '-3_F', '-3_G', '-3_H', '-3_I', '-3_K',
    '-3_L', '-3_M', '-3_N', '-3_P', '-3_Q', '-3_R', '-3_S', '-3_T', '-3_V',
    '-3_W', '-3_Y', '-1_A', '-1_C', '-1_D', '-1_E', '-1_F', '-1_G', '-1_H',
    '-1_I', '-1_K', '-1_L', '-1_M', '-1_N', '-1_P', '-1_Q', '-1_R', '-1_S',
    '-1_T', '-1_V', '-1_W', '-1_Y'
]

# ============================================================================
# DATA LOADING
# ============================================================================

def load_grasso_data():
    """
    Load Grasso dataset from configured file path.
    
    RETURNS:
    pandas.DataFrame: Complete Grasso experimental dataset
    """
    print("LOADING GRASSO DATASET")
    print("-" * 40)
    
    # Use primary data file
    filename = DATA_CONFIG['default_data_files'][0]  # sb2c00328_si_011.csv
    
    print(f"Loading: {filename}")
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filename, low_memory=False)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filename, sheet_name=DATA_CONFIG['excel_sheet_name'])
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        print(f"✓ Successfully loaded: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        return df
        
    except FileNotFoundError:
        print(f"ERROR: {filename} not found in current directory")
        print("Please ensure the Grasso dataset file is available")
        raise
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        raise

# ============================================================================
# DATA QUALITY CONTROL
# ============================================================================

def apply_quality_filters(df):
    """
    Apply Grasso quality control filters using exact criteria from original study.
    
    PARAMETERS:
    df (DataFrame): Raw Grasso dataset
    
    RETURNS:
    DataFrame: Quality-filtered dataset
    """
    print("APPLYING GRASSO QUALITY CONTROL FILTERS")
    print("-" * 50)
    
    print("Quality control criteria:")
    print(f"  Signal peptide length: {DATA_CONFIG['min_signal_peptide_length']}-{DATA_CONFIG['max_signal_peptide_length']} amino acids")
    print(f"  WA score range: {DATA_CONFIG['min_wa_value']}-{DATA_CONFIG['max_wa_value']}")
    print()
    
    original_size = len(df)
    
    # Apply Grasso filtering criteria
    valid_mask = (
        df['WA'].notna() &                          # Non-missing WA scores
        df['SP_aa'].notna() &                       # Non-missing sequences
        (df['SP_aa'].str.len() >= DATA_CONFIG['min_signal_peptide_length']) &
        (df['SP_aa'].str.len() <= DATA_CONFIG['max_signal_peptide_length']) &
        (df['WA'] >= DATA_CONFIG['min_wa_value']) &
        (df['WA'] <= DATA_CONFIG['max_wa_value'])
    )
    
    df_clean = df[valid_mask].copy()
    
    print(f"Quality filtering results:")
    print(f"  Original samples: {original_size:,}")
    print(f"  After filtering: {len(df_clean):,}")
    print(f"  Retention rate: {len(df_clean)/original_size*100:.1f}%")
    print()
    
    return df_clean

# ============================================================================
# ORIGINAL TRAIN/TEST SPLITS
# ============================================================================

def use_original_grasso_splits(df_clean):
    """
    Extract original Grasso train/test splits from 'Set' column.
    
    This is essential for accurate reproduction - uses exactly the same
    data partitions as the original study.
    
    PARAMETERS:
    df_clean (DataFrame): Quality-filtered dataset
    
    RETURNS:
    tuple: (X_train, X_test, y_train, y_test, available_features)
    """
    print("USING ORIGINAL GRASSO TRAIN/TEST SPLITS")
    print("-" * 50)
    print("Extracting exact data partitions from original study")
    print()
    
    # Filter to samples with Set assignments
    samples_with_sets = df_clean[df_clean['Set'].notna()].copy()
    
    print("Original split composition:")
    set_counts = samples_with_sets['Set'].value_counts()
    for set_name, count in set_counts.items():
        print(f"  {set_name}: {count:,} samples")
    print(f"  Total: {len(samples_with_sets):,}")
    print()
    
    # Create train and test datasets
    train_data = samples_with_sets[samples_with_sets['Set'] == 'Train'].copy()
    test_data = samples_with_sets[samples_with_sets['Set'] == 'Test'].copy()
    
    # Extract Grasso validated features
    available_features = [f for f in GRASSO_VALIDATED_FEATURES if f in df_clean.columns]
    missing_features = [f for f in GRASSO_VALIDATED_FEATURES if f not in df_clean.columns]
    
    print(f"Feature availability:")
    print(f"  Expected features: {len(GRASSO_VALIDATED_FEATURES)}")
    print(f"  Available features: {len(available_features)}")
    if missing_features:
        print(f"  Missing features: {len(missing_features)}")
    print()
    
    # Create feature matrices and targets
    X_train = train_data[available_features].fillna(0)
    X_test = test_data[available_features].fillna(0)
    y_train = train_data['WA'].values
    y_test = test_data['WA'].values
    
    print("Final dataset characteristics:")
    print(f"  Training: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"  Test: {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
    print(f"  Training WA: {y_train.mean():.3f} ± {y_train.std():.3f}")
    print(f"  Test WA: {y_test.mean():.3f} ± {y_test.std():.3f}")
    print()
    
    return X_train, X_test, y_train, y_test, available_features

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def apply_preprocessing(X_train, X_test, y_train, y_test):
    """
    Apply data preprocessing using configuration parameters.
    
    PARAMETERS:
    X_train, X_test: Feature matrices
    y_train, y_test: Target vectors
    
    RETURNS:
    tuple: Processed data and transformation information
    """
    print("APPLYING PREPROCESSING")
    print("-" * 40)
    
    # Feature scaling assessment
    if PREPROCESSING_CONFIG['auto_scaling']:
        feature_ranges = []
        for col in X_train.columns:
            col_range = X_train[col].max() - X_train[col].min()
            if col_range > 0:
                feature_ranges.append(col_range)
        
        if feature_ranges:
            scale_ratio = max(feature_ranges) / min(feature_ranges)
            
            if scale_ratio > PREPROCESSING_CONFIG['scaling_threshold']:
                print(f"Applying {PREPROCESSING_CONFIG['scaling_method']} scaling (scale ratio: {scale_ratio:.1f})")
                scaler = StandardScaler()
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train), 
                    columns=X_train.columns, 
                    index=X_train.index
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test), 
                    columns=X_test.columns, 
                    index=X_test.index
                )
                scaling_applied = True
            else:
                print("No scaling applied - features have similar scales")
                X_train_scaled = X_train
                X_test_scaled = X_test
                scaler = None
                scaling_applied = False
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            scaler = None
            scaling_applied = False
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        scaler = None
        scaling_applied = False
    
    # Target transformation assessment
    if PREPROCESSING_CONFIG['target_transformation'] == 'auto':
        skewness = stats.skew(y_train)
        if abs(skewness) > PREPROCESSING_CONFIG['skewness_threshold']:
            print(f"Applying log1p transformation (skewness: {skewness:.3f})")
            y_train_transformed = np.log1p(y_train)
            y_test_transformed = np.log1p(y_test)
            inverse_function = np.expm1
            transformation_applied = True
        else:
            print("No target transformation needed")
            y_train_transformed = y_train
            y_test_transformed = y_test
            inverse_function = None
            transformation_applied = False
    else:
        y_train_transformed = y_train
        y_test_transformed = y_test
        inverse_function = None
        transformation_applied = False
    
    print()
    return (X_train_scaled, X_test_scaled, y_train_transformed, y_test_transformed,
            scaler, inverse_function, scaling_applied, transformation_applied)

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_grasso_model(X_train, y_train):
    """
    Train Random Forest using exact Grasso hyperparameters.
    
    Uses the same hyperparameters reported in the original study
    for faithful methodology reproduction.
    
    PARAMETERS:
    X_train: Training feature matrix
    y_train: Training target vector
    
    RETURNS:
    RandomForestRegressor: Trained model
    """
    print("TRAINING RANDOM FOREST MODEL")
    print("-" * 40)
    print("Using Grasso et al. hyperparameters:")
    
    # Extract model parameters from config
    model_params = {
        'n_estimators': MODEL_CONFIG['n_estimators'],
        'max_depth': MODEL_CONFIG['max_depth'],
        'min_samples_split': MODEL_CONFIG['min_samples_split'],
        'min_samples_leaf': MODEL_CONFIG['min_samples_leaf'],
        'max_features': min(MODEL_CONFIG['max_features'], X_train.shape[1]),
        'random_state': MODEL_CONFIG['random_state'],
        'n_jobs': MODEL_CONFIG['n_jobs']
    }
    
    for param, value in model_params.items():
        print(f"  {param}: {value}")
    print()
    
    print("Training model...")
    start_time = time.time()
    
    model = RandomForestRegressor(**model_params)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")
    print()
    
    return model

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, 
                  inverse_function=None, original_y_train=None, original_y_test=None):
    """
    Comprehensive model evaluation with comparison to original benchmarks.
    
    PARAMETERS:
    model: Trained RandomForestRegressor
    X_train, X_test: Feature matrices
    y_train, y_test: Target vectors (possibly transformed)
    inverse_function: Function to reverse target transformation
    original_y_train, original_y_test: Original scale targets
    
    RETURNS:
    dict: Complete evaluation results
    """
    print("EVALUATING MODEL PERFORMANCE")
    print("-" * 40)
    
    # Generate predictions
    train_pred_transformed = model.predict(X_train)
    test_pred_transformed = model.predict(X_test)
    
    # Transform predictions back to original scale if needed
    if inverse_function is not None:
        train_pred = inverse_function(train_pred_transformed)
        test_pred = inverse_function(test_pred_transformed)
        y_train_eval = original_y_train
        y_test_eval = original_y_test
    else:
        train_pred = train_pred_transformed
        test_pred = test_pred_transformed
        y_train_eval = y_train
        y_test_eval = y_test
    
    # Calculate performance metrics
    train_mse = mean_squared_error(y_train_eval, train_pred)
    test_mse = mean_squared_error(y_test_eval, test_pred)
    train_r2 = r2_score(y_train_eval, train_pred)
    test_r2 = r2_score(y_test_eval, test_pred)
    train_mae = mean_absolute_error(y_train_eval, train_pred)
    test_mae = mean_absolute_error(y_test_eval, test_pred)
    
    # Cross-validation assessment
    if ANALYSIS_CONFIG['calculate_cross_validation']:
        cv_scores = cross_val_score(model, X_train, y_train, 
                                  cv=MODEL_CONFIG['cross_validation_folds'], 
                                  scoring='r2')
        cv_r2_mean = np.mean(cv_scores)
        cv_r2_std = np.std(cv_scores)
    else:
        cv_r2_mean = test_r2
        cv_r2_std = 0.0
    
    # Comparison with Grasso et al. targets
    target_train_mse = ANALYSIS_CONFIG['target_train_mse']
    target_test_mse = ANALYSIS_CONFIG['target_test_mse']
    
    relative_error = abs(test_mse - target_test_mse) / target_test_mse * 100
    
    print("Performance Results:")
    print(f"  Training MSE: {train_mse:.3f} WA² (Target: {target_train_mse:.2f})")
    print(f"  Test MSE: {test_mse:.3f} WA² (Target: {target_test_mse:.2f})")
    print(f"  Test R²: {test_r2:.3f}")
    print(f"  Reproduction error: {relative_error:.1f}%")
    
    # Residual analysis for model validation
    residuals = test_pred - y_test_eval
    print(f"  Prediction bias: {residuals.mean():.3f} WA (should be ~0)")
    print(f"  Residual std: {residuals.std():.3f} WA")
    print()
    
    results = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'relative_error_percent': relative_error,
        'train_predictions': train_pred,
        'test_predictions': test_pred,
        'y_train_eval': y_train_eval,
        'y_test_eval': y_test_eval,
        'target_train_mse': target_train_mse,
        'target_test_mse': target_test_mse
    }
    
    return results

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(model, feature_names):
    """
    Analyze Random Forest feature importance with biological validation.
    
    PARAMETERS:
    model: Trained RandomForestRegressor
    feature_names: List of feature names
    
    RETURNS:
    DataFrame: Feature importance rankings
    """
    print("ANALYZING FEATURE IMPORTANCE")
    print("-" * 40)
    
    # Calculate feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Display top features with biological assessment
    n_top = ANALYSIS_CONFIG['top_features_to_show']
    print(f"Top {n_top} most important features:")
    
    for i, (_, row) in enumerate(importance_df.head(n_top).iterrows(), 1):
        feature = row['Feature']
        importance = row['Importance']
        
        # Assess biological relevance
        is_biological = any(keyword.lower() in feature.lower() 
                          for category_features in BIOLOGICAL_CONFIG['biological_feature_keywords'].values()
                          for keyword in category_features)
        
        marker = "🧬" if is_biological else "❓"
        print(f"  {i:2d}. {marker} {feature:<20s} {importance:.4f}")
    
    print()
    return importance_df

# ============================================================================
# RESULTS VISUALIZATION
# ============================================================================

def create_results_visualization(results, feature_importance_df):
    """
    Generate comprehensive visualization of reproduction study results.
    
    PARAMETERS:
    results: Dictionary of evaluation results
    feature_importance_df: Feature importance rankings
    
    RETURNS:
    matplotlib.Figure: Results visualization
    """
    print("CREATING RESULTS VISUALIZATION")
    print("-" * 40)
    
    # Configure figure
    fig_size = VISUALIZATION_CONFIG['figure_size']
    dpi = VISUALIZATION_CONFIG['figure_dpi']
    
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    fig.suptitle('Grasso Reproduction Study Results', fontsize=16, fontweight='bold')
    
    # Panel 1: Prediction accuracy
    ax1 = axes[0, 0]
    y_test = results['y_test_eval']
    test_pred = results['test_predictions']
    
    ax1.scatter(y_test, test_pred, alpha=0.6, s=25, color='steelblue')
    
    min_val = min(min(y_test), min(test_pred))
    max_val = max(max(y_test), max(test_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax1.set_xlabel('Experimental WA')
    ax1.set_ylabel('Predicted WA')
    ax1.set_title(f'Prediction Accuracy\nR² = {results["test_r2"]:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Performance comparison
    ax2 = axes[0, 1]
    categories = ['Train MSE', 'Test MSE']
    achieved = [results['train_mse'], results['test_mse']]
    targets = [results['target_train_mse'], results['target_test_mse']]
    
    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width/2, achieved, width, label='This Study', alpha=0.8)
    ax2.bar(x + width/2, targets, width, label='Grasso et al.', alpha=0.8)
    
    ax2.set_ylabel('MSE (WA²)')
    ax2.set_title('Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Feature importance
    ax3 = axes[1, 0]
    top_features = feature_importance_df.head(8)
    y_pos = np.arange(len(top_features))
    
    ax3.barh(y_pos, top_features['Importance'], alpha=0.8, color='forestgreen')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([f[:15] for f in top_features['Feature']], fontsize=9)
    ax3.set_xlabel('Importance')
    ax3.set_title('Top Features')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Panel 4: Results summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary = f"""REPRODUCTION RESULTS

Performance:
• Test MSE: {results['test_mse']:.3f} WA²
• Target MSE: {results['target_test_mse']:.2f} WA²
• Error: {results['relative_error_percent']:.1f}%
• Test R²: {results['test_r2']:.3f}

Dataset:
• Train: {len(results['y_train_eval']):,} samples
• Test: {len(results['y_test_eval']):,} samples
• Features: {len(feature_importance_df)}

Top Feature:
• {feature_importance_df.iloc[0]['Feature']}
• Importance: {feature_importance_df.iloc[0]['Importance']:.3f}"""
    
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save figure
    filename = VISUALIZATION_CONFIG['main_results_filename']
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    if VISUALIZATION_CONFIG['show_figures']:
        plt.show()
    
    return fig

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def execute_grasso_reproduction():
    """
    Execute complete Grasso et al. (2023) reproduction study.
    
    WORKFLOW:
    1. Load and quality filter the Grasso dataset
    2. Extract original train/test splits
    3. Apply preprocessing pipeline
    4. Train Random Forest with exact hyperparameters
    5. Evaluate performance against published benchmarks
    6. Analyze feature importance with biological validation
    7. Generate comprehensive visualization
    
    RETURNS:
    dict: Complete reproduction study results
    """
    print("GRASSO REPRODUCTION STUDY")
    print("=" * 60)
    print("Computational reproduction of signal peptide efficiency prediction")
    print()
    
    start_time = time.time()
    
    try:
        # Load and prepare data
        df = load_grasso_data()
        df_clean = apply_quality_filters(df)
        
        # Extract original splits
        X_train, X_test, y_train, y_test, available_features = use_original_grasso_splits(df_clean)
        
        # Preprocessing
        (X_train_processed, X_test_processed, y_train_processed, y_test_processed,
         scaler, inverse_function, scaling_applied, transformation_applied) = apply_preprocessing(
            X_train, X_test, y_train, y_test)
        
        # Model training
        model = train_grasso_model(X_train_processed, y_train_processed)
        
        # Evaluation
        results = evaluate_model(model, X_train_processed, X_test_processed,
                               y_train_processed, y_test_processed,
                               inverse_function, y_train, y_test)
        
        # Feature importance analysis
        feature_importance_df = analyze_feature_importance(model, available_features)
        
        # Visualization
        fig = create_results_visualization(results, feature_importance_df)
        
        # Add additional results
        results['feature_importance'] = feature_importance_df
        results['model'] = model
        
        total_time = time.time() - start_time
        
        print("STUDY COMPLETED SUCCESSFULLY")
        print("-" * 40)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Test MSE: {results['test_mse']:.3f} WA² (Target: {results['target_test_mse']:.2f})")
        print(f"Reproduction accuracy: {100 - results['relative_error_percent']:.1f}%")
        
        return results
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run the reproduction study when script is executed directly.
    """
    results = execute_grasso_reproduction()
    
    if results is not None:
        print("\n✓ Reproduction study completed successfully!")
        print("Check the generated visualization for detailed results.")
    else:
        print("\n✗ Reproduction study failed. Check error messages above.")
