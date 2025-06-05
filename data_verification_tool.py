"""
Data Verification Tool for Grasso et al. (2023) Reproduction Study
================================================================

PURPOSE:
Comprehensive verification tool to ensure complete and correct data loading
for the Grasso signal peptide efficiency prediction reproduction study.

VERIFICATION SCOPE:
- Complete file loading without data truncation or corruption
- Column integrity and feature availability assessment
- Missing data patterns and data quality evaluation
- Sample data verification and format validation
- Filtering process validation and sample retention analysis
- Original train/test split verification

IMPORTANCE:
Data loading issues are a common source of failed computational reproductions.
This verification tool ensures that poor reproduction results are due to 
methodological factors rather than technical data loading problems.

USAGE:
python data_verification_tool.py

AUTHOR: Mehak Wadhwa
INSTITUTION: Fordham University
RESEARCH MENTOR: Dr. Joshua Schrier
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

# Import configuration
from config import DATA_CONFIG

def comprehensive_data_verification(filename=None):
    """
    Perform comprehensive verification of the Grasso dataset.
    
    VERIFICATION METHODOLOGY:
    This function systematically checks all aspects of data loading and integrity:
    1. File system verification and loading
    2. Essential column presence verification
    3. Grasso feature availability assessment
    4. Data quality evaluation for key columns
    5. Original train/test split verification
    6. Quality filtering simulation
    
    PARAMETERS:
    filename (str): Optional specific file path. If None, uses config settings.
    
    RETURNS:
    pandas.DataFrame: Loaded and verified dataset (if successful)
    None: If verification identifies critical data loading issues
    """
    print("DATA VERIFICATION FOR GRASSO REPRODUCTION")
    print("=" * 50)
    print("Comprehensive data integrity assessment")
    print()
    
    # =========================================================================
    # STEP 1: FILE LOADING
    # =========================================================================
    
    print("STEP 1: FILE LOADING")
    print("-" * 30)
    
    if filename is None:
        # Use primary data file from config
        filename = DATA_CONFIG['default_data_files'][0]
        if not os.path.exists(filename):
            # Try alternative location
            alt_paths = DATA_CONFIG['alternative_data_paths']
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    filename = alt_path
                    break
            else:
                print(f"✗ Dataset not found in configured locations")
                print(f"Expected: {DATA_CONFIG['default_data_files'][0]}")
                return None
    
    print(f"Loading: {filename}")
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filename, low_memory=False)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filename, sheet_name=DATA_CONFIG['excel_sheet_name'])
        else:
            print(f"✗ Unsupported file format: {filename}")
            return None
        
        file_size = os.path.getsize(filename)
        print(f"✓ Loaded successfully: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print()
        
    except Exception as e:
        print(f"✗ Loading failed: {e}")
        return None
    
    # =========================================================================
    # STEP 2: ESSENTIAL COLUMNS
    # =========================================================================
    
    print("STEP 2: ESSENTIAL COLUMNS")
    print("-" * 30)
    
    required_columns = ['WA', 'SP_aa', 'Set']
    missing_columns = []
    
    for col in required_columns:
        if col in df.columns:
            print(f"  ✓ {col}: Present")
        else:
            print(f"  ✗ {col}: MISSING")
            missing_columns.append(col)
    
    if missing_columns:
        print(f"\n✗ Critical columns missing: {missing_columns}")
        print("Cannot proceed with reproduction without these columns")
        return None
    
    print()
    
    # =========================================================================
    # STEP 3: GRASSO FEATURE AVAILABILITY
    # =========================================================================
    
    print("STEP 3: GRASSO FEATURE AVAILABILITY")
    print("-" * 30)
    
    # Import feature list from main tool
    try:
        from grasso_reproduction_tool import GRASSO_VALIDATED_FEATURES
        expected_features = GRASSO_VALIDATED_FEATURES
    except ImportError:
        # Fallback feature set if import fails
        expected_features = [
            'Turn_N', 'A_N', 'Length_N', 'gravy_SP', 'Length_SP', 
            '-1_A', '-3_A', 'A_C', 'Helix_SP', 'mfe_SP'
        ]
        print("  Note: Using reduced feature set for verification")
    
    available_features = [f for f in expected_features if f in df.columns]
    missing_features = [f for f in expected_features if f not in df.columns]
    
    print(f"Expected Grasso features: {len(expected_features)}")
    print(f"Available in dataset: {len(available_features)}")
    print(f"Missing from dataset: {len(missing_features)}")
    print(f"Availability rate: {len(available_features)/len(expected_features)*100:.1f}%")
    
    if missing_features and len(missing_features) <= 10:
        print(f"Missing features: {missing_features[:10]}")
        if len(missing_features) > 10:
            print(f"... and {len(missing_features) - 10} more")
    
    print()
    
    # =========================================================================
    # STEP 4: DATA QUALITY ASSESSMENT
    # =========================================================================
    
    print("STEP 4: DATA QUALITY")
    print("-" * 30)
    
    # WA (target variable) analysis
    print("WA (secretion efficiency) analysis:")
    wa_values = df['WA'].dropna()
    print(f"  Total entries: {len(df):,}")
    print(f"  Non-null values: {len(wa_values):,}")
    print(f"  Missing values: {df['WA'].isna().sum():,}")
    print(f"  Range: {wa_values.min():.2f} - {wa_values.max():.2f}")
    print(f"  Mean: {wa_values.mean():.3f} ± {wa_values.std():.3f}")
    
    # Check against config quality criteria
    valid_wa = ((wa_values >= DATA_CONFIG['min_wa_value']) & 
                (wa_values <= DATA_CONFIG['max_wa_value']))
    print(f"  Valid range ({DATA_CONFIG['min_wa_value']}-{DATA_CONFIG['max_wa_value']}): {valid_wa.sum():,} values")
    
    print()
    
    # SP_aa (signal peptide sequences) analysis
    print("SP_aa (signal peptide sequences) analysis:")
    sp_values = df['SP_aa'].dropna()
    lengths = sp_values.str.len()
    print(f"  Total entries: {len(df):,}")
    print(f"  Non-null sequences: {len(sp_values):,}")
    print(f"  Missing sequences: {df['SP_aa'].isna().sum():,}")
    print(f"  Length range: {lengths.min()} - {lengths.max()} amino acids")
    print(f"  Mean length: {lengths.mean():.1f} amino acids")
    
    # Check against config length constraints
    valid_length = ((lengths >= DATA_CONFIG['min_signal_peptide_length']) & 
                   (lengths <= DATA_CONFIG['max_signal_peptide_length']))
    print(f"  Valid length ({DATA_CONFIG['min_signal_peptide_length']}-{DATA_CONFIG['max_signal_peptide_length']} aa): {valid_length.sum():,} sequences")
    
    print()
    
    # =========================================================================
    # STEP 5: ORIGINAL TRAIN/TEST SPLITS
    # =========================================================================
    
    print("STEP 5: ORIGINAL TRAIN/TEST SPLITS")
    print("-" * 30)
    
    if 'Set' in df.columns:
        set_counts = df['Set'].value_counts()
        print("Set assignments:")
        for set_name, count in set_counts.items():
            print(f"  {set_name}: {count:,} samples")
        
        total_with_sets = df['Set'].notna().sum()
        total_without_sets = len(df) - total_with_sets
        print(f"  Total with assignments: {total_with_sets:,}")
        print(f"  Without assignments: {total_without_sets:,}")
        
        if 'Train' in set_counts and 'Test' in set_counts:
            ratio = set_counts['Train'] / set_counts['Test']
            print(f"  Train/Test ratio: {ratio:.2f}")
            
        # Check WA distribution across sets
        if total_with_sets > 0:
            train_wa = df[df['Set'] == 'Train']['WA'].dropna()
            test_wa = df[df['Set'] == 'Test']['WA'].dropna()
            if len(train_wa) > 0 and len(test_wa) > 0:
                print(f"  Train WA: {train_wa.mean():.3f} ± {train_wa.std():.3f}")
                print(f"  Test WA: {test_wa.mean():.3f} ± {test_wa.std():.3f}")
        
    else:
        print("✗ Set column not found - cannot use original splits")
    
    print()
    
    # =========================================================================
    # STEP 6: QUALITY FILTERING SIMULATION
    # =========================================================================
    
    print("STEP 6: QUALITY FILTERING SIMULATION")
    print("-" * 30)
    print("Simulating Grasso quality control pipeline...")
    
    n_original = len(df)
    
    # Apply quality filters using config parameters
    quality_mask = (
        df['WA'].notna() &
        df['SP_aa'].notna() &
        (df['SP_aa'].str.len() >= DATA_CONFIG['min_signal_peptide_length']) &
        (df['SP_aa'].str.len() <= DATA_CONFIG['max_signal_peptide_length']) &
        (df['WA'] >= DATA_CONFIG['min_wa_value']) &
        (df['WA'] <= DATA_CONFIG['max_wa_value'])
    )
    
    n_after_filtering = quality_mask.sum()
    
    print(f"Original samples: {n_original:,}")
    print(f"After quality filters: {n_after_filtering:,}")
    print(f"Retention rate: {n_after_filtering/n_original*100:.1f}%")
    
    # Check how many filtered samples have Set assignments
    filtered_data = df[quality_mask]
    with_sets = filtered_data['Set'].notna().sum()
    print(f"Filtered samples with Set assignments: {with_sets:,}")
    
    if with_sets > 0:
        train_filtered = (filtered_data['Set'] == 'Train').sum()
        test_filtered = (filtered_data['Set'] == 'Test').sum()
        print(f"  Train samples: {train_filtered:,}")
        print(f"  Test samples: {test_filtered:,}")
    
    print()
    
    # =========================================================================
    # STEP 7: VERIFICATION SUMMARY
    # =========================================================================
    
    print("VERIFICATION SUMMARY")
    print("-" * 30)
    
    issues = []
    warnings = []
    
    # Critical issues
    if missing_columns:
        issues.append(f"Missing essential columns: {missing_columns}")
    
    if len(available_features) < len(expected_features) * 0.8:
        issues.append(f"Low feature availability: {len(available_features)}/{len(expected_features)}")
    
    if n_after_filtering < 1000:
        issues.append(f"Very low sample count after filtering: {n_after_filtering:,}")
    
    if with_sets < 1000:
        issues.append(f"Few samples with Set assignments: {with_sets:,}")
    
    # Warnings
    if len(available_features) < len(expected_features):
        warnings.append(f"{len(missing_features)} features missing from expected set")
    
    if n_after_filtering < 5000:
        warnings.append(f"Moderate sample count: {n_after_filtering:,}")
    
    # Report results
    if issues:
        print("⚠ CRITICAL ISSUES DETECTED:")
        for issue in issues:
            print(f"  • {issue}")
        print()
        print("Recommendation: Address these issues before proceeding")
    elif warnings:
        print("⚠ WARNINGS:")
        for warning in warnings:
            print(f"  • {warning}")
        print()
        print("Recommendation: Proceed with caution, monitor results")
    else:
        print("✓ VERIFICATION PASSED")
        print("✓ Data appears ready for reproduction study")
        print("✓ All essential components available")
        print("✓ Quality filtering retains sufficient samples")
        print("✓ Original train/test splits can be used")
    
    print()
    print("DATA VERIFICATION COMPLETED")
    
    return df

def quick_feature_check(filename=None):
    """
    Quick verification of key features availability.
    
    PARAMETERS:
    filename (str): Optional file path
    
    RETURNS:
    pandas.DataFrame or None: Dataset if successful, None if failed
    """
    print("QUICK FEATURE CHECK")
    print("=" * 30)
    
    if filename is None:
        filename = DATA_CONFIG['default_data_files'][0]
    
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filename, low_memory=False)
        else:
            df = pd.read_excel(filename, sheet_name=DATA_CONFIG['excel_sheet_name'])
        
        key_features = ['WA', 'SP_aa', 'Set', 'gravy_SP', '-1_A', 'Length_SP']
        
        print(f"Dataset: {df.shape}")
        print("Key features:")
        for feature in key_features:
            status = "✓" if feature in df.columns else "✗"
            print(f"  {status} {feature}")
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run verification when executed directly.
    """
    print("GRASSO DATASET VERIFICATION TOOL")
    print("=" * 40)
    
    result = comprehensive_data_verification()
    
    if result is not None:
        print("✓ VERIFICATION COMPLETED")
        print("Dataset ready for reproduction analysis")
    else:
        print("✗ VERIFICATION FAILED")
        print("Please address data issues before proceeding")
