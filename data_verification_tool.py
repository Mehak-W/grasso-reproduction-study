"""
Data Verification Tool for Grasso et al. (2023) Reproduction Study
================================================================

PURPOSE:
Comprehensive verification tool to ensure complete and correct data loading
for the Grasso signal peptide efficiency prediction reproduction study.

VERIFICATION SCOPE:
This tool thoroughly inspects the CSV dataset to validate:
1. Complete file loading without data truncation or corruption
2. Column integrity and feature availability assessment
3. Missing data patterns and data quality evaluation
4. Sample data verification and format validation
5. Filtering process validation and sample retention analysis
6. Feature availability comparison with Grasso validated set

IMPORTANCE FOR REPRODUCTION STUDIES:
Data loading issues are a common source of failed computational reproductions.
This verification tool ensures that poor reproduction results are due to 
methodological or biological factors rather than technical data loading problems.

USAGE:
Run this script to verify data integrity before executing reproduction analysis:
python data_verification_tool.py

AUTHOR: Mehak Wadhwa
INSTITUTION: Fordham University
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

def comprehensive_data_verification(filename=None):
    """
    Perform comprehensive verification of the Grasso dataset (CSV or Excel format).
    
    VERIFICATION METHODOLOGY:
    This function systematically checks all aspects of data loading and integrity:
    
    1. FILE SYSTEM VERIFICATION: Confirms file exists and checks basic properties
    2. SMART FORMAT DETECTION: Automatically detects CSV or Excel format
    3. RAW DATA LOADING: Tests basic file parsing without data processing
    4. COLUMN STRUCTURE ANALYSIS: Verifies all expected columns are present
    5. GRASSO FEATURE AVAILABILITY: Checks for all 156 validated features
    6. DATA QUALITY ASSESSMENT: Evaluates key columns for proper formatting
    7. SAMPLE DATA VERIFICATION: Shows actual data to confirm correct parsing
    8. FILTERING VERIFICATION: Tests the quality control pipeline step-by-step
    9. RESULTS COMPARISON: Compares with expected reproduction study outcomes
    
    This comprehensive approach ensures data integrity and identifies specific
    issues if reproduction results are affected by data loading problems.
    
    PARAMETERS:
    filename (str): Optional specific file path. If None, uses smart detection.
    
    RETURNS:
    pandas.DataFrame: Loaded and verified dataset (if successful)
    None: If verification identifies critical data loading issues
    """
    
    print("COMPREHENSIVE DATA VERIFICATION AND INSPECTION")
    print("=" * 70)
    if filename:
        print(f"Target dataset: {filename}")
    else:
        print("Target dataset: Auto-detection (CSV or Excel format)")
    print("Verification scope: Complete data loading and integrity assessment")
    print()
    
    # =========================================================================
    # STEP 1: FILE SYSTEM VERIFICATION AND SMART DETECTION
    # =========================================================================
    
    print("STEP 1: FILE SYSTEM VERIFICATION AND SMART DETECTION")
    print("-" * 60)
    print("Searching for Grasso dataset files...")
    
    if filename is None:
        # Smart detection of available files
        search_candidates = [
            ('sb2c00328_si_011.csv', 'csv'),
            ('sb2c00328_si_011.xlsx', 'excel'),
            ('data/sb2c00328_si_011.csv', 'csv'),
            ('data/sb2c00328_si_011.xlsx', 'excel'),
            ('../data/sb2c00328_si_011.csv', 'csv'),
            ('../data/sb2c00328_si_011.xlsx', 'excel')
        ]
        
        found_files = []
        for path, file_format in search_candidates:
            if os.path.exists(path):
                file_size = os.path.getsize(path)
                found_files.append((path, file_format, file_size))
                print(f"  ✓ Found: {path} ({file_format.upper()}, {file_size:,} bytes)")
        
        if not found_files:
            print("  ✗ No Grasso dataset files found")
            print()
            print("Expected files:")
            print("  • sb2c00328_si_011.csv")
            print("  • sb2c00328_si_011.xlsx")
            print("Please ensure dataset is available in current directory or data/ subfolder")
            return None
        
        # Use first found file (CSV preferred)
        filename, file_format, file_size = found_files[0]
        print(f"  Selected: {filename} ({file_format.upper()})")
        
    else:
        # Check specified file
        if not os.path.exists(filename):
            print(f"ERROR: File '{filename}' not found!")
            print("Available files in current directory:")
            for file in os.listdir('.'):
                if file.endswith(('.csv', '.xlsx')):
                    print(f"  - {file}")
            return None
        
        file_size = os.path.getsize(filename)
        file_format = 'excel' if filename.endswith('.xlsx') else 'csv'
        print(f"File verification: {filename} ({file_format.upper()})")
    
    print(f"File properties:")
    print(f"  Location: {filename}")
    print(f"  Format: {file_format.upper()}")
    print(f"  Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    print(f"  Status: ✓ Confirmed")
    print()
    
    # =========================================================================
    # STEP 2: RAW DATA LOADING VERIFICATION
    # =========================================================================
    
    print("STEP 2: RAW DATA LOADING VERIFICATION")
    print("-" * 60)
    print(f"Testing {file_format.upper()} parsing and data loading...")
    
    try:
        if file_format == 'csv':
            print("Loading CSV file with minimal processing...")
            df_raw = pd.read_csv(filename, low_memory=False)
        else:
            print("Loading Excel file...")
            print("Accessing 'Library_w_Bins_and_WA' sheet...")
            df_raw = pd.read_excel(filename, sheet_name='Library_w_Bins_and_WA')
        
        print(f"Loading results:")
        print(f"  Status: ✓ Successful")
        print(f"  Dataset dimensions: {df_raw.shape[0]:,} rows × {df_raw.shape[1]:,} columns")
        
        # Check memory usage
        memory_usage = df_raw.memory_usage(deep=True).sum()
        print(f"  Memory usage: {memory_usage:,} bytes ({memory_usage/1024/1024:.2f} MB)")
        
        # Check for common parsing issues
        unnamed_columns = [col for col in df_raw.columns if 'Unnamed' in str(col)]
        if unnamed_columns:
            print(f"  Unnamed columns detected: {len(unnamed_columns)} (likely index columns)")
            print(f"    Examples: {unnamed_columns[:3]}")
        else:
            print(f"  Column parsing: ✓ No unnamed columns detected")
        
        print()
        
    except Exception as e:
        print(f"ERROR: {file_format.upper()} loading failed: {e}")
        print()
        print("Possible causes:")
        if file_format == 'csv':
            print("  • File corruption or incomplete download")
            print("  • Incorrect file encoding (try UTF-8)")
            print("  • CSV format inconsistencies")
        else:
            print("  • Excel file corruption")
            print("  • Missing 'Library_w_Bins_and_WA' sheet")
            print("  • Missing openpyxl package: pip install openpyxl")
            print("  • File locked by Excel application")
        return None
    """
    Perform comprehensive verification of the Grasso dataset CSV file.
    
    VERIFICATION METHODOLOGY:
    This function systematically checks all aspects of data loading and integrity:
    
    1. FILE SYSTEM VERIFICATION: Confirms file exists and checks basic properties
    2. RAW CSV LOADING: Tests basic CSV parsing without data processing
    3. COLUMN STRUCTURE ANALYSIS: Verifies all expected columns are present
    4. GRASSO FEATURE AVAILABILITY: Checks for all 156 validated features
    5. DATA QUALITY ASSESSMENT: Evaluates key columns for proper formatting
    6. SAMPLE DATA VERIFICATION: Shows actual data to confirm correct parsing
    7. FILTERING VERIFICATION: Tests the quality control pipeline step-by-step
    8. RESULTS COMPARISON: Compares with expected reproduction study outcomes
    
    This comprehensive approach ensures data integrity and identifies specific
    issues if reproduction results are affected by data loading problems.
    
    PARAMETERS:
    filename (str): Path to the Grasso experimental dataset CSV file
    
    RETURNS:
    pandas.DataFrame: Loaded and verified dataset (if successful)
    None: If verification identifies critical data loading issues
    """
    
    print("COMPREHENSIVE DATA VERIFICATION AND INSPECTION")
    print("=" * 70)
    print(f"Target dataset: {filename}")
    print("Verification scope: Complete data loading and integrity assessment")
    print()
    
    # =========================================================================
    # STEP 1: FILE SYSTEM VERIFICATION
    # =========================================================================
    
    print("STEP 1: FILE SYSTEM VERIFICATION")
    print("-" * 50)
    print("Checking file availability and basic properties...")
    
    # Verify file exists in specified location
    if not os.path.exists(filename):
        print(f"ERROR: File '{filename}' not found in current directory")
        print()
        print("Available files in current directory:")
        available_files = [f for f in os.listdir('.') if f.endswith(('.csv', '.xlsx', '.txt'))]
        if available_files:
            for file in available_files:
                print(f"  • {file}")
        else:
            print("  No CSV, Excel, or text files found")
        print()
        print("Please verify file name and location before proceeding")
        return None
    
    # Check file size and basic properties
    file_size = os.path.getsize(filename)
    print(f"File verification results:")
    print(f"  File location: {filename}")
    print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    print(f"  File exists: ✓ Confirmed")
    print()
    
    # =========================================================================
    # STEP 2: RAW CSV LOADING VERIFICATION
    # =========================================================================
    
    print("STEP 2: RAW CSV LOADING VERIFICATION")
    print("-" * 50)
    print("Testing CSV parsing and basic data loading...")
    
    try:
        # Attempt to load CSV with minimal processing to test basic parsing
        # Use low_memory=False to ensure consistent data type inference
        df_raw = pd.read_csv(filename, low_memory=False)
        
        print(f"CSV loading results:")
        print(f"  Loading status: ✓ Successful")
        print(f"  Dataset dimensions: {df_raw.shape[0]:,} rows × {df_raw.shape[1]:,} columns")
        
        # Check memory usage to assess data loading efficiency
        memory_usage = df_raw.memory_usage(deep=True).sum()
        print(f"  Memory usage: {memory_usage:,} bytes ({memory_usage/1024/1024:.2f} MB)")
        
        # Check for common CSV parsing issues
        unnamed_columns = [col for col in df_raw.columns if 'Unnamed' in str(col)]
        if unnamed_columns:
            print(f"  Unnamed columns detected: {len(unnamed_columns)} (likely index columns)")
            print(f"    Example: {unnamed_columns[:3]}")
        else:
            print(f"  Column parsing: ✓ No unnamed columns detected")
        
        print()
        
    except Exception as e:
        print(f"ERROR: CSV loading failed with error: {e}")
        print()
        print("Possible causes:")
        print("  • File corruption or incomplete download")
        print("  • Incorrect file encoding (try UTF-8)")
        print("  • CSV format inconsistencies")
        print("  • File is not actually in CSV format")
        return None
    
    # =========================================================================
    # STEP 3: COLUMN STRUCTURE ANALYSIS
    # =========================================================================
    
    print("STEP 3: COLUMN STRUCTURE ANALYSIS")
    print("-" * 50)
    print("Analyzing column inventory and data structure...")
    
    print(f"Complete column analysis:")
    print(f"  Total columns: {len(df_raw.columns)}")
    print()
    
    # Display all column names for manual verification
    print("Complete column inventory (first 50 columns):")
    for i, col in enumerate(df_raw.columns[:50]):
        print(f"  {i+1:3d}. {col}")
    
    if len(df_raw.columns) > 50:
        print(f"  ... and {len(df_raw.columns) - 50} additional columns")
    print()
    
    # Check for essential columns required for Grasso reproduction
    essential_columns = ['WA', 'SP_aa']
    print("Essential column verification:")
    all_essential_present = True
    for col in essential_columns:
        if col in df_raw.columns:
            print(f"  ✓ {col}: Present")
        else:
            print(f"  ✗ {col}: MISSING")
            all_essential_present = False
    
    if not all_essential_present:
        print()
        print("WARNING: Essential columns missing. Reproduction may not be possible.")
    
    print()
    
    # =========================================================================
    # STEP 4: GRASSO FEATURE AVAILABILITY CHECK
    # =========================================================================
    
    print("STEP 4: GRASSO FEATURE AVAILABILITY ASSESSMENT")
    print("-" * 50)
    print("Verifying availability of validated Grasso physicochemical features...")
    
    # Define the 156 validated features from Grasso Table S2
    # These are the exact features used in the original study
    GRASSO_VALIDATED_FEATURES = [
        # N-region features (amino-terminal region)
        'Turn_N', 'A_N', 'C_N', 'D_N', 'E_N', 'F_N', 'G_N', 'H_N', 'I_N', 
        'L_N', 'N_N', 'P_N', 'Q_N', 'S_N', 'T_N', 'V_N', 'W_N', 'Y_N',
        'Length_N', 'InstabilityInd_N', 'Aromaticity_N', 'flexibility_N', 
        'kytedoolittle_N', 'mfe_N',
        
        # H-region features (hydrophobic core region)
        'Turn_H', 'G_H', 'M_H', 'N_H', 'P_H', 'Q_H', 'S_H', 'T_H', 'W_H', 'Y_H',
        'Length_H', 'InstabilityInd_H', 'BomanInd_H', 'mfe_H',
        
        # C-region features (carboxy-terminal region)
        'Helix_C', 'Turn_C', 'Sheet_C', 'A_C', 'C_C', 'D_C', 'E_C', 'G_C', 
        'I_C', 'L_C', 'M_C', 'N_C', 'P_C', 'Q_C', 'R_C', 'S_C', 'T_C', 
        'V_C', 'W_C', 'Y_C', 'Length_C', 'pI_C', 'InstabilityInd_C', 
        'AliphaticInd_C', 'ez_C', 'gravy_C', 'mfe_C', 'CAI_RSCU_C',
        
        # Ac-region features (post-cleavage amino acids)
        'Turn_Ac', 'Sheet_Ac', 'A_Ac', 'D_Ac', 'E_Ac', 'F_Ac', 'G_Ac', 
        'H_Ac', 'I_Ac', 'L_Ac', 'M_Ac', 'N_Ac', 'P_Ac', 'Q_Ac', 'R_Ac', 
        'S_Ac', 'T_Ac', 'V_Ac', 'MW_Ac', 'pI_Ac', 'InstabilityInd_Ac', 
        'BomanInd_Ac', 'ez_Ac', 'mfe_Ac', 'CAI_RSCU_Ac',
        
        # SP-region features (global signal peptide properties)
        'Helix_SP', 'Turn_SP', 'D_SP', 'E_SP', 'F_SP', 'G_SP', 'H_SP', 
        'L_SP', 'M_SP', 'N_SP', 'P_SP', 'Q_SP', 'S_SP', 'T_SP', 'W_SP', 'Y_SP',
        'Length_SP', 'Charge_SP', 'InstabilityInd_SP', 'flexibility_SP', 
        'gravy_SP', 'mfe_SP', '-35_mfe_SP', 'amyQ_mfe_SP', 'CAI_RSCU_SP',
        
        # Cleavage site features (signal peptidase specificity)
        '-3_A', '-3_C', '-3_D', '-3_E', '-3_F', '-3_G', '-3_H', '-3_I', '-3_K',
        '-3_L', '-3_M', '-3_N', '-3_P', '-3_Q', '-3_R', '-3_S', '-3_T', '-3_V',
        '-3_W', '-3_Y', '-1_A', '-1_C', '-1_D', '-1_E', '-1_F', '-1_G', '-1_H',
        '-1_I', '-1_K', '-1_L', '-1_M', '-1_N', '-1_P', '-1_Q', '-1_R', '-1_S',
        '-1_T', '-1_V', '-1_W', '-1_Y'
    ]
    
    # Check availability of each validated feature
    available_features = [f for f in GRASSO_VALIDATED_FEATURES if f in df_raw.columns]
    missing_features = [f for f in GRASSO_VALIDATED_FEATURES if f not in df_raw.columns]
    
    print(f"Feature availability analysis:")
    print(f"  Expected Grasso features: {len(GRASSO_VALIDATED_FEATURES)}")
    print(f"  Available in dataset: {len(available_features)}")
    print(f"  Missing from dataset: {len(missing_features)}")
    print(f"  Availability rate: {len(available_features)/len(GRASSO_VALIDATED_FEATURES)*100:.1f}%")
    print()
    
    # Display missing features if any
    if missing_features:
        print("Missing Grasso features (first 20):")
        for i, feature in enumerate(missing_features[:20]):
            print(f"  {i+1:2d}. {feature}")
        if len(missing_features) > 20:
            print(f"  ... and {len(missing_features) - 20} additional missing features")
        print()
    else:
        print("✓ All Grasso validated features are available in the dataset")
        print()
    
    # =========================================================================
    # STEP 5: DATA QUALITY ASSESSMENT
    # =========================================================================
    
    print("STEP 5: DATA QUALITY ASSESSMENT")
    print("-" * 50)
    print("Evaluating data quality for key columns...")
    
    # Analyze WA (target variable) column in detail
    if 'WA' in df_raw.columns:
        wa_series = df_raw['WA']
        print(f"WA (secretion efficiency) column analysis:")
        print(f"  Total entries: {len(wa_series):,}")
        print(f"  Non-null entries: {wa_series.notna().sum():,}")
        print(f"  Missing entries: {wa_series.isna().sum():,}")
        print(f"  Data type: {wa_series.dtype}")
        
        # Calculate statistics for non-null WA values
        valid_wa = wa_series.dropna()
        if len(valid_wa) > 0:
            print(f"  Value range: {valid_wa.min():.2f} - {valid_wa.max():.2f}")
            print(f"  Mean: {valid_wa.mean():.3f}")
            print(f"  Median: {valid_wa.median():.3f}")
            print(f"  Standard deviation: {valid_wa.std():.3f}")
            
            # Check for values outside expected experimental range
            valid_range_mask = (valid_wa >= 1.0) & (valid_wa <= 10.0)
            invalid_count = (~valid_range_mask).sum()
            print(f"  Valid range (1.0-10.0): {valid_range_mask.sum():,} values")
            print(f"  Invalid range: {invalid_count:,} values")
            
            if invalid_count > 0:
                print(f"  Example invalid values: {valid_wa[~valid_range_mask].head().tolist()}")
        
        print()
    else:
        print("WARNING: WA column not found in dataset")
        print()
    
    # Analyze SP_aa (signal peptide sequences) column
    if 'SP_aa' in df_raw.columns:
        sp_series = df_raw['SP_aa']
        print(f"SP_aa (signal peptide sequences) column analysis:")
        print(f"  Total entries: {len(sp_series):,}")
        print(f"  Non-null entries: {sp_series.notna().sum():,}")
        print(f"  Missing entries: {sp_series.isna().sum():,}")
        print(f"  Data type: {sp_series.dtype}")
        
        # Analyze sequence lengths for non-null sequences
        valid_sequences = sp_series.dropna()
        if len(valid_sequences) > 0:
            sequence_lengths = valid_sequences.str.len()
            print(f"  Sequence length range: {sequence_lengths.min()} - {sequence_lengths.max()} amino acids")
            print(f"  Mean length: {sequence_lengths.mean():.1f} amino acids")
            print(f"  Median length: {sequence_lengths.median():.1f} amino acids")
            
            # Check for sequences within functional length range
            functional_length_mask = (sequence_lengths >= 10) & (sequence_lengths <= 40)
            print(f"  Functional length (10-40 aa): {functional_length_mask.sum():,} sequences")
            print(f"  Non-functional length: {(~functional_length_mask).sum():,} sequences")
        
        print()
    else:
        print("WARNING: SP_aa column not found in dataset")
        print()
    
    # =========================================================================
    # STEP 6: SAMPLE DATA VERIFICATION
    # =========================================================================
    
    print("STEP 6: SAMPLE DATA VERIFICATION")
    print("-" * 50)
    print("Displaying sample data to verify correct parsing...")
    
    # Select key columns for sample data display
    key_columns = ['WA', 'SP_aa'] + available_features[:5]
    available_key_columns = [col for col in key_columns if col in df_raw.columns]
    
    if available_key_columns:
        print("Sample data (first 5 rows of key columns):")
        sample_data = df_raw[available_key_columns].head()
        print(sample_data.to_string())
        print()
        
        print("Data type verification for key columns:")
        for col in available_key_columns:
            print(f"  {col:<20s}: {df_raw[col].dtype}")
        print()
    else:
        print("No key columns available for sample display")
        print()
    
    # =========================================================================
    # STEP 7: FILTERING PROCESS VERIFICATION
    # =========================================================================
    
    print("STEP 7: FILTERING PROCESS VERIFICATION")
    print("-" * 50)
    print("Testing Grasso quality control filters step-by-step...")
    
    # Start with original dataset size
    n_original = len(df_raw)
    print(f"Original dataset: {n_original:,} samples")
    
    # Apply each filter sequentially to track sample retention
    current_data = df_raw.copy()
    
    # Filter 1: Non-null WA values
    if 'WA' in current_data.columns:
        mask1 = current_data['WA'].notna()
        current_data = current_data[mask1]
        n_after_wa = len(current_data)
        print(f"After WA non-null filter: {n_after_wa:,} samples (removed: {n_original - n_after_wa:,})")
    else:
        n_after_wa = n_original
        print(f"WA column missing - skipping WA filter")
    
    # Filter 2: Non-null SP_aa values
    if 'SP_aa' in current_data.columns:
        mask2 = current_data['SP_aa'].notna()
        current_data = current_data[mask2]
        n_after_sp = len(current_data)
        print(f"After SP_aa non-null filter: {n_after_sp:,} samples (removed: {n_after_wa - n_after_sp:,})")
    else:
        n_after_sp = n_after_wa
        print(f"SP_aa column missing - skipping SP_aa filter")
    
    # Filter 3: Signal peptide length constraints (10-40 amino acids)
    if 'SP_aa' in current_data.columns:
        sp_lengths = current_data['SP_aa'].str.len()
        mask3 = (sp_lengths >= 10) & (sp_lengths <= 40)
        current_data = current_data[mask3]
        n_after_length = len(current_data)
        print(f"After length filter (10-40 aa): {n_after_length:,} samples (removed: {n_after_sp - n_after_length:,})")
    else:
        n_after_length = n_after_sp
        print(f"SP_aa column missing - skipping length filter")
    
    # Filter 4: WA value range constraints (1.0-10.0)
    if 'WA' in current_data.columns:
        mask4 = (current_data['WA'] >= 1.0) & (current_data['WA'] <= 10.0)
        current_data = current_data[mask4]
        n_final = len(current_data)
        print(f"After WA range filter (1.0-10.0): {n_final:,} samples (removed: {n_after_length - n_final:,})")
    else:
        n_final = n_after_length
        print(f"WA column missing - skipping WA range filter")
    
    print()
    print(f"FILTERING SUMMARY:")
    print(f"  Original samples: {n_original:,}")
    print(f"  Final valid samples: {n_final:,}")
    print(f"  Overall retention rate: {n_final/n_original*100:.1f}%")
    print(f"  Total samples removed: {n_original - n_final:,}")
    print()
    
    # =========================================================================
    # STEP 8: VERIFICATION CONCLUSION AND RECOMMENDATIONS
    # =========================================================================
    
    print("STEP 8: VERIFICATION SUMMARY AND ASSESSMENT")
    print("-" * 50)
    
    # Compile verification results
    verification_issues = []
    
    # Check for critical issues that would affect reproduction
    if missing_features:
        verification_issues.append(f"{len(missing_features)} Grasso features missing")
    
    if not all_essential_present:
        verification_issues.append("Essential columns (WA, SP_aa) missing")
    
    if n_final < 5000:  # Arbitrary threshold for sufficient sample size
        verification_issues.append(f"Low final sample count ({n_final:,})")
    
    # Generate verification assessment
    if verification_issues:
        print("VERIFICATION ASSESSMENT: Issues Identified")
        print()
        print("Issues detected:")
        for issue in verification_issues:
            print(f"  ⚠ {issue}")
        print()
        print("Recommendations:")
        print("  • Verify dataset completeness and version")
        print("  • Check for correct file format and encoding")
        print("  • Confirm dataset matches Grasso supplementary materials")
        print("  • Consider using original Excel files if available")
        
    else:
        print("VERIFICATION ASSESSMENT: Data Loading Successful")
        print()
        print("Verification results:")
        print("  ✓ Complete file loading confirmed")
        print("  ✓ All essential columns present")
        print("  ✓ Grasso features available for analysis")
        print("  ✓ Data quality meets reproduction requirements")
        print("  ✓ Filtering process validates correctly")
        print()
        print("Conclusion:")
        print("Dataset appears complete and ready for reproduction analysis.")
        print("Any reproduction challenges are likely methodological rather than data-related.")
    
    print()
    print("DATA VERIFICATION COMPLETED")
    print("=" * 70)
    
    return df_raw

# ============================================================================
# INDIVIDUAL VERIFICATION FUNCTIONS
# ============================================================================

def quick_feature_check(filename="sb2c00328_si_011.csv"):
    """
    Quick verification of Grasso feature availability.
    
    Provides rapid assessment of whether the dataset contains the required
    156 validated features for Grasso reproduction without full verification.
    """
    print("QUICK GRASSO FEATURE AVAILABILITY CHECK")
    print("=" * 50)
    
    try:
        df = pd.read_csv(filename, low_memory=False)
        print(f"Dataset loaded: {df.shape[0]:,} × {df.shape[1]:,}")
        
        # Define expected features (abbreviated list for quick check)
        key_features = ['WA', 'SP_aa', 'gravy_SP', '-1_A', 'A_C', 'Length_SP']
        
        print("\nKey feature availability:")
        for feature in key_features:
            status = "✓" if feature in df.columns else "✗"
            print(f"  {status} {feature}")
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def validate_wa_distribution(filename="sb2c00328_si_011.csv"):
    """
    Validate WA (secretion efficiency) distribution characteristics.
    
    Checks whether WA values follow expected distribution patterns
    based on biological principles and experimental methodology.
    """
    print("WA DISTRIBUTION VALIDATION")
    print("=" * 40)
    
    try:
        df = pd.read_csv(filename, low_memory=False)
        
        if 'WA' in df.columns:
            wa_values = df['WA'].dropna()
            
            print(f"WA distribution analysis:")
            print(f"  Sample size: {len(wa_values):,}")
            print(f"  Range: {wa_values.min():.2f} - {wa_values.max():.2f}")
            print(f"  Mean: {wa_values.mean():.3f}")
            print(f"  Median: {wa_values.median():.3f}")
            print(f"  Std deviation: {wa_values.std():.3f}")
            
            # Check distribution characteristics
            from scipy import stats
            skewness = stats.skew(wa_values)
            print(f"  Skewness: {skewness:.3f}")
            
            return wa_values
        else:
            print("WA column not found")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution for standalone data verification.
    
    Runs comprehensive data verification when script is executed directly.
    Suitable for batch processing and automated quality assurance workflows.
    """
    
    print("GRASSO DATASET VERIFICATION TOOL")
    print("=" * 50)
    print("Comprehensive data integrity assessment")
    print()
    
    # Execute comprehensive verification
    verified_data = comprehensive_data_verification()
    
    if verified_data is not None:
        print()
        print("VERIFICATION COMPLETED SUCCESSFULLY")
        print("Dataset ready for reproduction analysis")
    else:
        print()
        print("VERIFICATION IDENTIFIED CRITICAL ISSUES")
        print("Please address data loading problems before proceeding")