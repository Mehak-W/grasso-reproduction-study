"""
Quick Runner for Grasso Reproduction Study
=========================================

SIMPLE EXECUTION INTERFACE:
This script provides a streamlined way to execute the complete Grasso reproduction
study with minimal setup. Designed for users who want results without needing to
understand all implementation details.

EXECUTION OPTIONS:
1. Complete analysis with all components
2. Data verification only
3. Quick feature check
4. Research findings analysis (if results available)

USAGE:
Simply run this script directly:
python quick_run.py

Or import specific functions:
from quick_run import run_complete_analysis

AUTHOR: Mehak Wadhwa
INSTITUTION: Fordham University
"""

import os
import sys
import time

def check_requirements():
    """
    Verify that all required packages are available before execution.
    
    Checks for essential Python packages needed for the reproduction study
    and provides helpful error messages if packages are missing.
    """
    print("CHECKING SYSTEM REQUIREMENTS")
    print("-" * 40)
    
    required_packages = [
        ('pandas', 'Data manipulation and analysis'),
        ('numpy', 'Numerical computing'),
        ('sklearn', 'Machine learning algorithms'),
        ('scipy', 'Statistical analysis'),
        ('matplotlib', 'Visualization')
    ]
    
    missing_packages = []
    
    for package, description in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"  ✓ {package:<12s}: Available ({description})")
        except ImportError:
            print(f"  ✗ {package:<12s}: MISSING ({description})")
            missing_packages.append(package)
    
    if missing_packages:
        print()
        print("ERROR: Missing required packages")
        print("Install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        print()
        print("Or install all requirements:")
        print("pip install -r requirements.txt")
        return False
    
    print()
    print("✓ All required packages available")
    return True

def check_data_availability():
    """
    Check if the required Grasso dataset is available in CSV or Excel format.
    
    Looks for both file formats and provides guidance if not found.
    """
    print("CHECKING DATA AVAILABILITY")
    print("-" * 40)
    
    expected_files = [
        ("sb2c00328_si_011.csv", "CSV format (faster loading)"),
        ("sb2c00328_si_011.xlsx", "Excel format (original supplementary)"),
        ("data/sb2c00328_si_011.csv", "CSV in data folder"),
        ("data/sb2c00328_si_011.xlsx", "Excel in data folder")
    ]
    
    found_files = []
    for filename, description in expected_files:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            found_files.append((filename, description, file_size))
            print(f"  ✓ Dataset found: {filename}")
            print(f"    Description: {description}")
            print(f"    File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            print()
    
    if found_files:
        # Return the first found file (CSV preferred)
        return found_files[0][0]
    
    print("  ✗ Grasso dataset not found")
    print()
    print("Expected files (either format works):")
    print("  CSV format: sb2c00328_si_011.csv")
    print("  Excel format: sb2c00328_si_011.xlsx")
    print()
    print("Expected locations:")
    print("  • Current directory")
    print("  • data/ subdirectory")
    print()
    print("Dataset requirements:")
    print("  • File size: ~13-27 MB")
    print("  • Contains ~11,643 signal peptide variants")
    print("  • Includes 156+ physicochemical features")
    print("  • Contains WA scores and SP_aa sequences")
    print("  • Excel files: 'Library_w_Bins_and_WA' sheet required")
    
    return None

def run_data_verification_only():
    """
    Execute only the data verification component.
    
    Useful for checking data integrity without running the full analysis.
    """
    print("EXECUTING DATA VERIFICATION ONLY")
    print("=" * 50)
    print("Running comprehensive data integrity assessment...")
    print()
    
    try:
        from data_verification_tool import comprehensive_data_verification
        result = comprehensive_data_verification()
        
        if result is not None:
            print()
            print("✓ DATA VERIFICATION COMPLETED SUCCESSFULLY")
            print("Dataset ready for reproduction analysis")
            return True
        else:
            print()
            print("✗ DATA VERIFICATION IDENTIFIED ISSUES")
            print("Please address data problems before proceeding")
            return False
            
    except Exception as e:
        print(f"Error during data verification: {e}")
        return False

def run_quick_feature_check():
    """
    Execute quick feature availability check.
    
    Rapid assessment of whether key Grasso features are present in the dataset.
    """
    print("EXECUTING QUICK FEATURE CHECK")
    print("=" * 50)
    print("Checking availability of key Grasso features...")
    print()
    
    try:
        from data_verification_tool import quick_feature_check
        result = quick_feature_check()
        
        if result is not None:
            print()
            print("✓ FEATURE CHECK COMPLETED")
            return True
        else:
            print()
            print("✗ FEATURE CHECK FAILED")
            return False
            
    except Exception as e:
        print(f"Error during feature check: {e}")
        return False

def run_complete_analysis():
    """
    Execute the complete Grasso reproduction study.
    
    Runs the full analysis pipeline including:
    - Data loading and verification
    - Preprocessing and model training
    - Performance evaluation
    - Feature importance analysis
    - Results visualization
    """
    print("EXECUTING COMPLETE GRASSO REPRODUCTION ANALYSIS")
    print("=" * 60)
    print("Running comprehensive reproduction study...")
    print("Expected execution time: 3-5 minutes")
    print()
    
    start_time = time.time()
    
    try:
        from grasso_reproduction_tool import execute_grasso_reproduction
        results = execute_grasso_reproduction()
        
        execution_time = time.time() - start_time
        
        if results is not None:
            print()
            print("✓ COMPLETE ANALYSIS FINISHED SUCCESSFULLY")
            print(f"Total execution time: {execution_time:.1f} seconds")
            print()
            print("Generated outputs:")
            print("  • grasso_reproduction_results.png (comprehensive visualization)")
            print("  • Complete performance metrics and analysis")
            print("  • Feature importance rankings with biological validation")
            print()
            print("Key results:")
            print(f"  • Test MSE: {results['test_mse']:.3f} WA² (Target: {results['grasso_test_target']:.2f})")
            print(f"  • Test R²: {results['test_r2']:.3f}")
            print(f"  • Reproduction accuracy: {100 - results['relative_error_percent']:.1f}%")
            print(f"  • Most important feature: {results['feature_importance'].iloc[0]['Feature']}")
            
            return results
        else:
            print()
            print("✗ ANALYSIS FAILED TO COMPLETE")
            print("Check error messages above for troubleshooting guidance")
            return None
            
    except Exception as e:
        print(f"Error during complete analysis: {e}")
        print()
        print("Troubleshooting suggestions:")
        print("  • Verify data file availability and format")
        print("  • Check Python package installations")
        print("  • Ensure sufficient memory available")
        return None

def run_research_findings_analysis(results=None):
    """
    Execute research findings analysis on completed results.
    
    Provides detailed scientific interpretation and biological validation
    of reproduction study outcomes.
    """
    print("EXECUTING RESEARCH FINDINGS ANALYSIS")
    print("=" * 50)
    
    if results is None:
        print("No results provided. Running complete analysis first...")
        results = run_complete_analysis()
        
        if results is None:
            print("Cannot proceed without valid results")
            return None
    
    print("Analyzing research findings and biological implications...")
    print()
    
    try:
        from research_findings_discussion import analyze_reproduction_outcomes, generate_research_summary
        
        analysis_results = analyze_reproduction_outcomes(results, results['feature_importance'])
        
        print()
        print("=" * 80)
        generate_research_summary(results, analysis_results)
        
        print()
        print("✓ RESEARCH FINDINGS ANALYSIS COMPLETED")
        print("Generated comprehensive scientific interpretation")
        
        return analysis_results
        
    except Exception as e:
        print(f"Error during research findings analysis: {e}")
        return None

def display_menu():
    """
    Display interactive menu for user selection.
    """
    print("GRASSO REPRODUCTION STUDY - QUICK RUNNER")
    print("=" * 50)
    print("Select execution option:")
    print()
    print("  1. Complete Analysis (Full reproduction study)")
    print("  2. Data Verification Only (Check data integrity)")
    print("  3. Quick Feature Check (Verify key features available)")
    print("  4. Research Findings Analysis (Interpret results)")
    print("  5. System Requirements Check")
    print("  0. Exit")
    print()

def main():
    """
    Main interactive execution function.
    
    Provides menu-driven interface for different analysis options.
    """
    # Initial system check
    if not check_requirements():
        print("Please install required packages before proceeding")
        return
    
    data_file = check_data_availability()
    if data_file is None:
        print("Please ensure dataset is available before proceeding")
        return
    
    print("✓ System ready for analysis")
    print()
    
    while True:
        display_menu()
        
        try:
            choice = input("Enter your choice (0-5): ").strip()
            print()
            
            if choice == '0':
                print("Exiting Grasso reproduction study runner")
                break
                
            elif choice == '1':
                results = run_complete_analysis()
                if results:
                    # Offer research findings analysis
                    print()
                    proceed = input("Run research findings analysis? (y/n): ").strip().lower()
                    if proceed in ['y', 'yes']:
                        run_research_findings_analysis(results)
                
            elif choice == '2':
                run_data_verification_only()
                
            elif choice == '3':
                run_quick_feature_check()
                
            elif choice == '4':
                run_research_findings_analysis()
                
            elif choice == '5':
                check_requirements()
                check_data_availability()
                
            else:
                print("Invalid choice. Please enter 0-5.")
            
            print()
            input("Press Enter to continue...")
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()

# ============================================================================
# DIRECT EXECUTION FUNCTIONS
# ============================================================================

def run_all():
    """
    Execute complete analysis pipeline without interactive menu.
    
    Suitable for automated execution or script integration.
    """
    print("AUTOMATED COMPLETE EXECUTION")
    print("=" * 40)
    
    # System checks
    if not check_requirements():
        return None
    
    data_file = check_data_availability()
    if data_file is None:
        return None
    
    # Run complete analysis
    results = run_complete_analysis()
    
    if results:
        # Run research findings analysis
        analysis_results = run_research_findings_analysis(results)
        
        return {
            'reproduction_results': results,
            'research_analysis': analysis_results
        }
    
    return None

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Script execution entry point.
    
    Determines execution mode based on command line arguments or runs
    interactive menu by default.
    """
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'all':
            run_all()
        elif mode == 'verify':
            run_data_verification_only()
        elif mode == 'check':
            run_quick_feature_check()
        elif mode == 'analysis':
            run_complete_analysis()
        elif mode == 'findings':
            run_research_findings_analysis()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: all, verify, check, analysis, findings")
    else:
        # Interactive menu mode
        main()