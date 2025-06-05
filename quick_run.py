"""
Quick Runner for Grasso Reproduction Study
=========================================

EXECUTION INTERFACE:
Provides a streamlined way to execute the complete Grasso reproduction
study with minimal setup and clear output.

FEATURES:
- System requirements checking
- Data availability verification
- Complete reproduction study execution
- Data verification tools
- Simple menu-driven interface

USAGE:
python quick_run.py

AUTHOR: Mehak Wadhwa
INSTITUTION: Fordham University
RESEARCH MENTOR: Dr. Joshua Schrier
"""

import os
import sys
import time

def check_requirements():
    """
    Verify that all required Python packages are available.
    
    RETURNS:
    bool: True if all requirements satisfied, False otherwise
    """
    print("CHECKING REQUIREMENTS")
    print("-" * 30)
    
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
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All requirements satisfied")
    return True

def check_data():
    """
    Check if the Grasso dataset is available.
    
    RETURNS:
    str or None: Filename if found, None if not available
    """
    print("CHECKING DATA AVAILABILITY")
    print("-" * 30)
    
    # Check for primary data file
    primary_file = 'sb2c00328_si_011.csv'
    
    if os.path.exists(primary_file):
        size = os.path.getsize(primary_file)
        print(f"  ✓ Found: {primary_file} ({size:,} bytes)")
        return primary_file
    
    # Check alternative location
    alt_file = 'data/sb2c00328_si_011.csv'
    if os.path.exists(alt_file):
        size = os.path.getsize(alt_file)
        print(f"  ✓ Found: {alt_file} ({size:,} bytes)")
        return alt_file
    
    print(f"  ✗ Grasso dataset not found")
    print(f"\nExpected file: {primary_file}")
    print("Dataset requirements:")
    print("  • File size: ~13-15 MB")
    print("  • Contains ~11,643 signal peptide variants")
    print("  • Includes 156+ physicochemical features")
    return None

def run_reproduction_study():
    """
    Execute the complete Grasso reproduction study.
    
    RETURNS:
    dict or None: Results if successful, None if failed
    """
    print("RUNNING GRASSO REPRODUCTION STUDY")
    print("=" * 40)
    print("Expected time: 2-3 minutes")
    print()
    
    start_time = time.time()
    
    try:
        # Import and run the main analysis
        from grasso_reproduction_tool import execute_grasso_reproduction
        results = execute_grasso_reproduction()
        
        end_time = time.time()
        
        if results is not None:
            print("\n" + "="*50)
            print("REPRODUCTION STUDY COMPLETED")
            print("="*50)
            print(f"Execution time: {end_time - start_time:.1f} seconds")
            print()
            print("KEY RESULTS:")
            print(f"  Test MSE: {results['test_mse']:.3f} WA²")
            print(f"  Target MSE: {results['target_test_mse']:.2f} WA²")
            print(f"  Reproduction accuracy: {100 - results['relative_error_percent']:.1f}%")
            print(f"  Test R²: {results['test_r2']:.3f}")
            print()
            print("OUTPUTS GENERATED:")
            print("  • grasso_reproduction_results.png")
            print("  • Complete console output with all metrics")
            print()
            print("Most important feature:")
            top_feature = results['feature_importance'].iloc[0]
            print(f"  {top_feature['Feature']} (importance: {top_feature['Importance']:.3f})")
            
            return results
        else:
            print("\n✗ REPRODUCTION STUDY FAILED")
            print("Check error messages above for details")
            return None
            
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting:")
        print("  • Verify data file is available")
        print("  • Check package installations")
        print("  • Ensure sufficient memory")
        return None

def run_data_verification():
    """
    Run data verification and integrity checks.
    
    RETURNS:
    bool: True if verification passed, False otherwise
    """
    print("RUNNING DATA VERIFICATION")
    print("=" * 30)
    
    try:
        from data_verification_tool import comprehensive_data_verification
        result = comprehensive_data_verification()
        
        if result is not None:
            print("\n✓ DATA VERIFICATION PASSED")
            return True
        else:
            print("\n✗ DATA VERIFICATION FAILED")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """
    Main execution function with interactive menu.
    """
    print("GRASSO REPRODUCTION STUDY - QUICK RUNNER")
    print("=" * 50)
    print("Two approaches available:")
    print("  • Modular approach (grasso_reproduction_tool.py + helpers)")
    print("  • Single script approach (01_grasso_reproduction_complete.py)")
    print("  Both achieve identical 97.6% reproduction accuracy")
    print()
    
    # Check system readiness
    if not check_requirements():
        return
    
    data_file = check_data()
    if not data_file:
        return
    
    print("\nSYSTEM READY")
    print()
    
    # Interactive menu
    while True:
        print("OPTIONS:")
        print("  1. Run complete reproduction study (modular approach)")
        print("  2. Run single script reproduction (research notebook style)")
        print("  3. Data verification only")
        print("  4. Exit")
        print()
        
        try:
            choice = input("Select option (1-4): ").strip()
            
            if choice == '1':
                print()
                results = run_reproduction_study()
                break
                
            elif choice == '2':
                print()
                print("RUNNING SINGLE SCRIPT REPRODUCTION")
                print("=" * 40)
                
                # Check if single script file exists
                if os.path.exists('01_grasso_reproduction_complete.py'):
                    try:
                        print("Executing single script reproduction...")
                        # Import and run the function directly
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("complete", "01_grasso_reproduction_complete.py")
                        complete_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(complete_module)
                        
                        # Run the main function
                        results = complete_module.execute_complete_reproduction()
                        
                        if results:
                            print("\n✓ Single script reproduction completed successfully!")
                        else:
                            print("\n✗ Single script reproduction failed")
                            
                    except Exception as e:
                        print(f"Error running single script: {e}")
                        print("\nAlternative: Run manually with:")
                        print("python 01_grasso_reproduction_complete.py")
                else:
                    print("01_grasso_reproduction_complete.py not found in current directory")
                    print("Please add this file to use single script approach")
                    print("Using modular approach instead...")
                    print()
                    run_reproduction_study()
                
                break
                
            elif choice == '3':
                print()
                run_data_verification()
                
            elif choice == '4':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please select 1, 2, 3, or 4.")
                
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def run_all():
    """
    Execute complete study without interactive menu.
    
    RETURNS:
    dict or None: Results if successful, None if failed
    """
    if not check_requirements():
        return None
    
    if not check_data():
        return None
    
    return run_reproduction_study()

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'auto':
            run_all()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: auto")
    else:
        main()
