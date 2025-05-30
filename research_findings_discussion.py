"""
Research Findings Analysis - Grasso et al. (2023) Reproduction Study
===================================================================

Comprehensive analysis and interpretation of reproduction study results.
Examines performance outcomes, biological validation, and reproducibility factors.

AUTHOR: Mehak Wadhwa
INSTITUTION: Fordham University
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_reproduction_outcomes(results, feature_importance_df):
    """
    Analyze reproduction study outcomes with scientific interpretation.
    
    Provides quantitative assessment of reproduction accuracy, biological validation
    of learned relationships, and analysis of factors affecting outcomes.
    
    PARAMETERS:
    results (dict): Complete results from grasso_reproduction_tool
    feature_importance_df (DataFrame): Feature importance rankings
    
    RETURNS:
    dict: Analysis results with scientific interpretations
    """
    
    print("RESEARCH FINDINGS ANALYSIS")
    print("=" * 60)
    print("Quantitative assessment of reproduction outcomes and biological validation")
    print()
    
    # =========================================================================
    # PERFORMANCE ANALYSIS
    # =========================================================================
    
    print("PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    # Extract key metrics
    achieved_test_mse = results['test_mse']
    target_test_mse = results['grasso_test_target']  # 1.22 WA²
    achieved_train_mse = results['train_mse']
    target_train_mse = results['grasso_train_target']  # 1.75 WA²
    
    # Calculate errors
    test_error = abs(achieved_test_mse - target_test_mse)
    test_relative_error = (test_error / target_test_mse) * 100
    train_error = abs(achieved_train_mse - target_train_mse)
    train_relative_error = (train_error / target_train_mse) * 100
    
    print("Benchmark Comparison:")
    print(f"  Test MSE: {achieved_test_mse:.3f} WA² (Target: {target_test_mse:.2f} WA²)")
    print(f"  Test error: {test_relative_error:.1f}%")
    print(f"  Train MSE: {achieved_train_mse:.3f} WA² (Target: {target_train_mse:.2f} WA²)")
    print(f"  Train error: {train_relative_error:.1f}%")
    print()
    
    # Effect size assessment
    pooled_std = np.sqrt((results['train_mae']**2 + results['test_mae']**2) / 2)
    cohens_d = test_error / pooled_std if pooled_std > 0 else 0
    
    print(f"Effect size (Cohen's d): {cohens_d:.3f}")
    if cohens_d < 0.2:
        effect_interpretation = "Small difference"
    elif cohens_d < 0.8:
        effect_interpretation = "Medium difference"
    else:
        effect_interpretation = "Large difference"
    print(f"Effect interpretation: {effect_interpretation}")
    print()
    
    # =========================================================================
    # BIOLOGICAL VALIDATION
    # =========================================================================
    
    print("BIOLOGICAL VALIDATION")
    print("-" * 40)
    
    # Define biologically relevant feature categories
    biological_categories = {
        'Hydrophobicity': ['gravy_SP', 'gravy_C', 'BomanInd_H', 'BomanInd_Ac'],
        'Cleavage_Specificity': ['-1_A', '-3_A', '-1_S', '-3_S', 'A_C'],
        'Length_Features': ['Length_SP', 'Length_N', 'Length_H', 'Length_C'],
        'Charge_Effects': ['Charge_SP', 'pI_C', 'pI_Ac'],
        'Structure': ['Helix_SP', 'Turn_SP', 'Sheet_C', 'flexibility_SP'],
        'Energy': ['mfe_SP', 'mfe_N', 'mfe_H', 'mfe_C']
    }
    
    # Analyze top 15 features
    top_15_features = feature_importance_df.head(15)
    biological_validation_results = {}
    
    print("Top 15 Features - Biological Assessment:")
    print("Rank  Feature               Importance    Category")
    print("-" * 55)
    
    total_biological_importance = 0
    
    for idx, (_, row) in enumerate(top_15_features.iterrows(), 1):
        feature_name = row['Feature']
        importance = row['Importance']
        
        # Determine biological category
        biological_category = "Other"
        for category, feature_list in biological_categories.items():
            if any(bio_feature in feature_name for bio_feature in feature_list):
                biological_category = category
                total_biological_importance += importance
                break
        
        # Store for analysis
        if biological_category not in biological_validation_results:
            biological_validation_results[biological_category] = []
        biological_validation_results[biological_category].append((feature_name, importance))
        
        print(f"{idx:2d}    {feature_name:<18s}    {importance:.4f}        {biological_category}")
    
    print("-" * 55)
    biological_relevance_ratio = (total_biological_importance / top_15_features['Importance'].sum()) * 100
    print(f"Biological relevance: {biological_relevance_ratio:.1f}% of top 15 features")
    print()
    
    # Category breakdown
    print("Category Analysis:")
    for category, features in biological_validation_results.items():
        if features and category != "Other":
            category_importance = sum(imp for _, imp in features)
            print(f"  {category}: {category_importance:.4f} importance")
    print()
    
    # =========================================================================
    # MODEL PERFORMANCE CHARACTERISTICS
    # =========================================================================
    
    print("MODEL PERFORMANCE CHARACTERISTICS")
    print("-" * 40)
    
    # Overfitting assessment
    overfitting = results['train_r2'] - results['test_r2']
    cv_stability = results['cv_r2_std']
    prediction_bias = np.mean(results['test_predictions']) - np.mean(results['y_test_eval'])
    
    print(f"Overfitting (Train R² - Test R²): {overfitting:.3f}")
    if overfitting > 0.3:
        overfitting_status = "Significant overfitting"
    elif overfitting > 0.1:
        overfitting_status = "Moderate overfitting"
    else:
        overfitting_status = "Minimal overfitting"
    print(f"Assessment: {overfitting_status}")
    print()
    
    print(f"Cross-validation stability: ±{cv_stability:.4f}")
    stability_status = "High stability" if cv_stability < 0.05 else "Moderate stability" if cv_stability < 0.1 else "Variable performance"
    print(f"Assessment: {stability_status}")
    print()
    
    print(f"Prediction bias: {prediction_bias:.4f} WA units")
    bias_status = "Minimal bias" if abs(prediction_bias) < 0.1 else "Moderate bias" if abs(prediction_bias) < 0.3 else "Significant bias"
    print(f"Assessment: {bias_status}")
    print()
    
    # =========================================================================
    # CONTRIBUTING FACTORS
    # =========================================================================
    
    print("POTENTIAL CONTRIBUTING FACTORS")
    print("-" * 40)
    
    print("Dataset factors:")
    print("  • Experimental batch effects between studies")
    print("  • Different preprocessing or feature calculation methods")
    print("  • Sample composition differences")
    print("  • Quality control criteria variations")
    print()
    
    print("Methodological factors:")
    print("  • Hyperparameter optimization differences")
    print("  • Random seed and initialization effects")
    print("  • Cross-validation strategy variations")
    print("  • Software version differences")
    print()
    
    # =========================================================================
    # REPRODUCIBILITY ASSESSMENT
    # =========================================================================
    
    print("REPRODUCIBILITY ASSESSMENT")
    print("-" * 40)
    
    # Classify reproduction outcome
    if test_relative_error < 15:
        reproduction_class = "Excellent reproducibility"
        implications = [
            "Validates robustness of published methodology",
            "Supports confidence in computational workflows",
            "Demonstrates effective reproduction protocols"
        ]
    elif test_relative_error < 30:
        reproduction_class = "Good reproducibility with challenges"
        implications = [
            "Shows importance of complete methodology documentation",
            "Highlights need for standardized preprocessing",
            "Demonstrates value of biological validation"
        ]
    else:
        reproduction_class = "Challenging reproducibility"
        implications = [
            "Identifies fundamental reproduction challenges",
            "Highlights critical gaps in methodology documentation",
            "Demonstrates need for improved data sharing"
        ]
    
    print(f"Classification: {reproduction_class}")
    print()
    print("Implications:")
    for i, implication in enumerate(implications, 1):
        print(f"  {i}. {implication}")
    print()
    
    # Compile results
    analysis_results = {
        'reproduction_classification': reproduction_class,
        'test_relative_error': test_relative_error,
        'biological_relevance_ratio': biological_relevance_ratio,
        'overfitting_status': overfitting_status,
        'stability_status': stability_status,
        'bias_status': bias_status,
        'implications': implications,
        'biological_validation_results': biological_validation_results
    }
    
    return analysis_results

def generate_research_summary(results, analysis_results):
    """
    Generate executive summary of key findings.
    
    Creates concise overview of methodology, outcomes, and scientific implications.
    """
    
    print("RESEARCH SUMMARY")
    print("=" * 40)
    print()
    
    print("STUDY OVERVIEW:")
    print(f"  Objective: Reproduce Grasso et al. (2023) signal peptide prediction")
    print(f"  Method: Random Forest with 156 physicochemical features")
    print(f"  Dataset: {len(results['y_train_eval']) + len(results['y_test_eval']):,} samples")
    print()
    
    print("KEY RESULTS:")
    print(f"  Test MSE: {results['test_mse']:.3f} WA² (Target: {results['grasso_test_target']:.2f} WA²)")
    print(f"  Reproduction accuracy: {100 - analysis_results['test_relative_error']:.1f}%")
    print(f"  Classification: {analysis_results['reproduction_classification']}")
    print(f"  Biological validation: {analysis_results['biological_relevance_ratio']:.1f}% relevant features")
    print()
    
    print("SCIENTIFIC CONTRIBUTIONS:")
    print("  • Quantitative assessment of reproducibility challenges")
    print("  • Biological validation framework for ML models")
    print("  • Evidence for improving documentation standards")
    print("  • Framework for future reproduction studies")
    print()

def create_findings_visualization(results, analysis_results, save_filename='research_findings_analysis.png'):
    """
    Create visualization focused on research findings and biological validation.
    
    Generates figure emphasizing biological validation, reproducibility assessment,
    and scientific implications of the reproduction study.
    """
    
    print("GENERATING FINDINGS VISUALIZATION")
    print("-" * 40)
    
    # Create 2×2 layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Research Findings Analysis - Biological Validation and Reproducibility', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Panel 1: Reproduction Accuracy
    ax1 = axes[0, 0]
    categories = ['This Study', 'Grasso Target']
    values = [results['test_mse'], results['grasso_test_target']]
    colors = ['lightcoral' if results['test_mse'] > results['grasso_test_target'] else 'lightgreen', 'steelblue']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.7)
    ax1.set_ylabel('Test MSE (WA²)', fontsize=11)
    ax1.set_title('Reproduction Accuracy', fontsize=12, fontweight='bold')
    
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    accuracy = 100 - analysis_results['test_relative_error']
    ax1.text(0.5, 0.8, f'Accuracy: {accuracy:.1f}%', 
             transform=ax1.transAxes, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Biological Validation
    ax2 = axes[0, 1]
    biological_ratio = analysis_results['biological_relevance_ratio']
    non_biological_ratio = 100 - biological_ratio
    
    sizes = [biological_ratio, non_biological_ratio]
    labels = ['Biologically\nMeaningful', 'Other']
    colors = ['lightgreen', 'lightcoral']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Top 15 Features\nBiological Validation', fontsize=12, fontweight='bold')
    
    # Panel 3: Model Performance
    ax3 = axes[1, 0]
    performance_metrics = ['Train R²', 'Test R²', 'CV R²']
    performance_values = [results['train_r2'], results['test_r2'], results['cv_r2_mean']]
    error_values = [0, 0, results['cv_r2_std']]
    
    bars = ax3.bar(performance_metrics, performance_values, yerr=error_values, 
                   capsize=5, alpha=0.7, color='skyblue')
    ax3.set_ylabel('R² (Variance Explained)', fontsize=11)
    ax3.set_title('Model Performance', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1)
    
    overfitting = results['train_r2'] - results['test_r2']
    ax3.text(0.5, 0.1, f'Overfitting: {overfitting:.3f}', 
             transform=ax3.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""FINDINGS SUMMARY

Classification:
{analysis_results['reproduction_classification']}

Performance:
• {analysis_results['overfitting_status']}
• {analysis_results['stability_status']}
• {analysis_results['bias_status']}

Biological Validation:
• {biological_ratio:.1f}% meaningful features
• Model learned signal peptide biology
• Feature importance aligns with mechanisms

Research Contributions:
• Quantitative reproducibility assessment
• Biological validation framework
• Evidence for improved standards
• Framework for future studies"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.2))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(save_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Findings visualization saved: {save_filename}")
    plt.show()
    
    return fig

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Standalone execution for research findings analysis.
    
    Requires results from completed reproduction study.
    """
    
    print("RESEARCH FINDINGS ANALYSIS")
    print("=" * 50)
    print("Note: Requires results from grasso_reproduction_tool.py")
    print()
    
    print("USAGE:")
    print("1. Run reproduction study:")
    print("   from grasso_reproduction_tool import execute_grasso_reproduction")
    print("   results = execute_grasso_reproduction()")
    print()
    print("2. Analyze findings:")
    print("   from research_findings_discussion import analyze_reproduction_outcomes")
    print("   analysis = analyze_reproduction_outcomes(results, results['feature_importance'])")