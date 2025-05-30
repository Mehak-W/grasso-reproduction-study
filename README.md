# Grasso et al. (2023) Reproduction Study

## Overview

Computational reproduction of machine learning methodology from Grasso et al. (2023) "Signal Peptide Efficiency: From High-Throughput Data to Prediction and Explanation" published in *ACS Synthetic Biology*.

### Research Objectives
1. **Science Goal**: Reproduce Grasso et al. Random Forest methodology for signal peptide efficiency prediction
2. **Science Goal**: Identify factors affecting computational reproducibility in biological machine learning
3. **Science Goal**: Provide framework for reproduction studies in computational biology
4. **Learning Goal**: Demonstrate advanced Python and data science skills in biological applications

## Scientific Background

Signal peptides are short amino acid sequences (15-30 residues) that direct proteins to the secretory pathway in bacterial cells. The efficiency of this process varies significantly between sequences, affecting protein production in biotechnology applications.

The original Grasso study developed a Random Forest regression model to predict signal peptide secretion efficiency using 156 physicochemical features derived from experimental screening of ~12,000 variants in *Bacillus subtilis*.

## Repository Structure

```
grasso-reproduction-study/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── config.py                          # Configuration parameters
├── grasso_reproduction_tool.py        # Main reproduction analysis
├── data_verification_tool.py          # Data integrity verification
├── research_findings_discussion.py    # Results interpretation
├── quick_run.py                       # Simple execution interface
├── sb2c00328_si_011.csv              # Grasso dataset (CSV format)
│   OR sb2c00328_si_011.xlsx          # Grasso dataset (Excel format)
└── data/                              # Additional data storage (optional)
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Required packages listed in `requirements.txt`

### Installation
```bash
# Clone or download repository
cd grasso-reproduction-study

# Install dependencies
pip install -r requirements.txt
```

### Data Requirements
Place the Grasso dataset in the main project directory. The analysis supports both formats:

**CSV Format (Recommended):**
- File: `sb2c00328_si_011.csv`
- Faster loading and processing
- Should contain ~11,643 rows and ~198 columns

**Excel Format (Original):**
- File: `sb2c00328_si_011.xlsx`
- Original supplementary file from Grasso et al.
- Uses 'Library_w_Bins_and_WA' sheet (second sheet)
- Requires openpyxl package (included in requirements.txt)

**Dataset Contents:**
- 11,643 signal peptide variants with experimental data
- 156 validated physicochemical features
- WA (weighted average) secretion efficiency scores
- Signal peptide amino acid sequences (SP_aa)

## Usage

### Quick Start (Recommended)
```python
# Simple interactive execution
python quick_run.py
# Select option 1 for complete analysis
```

### Basic Reproduction Analysis
```python
# Execute complete reproduction study
python grasso_reproduction_tool.py

# Or in Python/Jupyter:
from grasso_reproduction_tool import execute_grasso_reproduction
results = execute_grasso_reproduction()
```

### Data Verification (Recommended First Step)
```python
# Verify data integrity before analysis
python data_verification_tool.py

# Or programmatically:
from data_verification_tool import comprehensive_data_verification
verified_data = comprehensive_data_verification()
```

## Methodology

### Feature Set
Uses exactly 156 validated physicochemical features from Grasso supplementary Table S2:
- **N-region features**: Amino-terminal region properties (24 features)
- **H-region features**: Hydrophobic core properties (14 features)  
- **C-region features**: Carboxy-terminal region properties (29 features)
- **Ac-region features**: Post-cleavage region properties (25 features)
- **SP-region features**: Global signal peptide properties (24 features)
- **Cleavage site features**: Position-specific amino acid indicators (40 features)

### Preprocessing Pipeline
1. **Quality Control**: Applies Grasso filtering criteria
   - Non-missing WA scores and sequences
   - Signal peptide length: 10-40 amino acids
   - WA values: 1.0-10.0 (valid experimental range)

2. **Stratified Sampling**: Ensures representative test sets across performance ranges

3. **Intelligent Preprocessing**: Domain-specific data preparation
   - Feature scaling (if scale differences > 100×)
   - Target transformation (if distribution significantly skewed)

4. **Model Training**: Random Forest with exact Grasso hyperparameters
   - 75 estimators, max_depth=25
   - Optimized for biological feature interactions

### Evaluation Metrics
- **Mean Squared Error (MSE)**: Primary reproduction benchmark
- **R-squared (R²)**: Variance explained assessment  
- **Mean Absolute Error (MAE)**: Interpretable error metric
- **Cross-validation**: Model stability evaluation
- **Feature importance**: Biological validation analysis

## Results

### Performance Benchmarks
- **Target MSE**: 1.22 WA² (Grasso et al. original result)
- **Achieved MSE**: [Results from analysis]
- **Biological validation**: Feature importance alignment with known signal peptide biology

### Outputs Generated
1. **Comprehensive visualization**: `grasso_reproduction_results.png` (4-panel analysis figure)
2. **Detailed metrics**: Complete performance assessment in console output
3. **Feature importance**: Biological validation analysis
4. **Research findings**: Optional detailed interpretation with `research_findings_analysis.png`

## Key Features

### Computational Biology Best Practices
- Exact feature replication from published study
- Rigorous data quality assessment and verification
- Domain-specific preprocessing for biological data
- Comprehensive biological validation of results

### Professional Implementation
- Extensive documentation explaining methodology and rationale
- Progress monitoring and transparent reporting
- Error handling and troubleshooting guidance
- Publication-quality visualization and analysis

### Reproducibility Focus
- Deterministic results with fixed random seeds
- Complete methodology documentation
- Data verification tools for quality assurance
- Comparison framework against published benchmarks

## Technical Requirements

### Python Environment
```python
# Core dependencies
pandas>=1.5.0      # Data manipulation and analysis
numpy>=1.21.0      # Numerical computing
scikit-learn>=1.1.0 # Machine learning algorithms
scipy>=1.9.0       # Statistical analysis
matplotlib>=3.5.0  # Visualization

# Excel file support
openpyxl>=3.0.0    # For Excel file reading (.xlsx format)

# Optional enhancements
seaborn>=0.11.0    # Advanced statistical plots
jupyter>=1.0.0     # Notebook execution
```

### Hardware Requirements
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 100MB for code and results
- **CPU**: Modern multi-core processor recommended for Random Forest training

## Troubleshooting

### Common Issues

**Data Loading Problems**
```bash
# Run data verification first
python data_verification_tool.py
```

**Missing Dependencies**
```bash
# Install all requirements
pip install -r requirements.txt
```

**Memory Issues**
- Reduce dataset size for testing
- Close other applications during analysis
- Consider running on more powerful hardware

### Support
For technical issues or questions about the methodology, refer to:
1. Comprehensive code documentation and comments
2. Data verification tool outputs
3. Original Grasso et al. (2023) publication and supplementary materials

## Citation

If using this reproduction study in academic work, please cite:

```
Wadhwa, M. (2024). Computational Reproduction of Grasso et al. (2023) 
Signal Peptide Efficiency Prediction Methodology. 
GitHub repository: [repository URL]
```

**Original Study:**
```
Grasso, S., et al. (2023). Signal Peptide Efficiency: From High-Throughput 
Data to Prediction and Explanation. ACS Synthetic Biology, 12(4), 1064-1077.
DOI: 10.1021/acssynbio.2c00328
```

## License

This reproduction study is provided for academic and research purposes. The original Grasso methodology and dataset are subject to their respective licenses and terms of use.

## Author

**Mehak Wadhwa**  
Fordham University  
Computational Biology Research

---

## Project Status

✅ **Completed Components:**
- Exact methodology reproduction with 156 validated features
- Comprehensive data verification and quality assessment  
- Professional visualization and analysis pipeline
- Detailed documentation for reproducibility

📊 **Research Contributions:**
- Quantitative assessment of computational reproducibility challenges
- Framework for biological machine learning reproduction studies
- Professional implementation suitable for academic presentation
- Complete analysis ready for scientific documentation