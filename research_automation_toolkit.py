# ================================================
# Research Analytics Automation Toolkit
# Author: Mit Mehta | MSMRA @ MSU Eli Broad
# Description: Automates survey data processing,
# segmentation analysis, and research reporting
# ================================================

import pandas as pd
import numpy as np
from scipy import stats
import json
import os
from datetime import datetime


# ================================================
# MODULE 1: DATA INGESTION & CLEANING
# ================================================

def load_survey_data(filepath):
    """
    Load and clean raw survey data from CSV export.
    Handles common AYTM/Qualtrics export formats.
    
    Args:
        filepath: path to raw survey CSV file
    Returns:
        cleaned pandas DataFrame
    """
    df = pd.read_csv(filepath)
    
    # Drop empty rows and duplicate respondents
    df = df.dropna(how='all')
    df = df.drop_duplicates()
    
    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace(':', '')
    
    print(f"✓ Loaded {len(df)} respondents across {len(df.columns)} variables")
    return df


def validate_responses(df, min_responses=100):
    """
    Validate survey dataset quality.
    Flags low response counts and high missing data.
    
    Args:
        df: cleaned DataFrame
        min_responses: minimum acceptable sample size
    Returns:
        validation report as dictionary
    """
    report = {
        'total_responses': len(df),
        'meets_minimum': len(df) >= min_responses,
        'missing_data_pct': round(df.isnull().sum().sum() / df.size * 100, 2),
        'complete_responses': len(df.dropna()),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    
    # Flag columns with high missing data
    high_missing = df.columns[df.isnull().mean() > 0.1].tolist()
    report['high_missing_columns'] = high_missing
    
    print(f"✓ Validation complete — {report['total_responses']} responses, "
          f"{report['missing_data_pct']}% missing data")
    return report


# ================================================
# MODULE 2: DEMOGRAPHIC PROFILING
# ================================================

def demographic_profile(df, demo_cols):
    """
    Generate demographic breakdown for survey respondents.
    
    Args:
        df: cleaned DataFrame
        demo_cols: list of demographic column names
    Returns:
        dictionary of frequency tables per demographic
    """
    profile = {}
    
    for col in demo_cols:
        if col in df.columns:
            freq = df[col].value_counts()
            pct = df[col].value_counts(normalize=True).round(3) * 100
            profile[col] = pd.DataFrame({
                'count': freq,
                'percentage': pct
            })
            print(f"✓ Profiled: {col}")
    
    return profile


def segment_respondents(df, segment_col, value_col):
    """
    Segment respondents and compare mean scores across segments.
    
    Args:
        df: cleaned DataFrame
        segment_col: column to segment by (e.g. 'gender', 'income_bracket')
        value_col: numeric column to compare across segments
    Returns:
        segmentation summary DataFrame
    """
    summary = df.groupby(segment_col)[value_col].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('std', 'std'),
        ('count', 'count')
    ]).round(2)
    
    print(f"✓ Segmented {value_col} across {segment_col} — "
          f"{len(summary)} segments identified")
    return summary


# ================================================
# MODULE 3: STATISTICAL ANALYSIS
# ================================================

def run_ttest(df, group_col, value_col, group1, group2):
    """
    Run independent samples t-test between two groups.
    Commonly used in survey analysis to compare segment means.
    
    Args:
        df: cleaned DataFrame
        group_col: column defining the groups
        value_col: numeric column to compare
        group1, group2: group labels to compare
    Returns:
        t-statistic, p-value, and interpretation
    """
    g1 = df[df[group_col] == group1][value_col].dropna()
    g2 = df[df[group_col] == group2][value_col].dropna()
    
    t_stat, p_value = stats.ttest_ind(g1, g2)
    
    result = {
        'group1': group1,
        'group2': group2,
        'group1_mean': round(g1.mean(), 3),
        'group2_mean': round(g2.mean(), 3),
        't_statistic': round(t_stat, 3),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
        'interpretation': (
            f"Significant difference between {group1} and {group2} (p={round(p_value,4)})"
            if p_value < 0.05
            else f"No significant difference between {group1} and {group2} (p={round(p_value,4)})"
        )
    }
    
    print(f"✓ T-test: {result['interpretation']}")
    return result


def calculate_cronbach_alpha(df, scale_cols):
    """
    Calculate Cronbach's Alpha for internal consistency
    of multi-item survey scales.
    
    Args:
        df: cleaned DataFrame
        scale_cols: list of columns forming the scale
    Returns:
        Cronbach's alpha coefficient and interpretation
    """
    scale_df = df[scale_cols].dropna()
    n_items = len(scale_cols)
    
    item_variances = scale_df.var(axis=0, ddof=1).sum()
    total_variance = scale_df.sum(axis=1).var(ddof=1)
    
    alpha = (n_items / (n_items - 1)) * (1 - item_variances / total_variance)
    alpha = round(alpha, 3)
    
    if alpha >= 0.9:
        interpretation = "Excellent"
    elif alpha >= 0.8:
        interpretation = "Good"
    elif alpha >= 0.7:
        interpretation = "Acceptable"
    elif alpha >= 0.6:
        interpretation = "Questionable"
    else:
        interpretation = "Poor"
    
    print(f"✓ Cronbach's Alpha: {alpha} — {interpretation} internal consistency")
    return {'alpha': alpha, 'interpretation': interpretation, 'n_items': n_items}


# ================================================
# MODULE 4: AUTOMATED REPORTING
# ================================================

def generate_summary_report(df, project_name, demo_cols, key_metrics):
    """
    Auto-generate a structured research summary report.
    
    Args:
        df: cleaned DataFrame
        project_name: name of the research project
        demo_cols: demographic columns to profile
        key_metrics: list of key metric columns to summarize
    Returns:
        report dictionary — can be exported to JSON or printed
    """
    report = {
        'project': project_name,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'sample_size': len(df),
        'demographics': {},
        'key_metrics': {},
        'data_quality': validate_responses(df)
    }
    
    # Demographics
    for col in demo_cols:
        if col in df.columns:
            top_group = df[col].value_counts().index[0]
            top_pct = round(df[col].value_counts(normalize=True).iloc[0] * 100, 1)
            report['demographics'][col] = {
                'top_group': top_group,
                'top_group_pct': top_pct,
                'unique_values': df[col].nunique()
            }
    
    # Key metrics
    for metric in key_metrics:
        if metric in df.columns:
            report['key_metrics'][metric] = {
                'mean': round(df[metric].mean(), 2),
                'median': round(df[metric].median(), 2),
                'std': round(df[metric].std(), 2),
                'min': df[metric].min(),
                'max': df[metric].max()
            }
    
    print(f"\n{'='*50}")
    print(f"RESEARCH SUMMARY: {project_name}")
    print(f"{'='*50}")
    print(f"Sample size: {report['sample_size']}")
    print(f"Generated: {report['generated_at']}")
    print(f"{'='*50}\n")
    
    return report


def export_report(report, output_path):
    """
    Export report to JSON file for sharing or archiving.
    
    Args:
        report: report dictionary from generate_summary_report
        output_path: file path for JSON output
    """
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    print(f"✓ Report exported to {output_path}")


# ================================================
# MODULE 5: WORKFLOW AUTOMATION
# ================================================

def run_full_pipeline(filepath, project_name, demo_cols, key_metrics):
    """
    Run the complete research analytics pipeline end-to-end.
    Automates: load → validate → profile → analyze → report
    
    Args:
        filepath: path to raw survey CSV
        project_name: name of the research project
        demo_cols: demographic columns
        key_metrics: key metric columns
    Returns:
        complete analysis report
    """
    print(f"\nStarting pipeline for: {project_name}")
    print("="*50)
    
    # Step 1 - Load and clean data
    df = load_survey_data(filepath)
    
    # Step 2 - Validate
    validation = validate_responses(df)
    if not validation['meets_minimum']:
        print(f"⚠ Warning: Sample size below minimum threshold")
    
    # Step 3 - Profile demographics
    demo_profile = demographic_profile(df, demo_cols)
    
    # Step 4 - Segment analysis
    if len(demo_cols) > 0 and len(key_metrics) > 0:
        segments = segment_respondents(df, demo_cols[0], key_metrics[0])
    
    # Step 5 - Generate report
    report = generate_summary_report(df, project_name, demo_cols, key_metrics)
    
    # Step 6 - Export
    output_path = f"{project_name.lower().replace(' ', '_')}_report.json"
    export_report(report, output_path)
    
    print(f"\n✓ Pipeline complete for {project_name}")
    return report


# ================================================
# EXAMPLE USAGE
# ================================================

if __name__ == "__main__":
    
    # Example: NexGen Archery Donor Study
    report = run_full_pipeline(
        filepath="survey_data.csv",
        project_name="NexGen Archery Donor Insights Study",
        demo_cols=["gender", "region", "income_bracket", "age"],
        key_metrics=["donation_likelihood", "brand_awareness", "interest_score"]
    )
