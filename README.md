# Research Analytics Automation Toolkit
### Python Toolkit for Survey Data Processing & Research Workflows | Mit Mehta

## Overview
A reusable Python library that automates end-to-end survey data processing,
segmentation analysis, statistical testing, and research reporting.

Built to eliminate 15+ hours/week of manual analytics setup across research
projects — directly inspired by automation work across client studies including
NexGen Archery (n=401) and Archive (n≈400).

## What It Does

### Module 1 — Data Ingestion & Cleaning
- Loads raw survey exports from AYTM and Qualtrics
- Standardizes column names and removes duplicates
- Flags high missing data columns automatically

### Module 2 — Demographic Profiling & Segmentation
- Generates frequency tables for all demographic variables
- Segments respondents and compares mean scores across groups
- Identifies high-value segments automatically

### Module 3 — Statistical Analysis
- Independent samples t-tests for group comparisons
- Cronbach's Alpha for scale reliability testing
- Returns plain-English interpretation alongside statistics

### Module 4 — Automated Reporting
- Generates structured summary reports from raw data
- Exports to JSON for sharing and archiving
- Includes data quality validation built-in

### Module 5 — Full Pipeline Automation
- Runs complete load → validate → profile → analyze → report workflow
- Single function call replaces hours of manual setup
- Reusable across any survey dataset

## Key Results
- 60% reduction in content production time
- 50% reduction in per-project analytics setup time
- Eliminated 15+ hrs/week of manual data processing
- Reused across 4+ research projects

## Tech Stack
Python · pandas · NumPy · SciPy · JSON

## Usage

```python
from research_automation_toolkit import run_full_pipeline

report = run_full_pipeline(
    filepath="survey_data.csv",
    project_name="NexGen Archery Donor Insights Study",
    demo_cols=["gender", "region", "income_bracket", "age"],
    key_metrics=["donation_likelihood", "brand_awareness", "interest_score"]
)
```

## Connection to Real Projects

**NexGen Archery (completed)** — Donor insights study (n=401).
This toolkit was used to automate demographic profiling and segmentation
analysis, identifying 3 high-value donor segments that reshaped the
client's go-to-market messaging strategy.

**Archive (in progress)** — Consumer insights study (n≈400) for a B2B
recommerce platform. The pipeline automates survey data cleaning and
reporting workflows for ongoing fieldwork.

## Files
- `research_automation_toolkit.py` — Full Python toolkit with 5 modules

## About
Mit Mehta | MS Marketing Research & Analytics @ MSU Eli Broad (Aug 2026)
Seeking full-time roles in Consumer Insights, Marketing Analytics, and Growth Analytics.
[LinkedIn](https://linkedin.com/in/mitmehta10) | [GitHub](https://github.com/MightyMIT10)
