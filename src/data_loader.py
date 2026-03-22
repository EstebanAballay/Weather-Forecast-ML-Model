# DONT WORRY IF THE IMPORTS ARE NOT FOUND, THEY EXECUTE INSIDE THE VENV
import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "outputs/reports"

# Load the CSV and parse date columns.
def load_data(filepath: str = "data/GlobalWeatherRepository.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["last_updated"])
    return df

def generate_report(df: pd.DataFrame) -> str:
    #each one of these lines make the report, at the end they join al together
    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("  DATA SUMMARY REPORT")
    lines.append(sep)
    #shows the general shape of the dataframe
    lines.append(f"\n Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    #shows the data types of each column
    lines.append(f"{'Column':<42} {'Dtype':<18} {'Non-Null':>10}")
    lines.append("-" * 70)
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        lines.append(f"{col:<42} {dtype:<18} {non_null:>10,}")

    #shows the missing values in each column
    lines.append(f"\n{sep}")
    lines.append("  MISSING VALUES")
    lines.append(sep)
    missing = df.isnull().sum()
    missing_percentage = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_percentage
    }).sort_values("Missing Count", ascending=False)

    has_missing = missing_df[missing_df["Missing Count"] > 0]
    if has_missing.empty:
        lines.append("\n No missing values found in any column.\n")
    else:
        lines.append(f"\n  {len(has_missing)} columns have missing values:\n")
        lines.append(f"{'Column':<42} {'Count':>10} {'%':>8}")
        lines.append("-" * 60)
        for col, row in has_missing.iterrows():
            lines.append(f"{col:<42} {int(row['Missing Count']):>10,} {row['Missing %']:>7.2f}%")

    # shows the statistical summary 
    lines.append(f"\n{sep}")
    lines.append("  NUMERICAL SUMMARY (describe)")
    lines.append(sep)
    #generates the summary and transposes it 
    num_desc = df.describe().T
    num_desc = num_desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]] #I select what I want to show
    lines.append(num_desc.to_string())

    #shows the categorical summary 
    lines.append(f"\n{sep}")
    lines.append("  CATEGORICAL COLUMNS — TOP VALUES")
    lines.append(sep)
    #it gets the categories by taking all the columns with text format
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    #For each category I found, I show the top 10 most frequent values
    for col in cat_cols:
        vc = df[col].value_counts().head(10)
        lines.append(f"\n Occurrences of {col} ({df[col].nunique()} unique)")
        lines.append("-" * 40)
        for val, cnt in vc.items():
            pct = cnt / len(df) * 100
            lines.append(f"  {str(val):<30} {cnt:>8,}  ({pct:.1f}%)")

    return "\n".join(lines)

#this command is run from the main.py, it is like the "gestor" of the file
def run(df: pd.DataFrame):
    """Execute the data loading & exploration phase."""
    print("\n" + "=" * 60)
    print("  Phase 1: DATA LOADING & EXPLORATION")
    print("=" * 60)

    report = generate_report(df)
    print(report)

    # Save report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "data_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n💾 Report saved to {report_path}")

    return df
