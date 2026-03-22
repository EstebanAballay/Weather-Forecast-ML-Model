# DONT WORRY IF THE IMPORTS ARE NOT FOUND, THEY EXECUTE INSIDE THE VENV
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "outputs/figures/bivariate"


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  📊 Saved {path}")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Full correlation heatmap of all numerical features."""
    num_df = df.select_dtypes(include=[np.number])

    drop_cols = [c for c in ["temperature_fahrenheit", "feels_like_fahrenheit",
                              "wind_mph", "pressure_in", "precip_in",
                              "visibility_miles", "gust_mph", "last_updated_epoch"]
                 if c in num_df.columns]
    num_df = num_df.drop(columns=drop_cols, errors="ignore")

    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt=".2f",
                linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 7}, square=True)
    ax.set_title("Correlation Matrix — Numerical Features", fontsize=18, fontweight="bold", pad=20)
    fig.tight_layout()
    _save(fig, "correlation_heatmap")

    print("\n  🔗 Top 15 strongest correlations:")
    corr_pairs = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for c1, c2, val in corr_pairs[:15]:
        print(f"     {c1:<35} ↔ {c2:<35} r={val:+.3f}")


def plot_temp_vs_pressure(df: pd.DataFrame):
    """Temperature vs Pressure scatter with regression line."""
    data = df[["temperature_celsius", "pressure_mb"]].dropna()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    sample = data.sample(min(5000, len(data)), random_state=42)
    ax.scatter(sample["temperature_celsius"], sample["pressure_mb"],
               alpha=0.3, s=10, color="#E74C3C", edgecolors="none")

    z = np.polyfit(data["temperature_celsius"], data["pressure_mb"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data["temperature_celsius"].min(), data["temperature_celsius"].max(), 100)
    ax.plot(x_line, p(x_line), color="#2ECC71", linewidth=2.5, linestyle="--", label="Trend line")

    r = data["temperature_celsius"].corr(data["pressure_mb"])
    ax.text(0.03, 0.97, f"r = {r:.3f}\nn = {len(data):,}",
            transform=ax.transAxes, verticalalignment="top",
            fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Pressure (mb)", fontsize=12)
    ax.set_title("Temperature vs Pressure", fontsize=16, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "scatter_temperature_celsius__vs__pressure_mb")


def run(df: pd.DataFrame):
    """Execute bivariate analysis."""
    print("\n" + "=" * 60)
    print("  Phase 3: BIVARIATE ANALYSIS")
    print("=" * 60)

    print("\n  ▸ Correlation heatmap...")
    plot_correlation_heatmap(df)

    print("\n  ▸ Temperature vs Pressure...")
    plot_temp_vs_pressure(df)

    print("\n  ✅ Bivariate analysis complete.")
