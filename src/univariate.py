# DONT WORRY IF THE IMPORTS ARE NOT FOUND, THEY EXECUTE INSIDE THE VENV
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

#save directory
OUTPUT_DIR = "outputs/figures/univariate"


def _add_logo(fig):
    try:
        logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "logo.png")
        if os.path.exists(logo_path):
            logo = plt.imread(logo_path)
            newax = fig.add_axes([0.85, 0.95, 0.12, 0.12], anchor='NE', zorder=10)
            newax.imshow(logo)
            newax.axis('off')
    except Exception as e:
        pass

def _save(fig, name):
    _add_logo(fig)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  📊 Saved {path}")


def _plot_distribution(df, col_name, label, color):
    """Generic histogram + KDE + boxplot for a single column."""
    data = df[col_name].dropna()
    if data.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Distribution of {label}", fontsize=16, fontweight="bold", y=1.02)

    # Histogram + KDE
    ax1 = axes[0]
    ax1.hist(data, bins=50, color=color, alpha=0.7, edgecolor="white", density=True)
    data.plot.kde(ax=ax1, color="black", linewidth=2)
    ax1.set_xlabel(label, fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Histogram + KDE")

    stats_text = f"Mean: {data.mean():.2f}\nMedian: {data.median():.2f}\nStd: {data.std():.2f}\nSkew: {data.skew():.2f}"
    ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes,
             verticalalignment="top", horizontalalignment="right",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    # Box plot
    ax2 = axes[1]
    ax2.boxplot(data, vert=True, patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.7),
                medianprops=dict(color="black", linewidth=2))
    ax2.set_ylabel(label, fontsize=12)
    ax2.set_title("Box Plot")

    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    outliers = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
    ax2.text(0.97, 0.97, f"Outliers: {outliers:,}", transform=ax2.transAxes,
             verticalalignment="top", horizontalalignment="right",
             fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    fig.tight_layout()
    _save(fig, f"dist_{col_name.replace('.', '_').replace('-', '_')}")

#Main.py calls this function
def run(df: pd.DataFrame):
    """Execute univariate analysis — only humidity and PM2.5."""
    print("\n" + "=" * 60)
    print("  Phase 2: UNIVARIATE ANALYSIS")
    print("=" * 60)

    #private functions not accesible from outside
    _plot_distribution(df, "air_quality_PM2.5", "Air Quality PM2.5 (µg/m³)", "#E67E22")

    print("\n  ✅ Univariate analysis complete.")
