# DONT WORRY IF THE IMPORTS ARE NOT FOUND, THEY EXECUTE INSIDE THE VENV
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "outputs/figures/geography"

# Approximate continent mapping by country name
# (major countries — fallback to longitude-based grouping for unmapped)
CONTINENT_MAP = {
    # Asia
    "China": "Asia", "India": "Asia", "Japan": "Asia", "South Korea": "Asia",
    "Indonesia": "Asia", "Thailand": "Asia", "Vietnam": "Asia", "Malaysia": "Asia",
    "Philippines": "Asia", "Bangladesh": "Asia", "Pakistan": "Asia", "Iran": "Asia",
    "Iraq": "Asia", "Saudi Arabia": "Asia", "Turkey": "Asia", "Israel": "Asia",
    "United Arab Emirates": "Asia", "Qatar": "Asia", "Kuwait": "Asia", "Oman": "Asia",
    "Afghanistan": "Asia", "Myanmar": "Asia", "Cambodia": "Asia", "Laos": "Asia",
    "Nepal": "Asia", "Sri Lanka": "Asia", "Mongolia": "Asia", "Kazakhstan": "Asia",
    "Uzbekistan": "Asia", "Tajikistan": "Asia", "Kyrgyzstan": "Asia",
    "Turkmenistan": "Asia", "Lebanon": "Asia", "Jordan": "Asia", "Syria": "Asia",
    "Yemen": "Asia", "Bahrain": "Asia", "Singapore": "Asia", "Brunei": "Asia",
    "Timor-Leste": "Asia", "Bhutan": "Asia", "Maldives": "Asia",
    "North Korea": "Asia", "Taiwan": "Asia",
    # Europe
    "United Kingdom": "Europe", "France": "Europe", "Germany": "Europe",
    "Italy": "Europe", "Spain": "Europe", "Portugal": "Europe", "Netherlands": "Europe",
    "Belgium": "Europe", "Switzerland": "Europe", "Austria": "Europe",
    "Sweden": "Europe", "Norway": "Europe", "Denmark": "Europe", "Finland": "Europe",
    "Poland": "Europe", "Czech Republic": "Europe", "Czechia": "Europe",
    "Hungary": "Europe", "Romania": "Europe", "Greece": "Europe", "Ireland": "Europe",
    "Croatia": "Europe", "Serbia": "Europe", "Bulgaria": "Europe", "Ukraine": "Europe",
    "Russia": "Europe", "Slovakia": "Europe", "Slovenia": "Europe", "Lithuania": "Europe",
    "Latvia": "Europe", "Estonia": "Europe", "Iceland": "Europe", "Luxembourg": "Europe",
    "Malta": "Europe", "Cyprus": "Europe", "Albania": "Europe", "Montenegro": "Europe",
    "Bosnia And Herzegovina": "Europe", "North Macedonia": "Europe", "Moldova": "Europe",
    "Belarus": "Europe", "Georgia": "Europe", "Armenia": "Europe", "Azerbaijan": "Europe",
    # Africa
    "Nigeria": "Africa", "South Africa": "Africa", "Egypt": "Africa", "Kenya": "Africa",
    "Ethiopia": "Africa", "Ghana": "Africa", "Tanzania": "Africa", "Morocco": "Africa",
    "Algeria": "Africa", "Cameroon": "Africa", "Ivory Coast": "Africa",
    "Madagascar": "Africa", "Mozambique": "Africa", "Angola": "Africa", "Sudan": "Africa",
    "Uganda": "Africa", "Senegal": "Africa", "Congo": "Africa", "Zambia": "Africa",
    "Zimbabwe": "Africa", "Tunisia": "Africa", "Libya": "Africa", "Rwanda": "Africa",
    "Mali": "Africa", "Burkina Faso": "Africa", "Niger": "Africa", "Chad": "Africa",
    "Somalia": "Africa", "Benin": "Africa", "Togo": "Africa", "Sierra Leone": "Africa",
    "Liberia": "Africa", "Mauritania": "Africa", "Eritrea": "Africa", "Gambia": "Africa",
    "Botswana": "Africa", "Namibia": "Africa", "Gabon": "Africa",
    "Lesotho": "Africa", "Equatorial Guinea": "Africa", "Mauritius": "Africa",
    "Eswatini": "Africa", "Djibouti": "Africa", "Comoros": "Africa",
    "Cape Verde": "Africa", "Sao Tome And Principe": "Africa", "Seychelles": "Africa",
    # Americas
    "United States of America": "Americas", "United States": "Americas",
    "Canada": "Americas", "Mexico": "Americas", "Brazil": "Americas",
    "Argentina": "Americas", "Colombia": "Americas", "Peru": "Americas",
    "Chile": "Americas", "Venezuela": "Americas", "Ecuador": "Americas",
    "Bolivia": "Americas", "Paraguay": "Americas", "Uruguay": "Americas",
    "Cuba": "Americas", "Dominican Republic": "Americas", "Guatemala": "Americas",
    "Honduras": "Americas", "El Salvador": "Americas", "Nicaragua": "Americas",
    "Costa Rica": "Americas", "Panama": "Americas", "Jamaica": "Americas",
    "Trinidad And Tobago": "Americas", "Haiti": "Americas", "Guyana": "Americas",
    "Suriname": "Americas", "Bahamas": "Americas", "Barbados": "Americas",
    "Belize": "Americas",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania", "Papua New Guinea": "Oceania",
    "Fiji": "Oceania", "Samoa": "Oceania", "Tonga": "Oceania",
    "Vanuatu": "Oceania", "Solomon Islands": "Oceania", "Micronesia": "Oceania",
    "Kiribati": "Oceania", "Palau": "Oceania", "Marshall Islands": "Oceania",
    "Tuvalu": "Oceania", "Nauru": "Oceania",
}

CONTINENT_COLORS = {
    "Asia": "#E74C3C",
    "Europe": "#3498DB",
    "Africa": "#F39C12",
    "Americas": "#2ECC71",
    "Oceania": "#9B59B6",
    "Other": "#7F8C8D",
}


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  📊 Saved {path}")


def _assign_continent(df):
    """Assign continent based on country name."""
    df = df.copy()
    df["continent"] = df["country"].map(CONTINENT_MAP).fillna("Other")
    return df


def plot_hemisphere_comparison(df: pd.DataFrame):
    """Boxplots comparing Northern vs Southern hemisphere."""
    if "latitude" not in df.columns:
        return

    df_h = df.copy()
    df_h["hemisphere"] = np.where(df_h["latitude"] >= 0, "Northern", "Southern")

    metrics = [
        ("temperature_celsius", "Temperature (°C)"),
        ("humidity", "Humidity (%)"),
        ("pressure_mb", "Pressure (mb)"),
        ("uv_index", "UV Index"),
    ]
    existing = [(c, l) for c, l in metrics if c in df_h.columns]

    fig, axes = plt.subplots(1, len(existing), figsize=(5 * len(existing), 6))
    if len(existing) == 1:
        axes = [axes]

    fig.suptitle("Northern vs Southern Hemisphere", fontsize=16, fontweight="bold", y=1.02)

    for ax, (col, label) in zip(axes, existing):
        data_n = df_h[df_h["hemisphere"] == "Northern"][col].dropna()
        data_s = df_h[df_h["hemisphere"] == "Southern"][col].dropna()
        bp = ax.boxplot([data_n, data_s], labels=["Northern", "Southern"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#5DADE2")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#F5B041")
        bp["boxes"][1].set_alpha(0.7)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, "hemisphere_comparison")


def plot_air_quality_correlation_by_region(df: pd.DataFrame):
    """Air quality correlation with all variables + geographic grouping."""
    df = _assign_continent(df)

    aq_col = "air_quality_PM2.5"
    if aq_col not in df.columns:
        aq_col = "air_quality_us-epa-index"
        if aq_col not in df.columns:
            print("  ⚠️  No air quality column found")
            return

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle("Air Quality Analysis — Correlations & Geographic Distribution",
                 fontsize=18, fontweight="bold", y=1.02)

    # ── 1. Correlation of PM2.5 with ALL other numeric variables ──
    ax = axes[0, 0]
    num_df = df.select_dtypes(include=[np.number])
    drop_redundant = ["temperature_fahrenheit", "feels_like_fahrenheit",
                      "wind_mph", "pressure_in", "precip_in",
                      "visibility_miles", "gust_mph", "last_updated_epoch"]
    num_df = num_df.drop(columns=[c for c in drop_redundant if c in num_df.columns], errors="ignore")

    if aq_col in num_df.columns:
        correlations = num_df.corr()[aq_col].drop(aq_col, errors="ignore").sort_values()
        colors = ["#E74C3C" if v < 0 else "#2ECC71" for v in correlations.values]
        ax.barh(correlations.index, correlations.values, color=colors, edgecolor="white", alpha=0.85)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel(f"Correlation with {aq_col}", fontsize=11)
        ax.set_title(f"{aq_col} Correlation with\nAll Variables", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

    # ── 2. Average PM2.5 by continent ──
    ax = axes[0, 1]
    continent_aq = df.groupby("continent")[aq_col].mean().sort_values(ascending=True)
    continent_aq = continent_aq.drop("Other", errors="ignore")
    bar_colors = [CONTINENT_COLORS.get(c, "#7F8C8D") for c in continent_aq.index]
    bars = ax.barh(continent_aq.index, continent_aq.values,
                   color=bar_colors, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, continent_aq.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel(f"Average {aq_col}", fontsize=11)
    ax.set_title(f"Average Air Quality by Continent\n(Higher = Worse)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # ── 3. Top 20 most polluted cities (focus on Asia) ──
    ax = axes[1, 0]
    city_aq = df.groupby(["location_name", "country", "continent"]).agg(
        avg_pm25=(aq_col, "mean")
    ).reset_index().sort_values("avg_pm25", ascending=False).head(20)

    bar_colors = [CONTINENT_COLORS.get(c, "#7F8C8D") for c in city_aq["continent"]]
    labels = [f"{row['location_name']} ({row['country'][:15]})" for _, row in city_aq.iterrows()]
    bars = ax.barh(labels[::-1], city_aq["avg_pm25"].values[::-1],
                   color=bar_colors[::-1], edgecolor="white", alpha=0.85)
    ax.set_xlabel(f"Average {aq_col}", fontsize=11)
    ax.set_title("Top 20 Most Polluted Capital Cities", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Add legend for continents
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CONTINENT_COLORS[c], label=c)
                       for c in ["Asia", "Africa", "Americas", "Europe", "Oceania"]]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # ── 4. World scatter map colored by PM2.5 ──
    ax = axes[1, 1]
    if "latitude" in df.columns and "longitude" in df.columns:
        city_avg = df.groupby(["location_name", "latitude", "longitude", "continent"]).agg(
            avg_pm25=(aq_col, "mean")
        ).reset_index()

        vmax_val = city_avg["avg_pm25"].quantile(0.95)
        scatter = ax.scatter(city_avg["longitude"], city_avg["latitude"],
                             c=city_avg["avg_pm25"], vmax=vmax_val,
                             cmap="YlOrRd", s=40, alpha=0.8,
                             edgecolors="gray", linewidth=0.3)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(f"Avg {aq_col}", fontsize=10)
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title("World Air Quality Map", fontsize=13, fontweight="bold")
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    _save(fig, "air_quality_geographic_analysis")


def plot_elevation_vs_rainfall(df: pd.DataFrame):
    """Elevation (pressure as proxy) vs rainfall across all capitals."""
    if "pressure_mb" not in df.columns or "precip_mm" not in df.columns:
        return

    df = _assign_continent(df)

    # Average pressure and precipitation per city
    city_avg = df.groupby(["location_name", "country", "continent", "latitude", "longitude"]).agg(
        avg_pressure=("pressure_mb", "mean"),
        avg_precip=("precip_mm", "mean"),
        total_precip=("precip_mm", "sum"),
    ).reset_index()

    # Estimate elevation: sea level ≈ 1013 mb. Lower pressure = higher elevation
    city_avg["estimated_elevation_m"] = (1013.25 - city_avg["avg_pressure"]) * 8.3

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Terrain Elevation (Pressure Proxy) vs Rainfall — All Capitals",
                 fontsize=17, fontweight="bold", y=1.02)

    # ── 1. Elevation vs Average Daily Precipitation ──
    ax = axes[0]
    for continent in ["Asia", "Europe", "Africa", "Americas", "Oceania"]:
        mask = city_avg["continent"] == continent
        ax.scatter(city_avg.loc[mask, "estimated_elevation_m"],
                   city_avg.loc[mask, "avg_precip"],
                   color=CONTINENT_COLORS.get(continent, "#7F8C8D"),
                   s=50, alpha=0.7, edgecolors="white", linewidth=0.5,
                   label=continent)

    ax.set_xlabel("Estimated Elevation (m) — based on pressure", fontsize=12)
    ax.set_ylabel("Avg Daily Precipitation (mm)", fontsize=12)
    ax.set_title("Elevation vs Average Rainfall", fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 2. Elevation vs Total Accumulated Precipitation ──
    ax = axes[1]
    for continent in ["Asia", "Europe", "Africa", "Americas", "Oceania"]:
        mask = city_avg["continent"] == continent
        ax.scatter(city_avg.loc[mask, "estimated_elevation_m"],
                   city_avg.loc[mask, "total_precip"],
                   color=CONTINENT_COLORS.get(continent, "#7F8C8D"),
                   s=50, alpha=0.7, edgecolors="white", linewidth=0.5,
                   label=continent)

    # Label outliers (top 5 rainiest)
    top_rain = city_avg.nlargest(5, "total_precip")
    for _, row in top_rain.iterrows():
        ax.annotate(row["location_name"],
                    (row["estimated_elevation_m"], row["total_precip"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8,
                    fontweight="bold", color="#2C3E50")

    ax.set_xlabel("Estimated Elevation (m) — based on pressure", fontsize=12)
    ax.set_ylabel("Total Accumulated Precipitation (mm)", fontsize=12)
    ax.set_title("Elevation vs Total Rainfall", fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "elevation_vs_rainfall")


def run(df: pd.DataFrame):
    """Execute geographic impact analysis."""
    print("\n" + "=" * 60)
    print("  Phase 5: GEOGRAPHIC IMPACT ON CLIMATE")
    print("=" * 60)

    print("\n  ▸ Hemisphere comparison...")
    plot_hemisphere_comparison(df)

    print("\n  ▸ Air quality correlation & geographic analysis...")
    plot_air_quality_correlation_by_region(df)

    print("\n  ▸ Elevation vs rainfall...")
    plot_elevation_vs_rainfall(df)

    print("\n  ✅ Geographic analysis complete.")
