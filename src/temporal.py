# DONT WORRY IF THE IMPORTS ARE NOT FOUND, THEY EXECUTE INSIDE THE VENV
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "outputs/figures/temporal"

CITY_COLORS = {
    "Buenos Aires": "#3498DB",
    "Canberra": "#E74C3C",
}

# Southern hemisphere seasons
SEASONS = {
    12: "Summer", 1: "Summer", 2: "Summer",
    3: "Autumn",  4: "Autumn",  5: "Autumn",
    6: "Winter",  7: "Winter",  8: "Winter",
    9: "Spring", 10: "Spring", 11: "Spring",
}
SEASON_ORDER = ["Summer", "Autumn", "Winter", "Spring"]
SEASON_COLORS = {
    "Summer": "#E74C3C",
    "Autumn": "#E67E22",
    "Winter": "#3498DB",
    "Spring": "#2ECC71",
}


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  📊 Saved {path}")


def _filter_city(df, city):
    mask = df["location_name"].str.contains(city, case=False, na=False)
    result = df[mask].sort_values("last_updated").copy()
    if result.empty:
        available = df["location_name"].unique().tolist()
        print(f"  ⚠️  '{city}' not found. Available cities (first 20): {available[:20]}")
    return result


def plot_global_warming_trend(df: pd.DataFrame):
    #If it doesn't have the last_updated column aborts the function
    if "last_updated" not in df.columns or "temperature_celsius" not in df.columns:
                    return

    #I group by date,then I take the mean of each day, and finally I reset the index
    #After that process I only get "timestamps" of a day and mean temperature
    daily_global = df.groupby(df["last_updated"].dt.date)["temperature_celsius"].mean().reset_index()
    daily_global.columns = ["date", "avg_temp"] #rename all the columns(just two)
    daily_global["date"] = pd.to_datetime(daily_global["date"])
    daily_global = daily_global.sort_values("date") #sorting to get a cronollogical order of temperatures

    fig, ax = plt.subplots(figsize=(16, 7))

    # Daily average (faded)
    ax.plot(daily_global["date"], daily_global["avg_temp"],
            alpha=0.3, color="#3498DB", linewidth=0.8, label="Daily global avg")

    # Here I create a new column that is the mean temp of 30 days before,grouping all the data in groups of 30
    daily_global["ma30"] = daily_global["avg_temp"].rolling(30, center=True).mean()
    ax.plot(daily_global["date"], daily_global["ma30"],
            color="#E74C3C", linewidth=2.5, label="30-day moving average")

    # Linear regression tendency calcculation
    x_num = (daily_global["date"] - daily_global["date"].min()).dt.days.values
    mask = ~np.isnan(daily_global["avg_temp"].values)
    z = np.polyfit(x_num[mask], daily_global["avg_temp"].values[mask], 1)
    p = np.poly1d(z)
    ax.plot(daily_global["date"], p(x_num),
            color="#2C3E50", linewidth=2, linestyle="--",
            label=f"Linear trend ({z[0]*365:.3f} °C/year)")

    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Global Average Temperature (°C)", fontsize=13)
    ax.set_title("Global Temperature Trend Over Time\n(Average across all capital cities)",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Annotate overall change
    if len(daily_global) > 60:
        first_30 = daily_global.head(30)["avg_temp"].mean()
        last_30 = daily_global.tail(30)["avg_temp"].mean()
        delta = last_30 - first_30
        ax.text(0.97, 0.03,
                f"Start avg: {first_30:.1f}°C\nEnd avg: {last_30:.1f}°C\nΔ = {delta:+.2f}°C",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", alpha=0.9))

    fig.tight_layout()
    _save(fig, "global_warming_trend")


def plot_ba_rain_by_season(df: pd.DataFrame):
    #Buenos Aires precipitation patterns grouped by season
    ba = _filter_city(df, "Buenos Aires")
    if ba.empty or "precip_mm" not in ba.columns:
        print("  ⚠️  No Buenos Aires precipitation data")
        return

    ba = ba.copy()
    ba["month"] = ba["last_updated"].dt.month
    ba["season"] = ba["month"].map(SEASONS)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Buenos Aires — Rainfall Patterns by Season (Southern Hemisphere)",
                 fontsize=16, fontweight="bold", y=1.02)

    #average precipitation per season
    ax = axes[0]
    season_stats = ba.groupby("season")["precip_mm"].agg(["mean", "std", "sum", "count"])
    season_stats = season_stats.reindex(SEASON_ORDER)
    bars = ax.bar(SEASON_ORDER, season_stats["mean"],
                  color=[SEASON_COLORS[s] for s in SEASON_ORDER],
                  edgecolor="white", alpha=0.85)
    ax.errorbar(SEASON_ORDER, season_stats["mean"], yerr=season_stats["std"],
                fmt="none", ecolor="black", capsize=5, alpha=0.5)
    for bar, val in zip(bars, season_stats["mean"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f} mm", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Avg Daily Precipitation (mm)", fontsize=12)
    ax.set_title("Average Daily Rainfall", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # total precipitation per season 
    ax = axes[1]
    bars = ax.bar(SEASON_ORDER, season_stats["sum"],
                  color=[SEASON_COLORS[s] for s in SEASON_ORDER],
                  edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, season_stats["sum"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f} mm", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Total Precipitation (mm)", fontsize=12)
    ax.set_title("Total Accumulated Rainfall", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # rainy days (precip > 0.5mm) per season
    ax = axes[2]
    rainy_days = ba[ba["precip_mm"] > 0.5].groupby("season").size().reindex(SEASON_ORDER, fill_value=0)
    total_days = ba.groupby("season").size().reindex(SEASON_ORDER, fill_value=1)
    rainy_pct = (rainy_days / total_days * 100)

    bars = ax.bar(SEASON_ORDER, rainy_pct,
                  color=[SEASON_COLORS[s] for s in SEASON_ORDER],
                  edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, rainy_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("% Days with Rain", fontsize=12)
    ax.set_title("Rainy Day Frequency", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, "ba_rain_by_season")

#Correlation between rainfall and N/S wind direction in Buenos Aires.
def plot_ba_rain_vs_wind_direction(df: pd.DataFrame):
    #get the ba row and quick check
    ba = _filter_city(df, "Buenos Aires")
    if ba.empty or "wind_direction" not in ba.columns or "precip_mm" not in ba.columns:
        print("  ⚠️  No wind/rain data for Buenos Aires")
        return

    ba = ba.copy()
    north_dirs = ["N", "NNE", "NNW", "NE", "NW"]
    south_dirs = ["S", "SSE", "SSW", "SE", "SW"]
    east_dirs = ["E", "ENE", "ESE"]
    west_dirs = ["W", "WNW", "WSW"]

    def classify_wind(d):
        if d in north_dirs:
            return "North"
        elif d in south_dirs:
            return "South"
        elif d in east_dirs:
            return "East"
        elif d in west_dirs:
            return "West"
        return "Other"

    ba["wind_group"] = ba["wind_direction"].apply(classify_wind)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Buenos Aires — Rainfall vs Wind Direction",
                 fontsize=16, fontweight="bold", y=1.02)

    group_colors = {"North": "#E74C3C", "South": "#3498DB", "East": "#2ECC71", "West": "#F39C12"}
    groups = ["North", "South", "East", "West"]

    # Avg precip by wind direction
    ax = axes[0]
    avg_precip = ba.groupby("wind_group")["precip_mm"].mean().reindex(groups, fill_value=0)
    bars = ax.bar(groups, avg_precip,
                  color=[group_colors[g] for g in groups], edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, avg_precip):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(top=avg_precip.max() * 1.15)
    ax.set_ylabel("Avg Precipitation (mm)", fontsize=12)
    ax.set_title("Avg Rainfall by Wind Direction", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    total = ba.groupby("wind_group").size().reindex(groups, fill_value=0)
    rainy = ba[ba["precip_mm"] > 0.5].groupby("wind_group").size().reindex(groups, fill_value=0)
    rain_pct = (rainy / total * 100).fillna(0)

    bars = ax.bar(groups, rain_pct,
                  color=[group_colors[g] for g in groups], edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, rain_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("% Days with Rain (>0.5mm)", fontsize=12)
    ax.set_title("Rain Probability by Wind Dir", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, "ba_rain_vs_wind_direction")


def plot_ba_vs_Canberra(df: pd.DataFrame):
    """Buenos Aires vs Canberra full comparison (similar latitudes)."""
    ba = _filter_city(df, "Buenos Aires")
    ct = _filter_city(df, "Canberra")

    if ba.empty or ct.empty:
        print("  ⚠️  Missing data for Buenos Aires or Canberra")
        return

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("Buenos Aires vs Canberra — Similar Latitudes Comparison\n"
                 f"(BA ≈ {ba['latitude'].iloc[0]:.1f}°  |  CT ≈ {ct['latitude'].iloc[0]:.1f}°)",
                 fontsize=17, fontweight="bold", y=1.03)

    cities = {"Buenos Aires": ba, "Canberra": ct}
    colors = CITY_COLORS

    # Temperature over time
    ax = axes[0, 0]
    for name, city_df in cities.items():
        ts = city_df.set_index("last_updated")["temperature_celsius"].rolling("14D").mean()
        ax.plot(ts.index, ts.values, color=colors[name], linewidth=2, label=name)
    ax.set_title("Temperature (14-day avg)", fontweight="bold")
    ax.set_ylabel("°C")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Temperature distribution
    ax = axes[0, 1]
    for name, city_df in cities.items():
        data = city_df["temperature_celsius"].dropna()
        ax.hist(data, bins=30, alpha=0.6, color=colors[name], label=name, edgecolor="white", density=True)
        data.plot.kde(ax=ax, color=colors[name], linewidth=2)
    ax.set_title("Temperature Distribution", fontweight="bold")
    ax.set_xlabel("°C")
    ax.legend(fontsize=10)

    # Humidity comparison
    ax = axes[0, 2]
    bp_data = [cities[n]["humidity"].dropna().values for n in ["Buenos Aires", "Canberra"]]
    bp = ax.boxplot(bp_data, labels=["Buenos Aires", "Canberra"], patch_artist=True)
    bp["boxes"][0].set_facecolor(colors["Buenos Aires"])
    bp["boxes"][1].set_facecolor(colors["Canberra"])
    for b in bp["boxes"]:
        b.set_alpha(0.7)
    ax.set_title("Humidity Distribution", fontweight="bold")
    ax.set_ylabel("%")

    # Wind speed comparison
    ax = axes[1, 0]
    bp_data = [cities[n]["wind_kph"].dropna().values for n in ["Buenos Aires", "Canberra"]]
    bp = ax.boxplot(bp_data, labels=["Buenos Aires", "Canberra"], patch_artist=True)
    bp["boxes"][0].set_facecolor(colors["Buenos Aires"])
    bp["boxes"][1].set_facecolor(colors["Canberra"])
    for b in bp["boxes"]:
        b.set_alpha(0.7)
    ax.set_title("Wind Speed Distribution", fontweight="bold")
    ax.set_ylabel("kph")

    # Precipitation comparison
    ax = axes[1, 1]
    for name, city_df in cities.items():
        monthly = city_df.copy()
        monthly["month"] = monthly["last_updated"].dt.month
        month_avg = monthly.groupby("month")["precip_mm"].mean().reindex(range(1, 13))
        months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        ax.plot(range(1, 13), month_avg.values, color=colors[name], linewidth=2.5,
                marker="o", markersize=6, label=name)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months)
    ax.set_title("Monthly Avg Precipitation", fontweight="bold")
    ax.set_ylabel("mm")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Air quality comparison
    ax = axes[1, 2]
    aq_col = "air_quality_PM2.5"
    if aq_col in ba.columns and aq_col in ct.columns:
        bp_data = [cities[n][aq_col].dropna().values for n in ["Buenos Aires", "Canberra"]]
        if all(len(d) > 0 for d in bp_data):
            bp = ax.boxplot(bp_data, labels=["Buenos Aires", "Canberra"], patch_artist=True)
            bp["boxes"][0].set_facecolor(colors["Buenos Aires"])
            bp["boxes"][1].set_facecolor(colors["Canberra"])
            for b in bp["boxes"]:
                b.set_alpha(0.7)
    ax.set_title("Air Quality PM2.5", fontweight="bold")
    ax.set_ylabel("µg/m³")

    fig.tight_layout()
    _save(fig, "ba_vs_cape_town_comparison")


def run(df: pd.DataFrame):
    #Execute temporal/regional analysis
    print("\n" + "=" * 60)
    print("  Phase 4: REGIONAL & TEMPORAL ANALYSIS")
    print("=" * 60)

    print("\n  ▸ Global temperature trend...")
    plot_global_warming_trend(df)

    print("\n  ▸ Buenos Aires rain by season...")
    plot_ba_rain_by_season(df)

    print("\n  ▸ Buenos Aires rain vs wind direction...")
    plot_ba_rain_vs_wind_direction(df)

    print("\n  ▸ Buenos Aires vs Canberra comparison...")
    plot_ba_vs_Canberra(df)

    print("\n   Temporal/regional analysis complete.")
