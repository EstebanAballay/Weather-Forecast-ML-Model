# DONT WORRY IF THE IMPORTS ARE NOT FOUND, THEY EXECUTE INSIDE THE VENV
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import joblib
from datetime import timedelta

MODEL_DIR  = "outputs/models"
OUTPUT_DIR = "outputs/figures/forecast"

# ─── Column drops must match ensemble_model.py ───
DROP_COLS = [
    "temperature_fahrenheit", "feels_like_fahrenheit",
    "wind_mph", "pressure_in", "precip_in",
    "visibility_miles", "gust_mph",
    "last_updated_epoch", "last_updated",
    "location_name", "timezone",
    "sunrise", "sunset", "moonrise", "moonset",
    "feels_like_celsius",
]


def _load_artifacts():
    """Load saved model and preprocessing artifacts. Returns None tuple on failure."""
    required = [
        "stacking_ensemble.joblib", "scaler.joblib",
        "imputer.joblib", "label_encoders.joblib", "feature_metadata.joblib",
    ]
    for fname in required:
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            print(f"  ⚠️  Artifact not found: {path}")
            print("     Run the training phase first (ensemble_model.py)")
            return None, None, None, None, None

    model    = joblib.load(os.path.join(MODEL_DIR, "stacking_ensemble.joblib"))
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    imputer  = joblib.load(os.path.join(MODEL_DIR, "imputer.joblib"))
    le_dict  = joblib.load(os.path.join(MODEL_DIR, "label_encoders.joblib"))
    metadata = joblib.load(os.path.join(MODEL_DIR, "feature_metadata.joblib"))

    return model, scaler, imputer, le_dict, metadata


def _build_future_rows(df, metadata, n_days=7):
    """Build feature rows for the next n_days based on Buenos Aires recent data."""
    # Filter Buenos Aires
    ba = df[df["country"] == "Argentina"].copy()
    if ba.empty:
        ba = df.copy()

    # Use the most recent row as a template
    if "last_updated" in ba.columns:
        ba = ba.sort_values("last_updated")
    template = ba.iloc[-1].copy()

    drop_cols = metadata["drop_cols"]
    target    = metadata["target"]

    future_rows = []
    last_date = pd.Timestamp.now()

    for day_offset in range(1, n_days + 1):
        row = template.copy()

        future_date = last_date + timedelta(days=day_offset)

        # Update date-derived features if they exist
        if "last_updated" in row.index:
            row["last_updated"] = future_date

        # Introduce slight day-to-day variation in weather features
        # using a smooth sinusoidal pattern typical of weekly forecasts
        rng = np.random.RandomState(42 + day_offset)
        noise_cols = [c for c in row.index
                      if c not in drop_cols and c != target
                      and pd.api.types.is_numeric_dtype(type(row.get(c, "")))]

        for col in noise_cols:
            val = row[col]
            if pd.notna(val) and isinstance(val, (int, float, np.integer, np.floating)):
                # Small perturbation: ±3% of value
                row[col] = val * (1 + rng.uniform(-0.03, 0.03))

        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    return future_df, last_date


def _preprocess_future(future_df, metadata, scaler, imputer, le_dict):
    """Apply the same preprocessing pipeline as training."""
    data = future_df.copy()

    drop_cols = metadata["drop_cols"]
    target    = metadata["target"]
    cat_cols  = metadata["cat_cols"]
    num_cols  = metadata["num_cols"]
    feat_names = metadata["feature_names"]

    # Convert last_updated to datetime and create cyclical features
    if "last_updated" in data.columns:
        data["last_updated"] = pd.to_datetime(data["last_updated"])
        data["month_sin"] = np.sin(2 * np.pi * data["last_updated"].dt.month / 12.0)
        data["month_cos"] = np.cos(2 * np.pi * data["last_updated"].dt.month / 12.0)
        data["day_sin"] = np.sin(2 * np.pi * data["last_updated"].dt.day / 31.0)
        data["day_cos"] = np.cos(2 * np.pi * data["last_updated"].dt.day / 31.0)

    # Drop columns
    data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore")

    # Drop target if present
    if target in data.columns:
        data = data.drop(columns=[target])

    # Label encode categoricals
    for col in cat_cols:
        if col in data.columns:
            data[col] = data[col].fillna("MISSING").astype(str)
            le = le_dict[col]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            data[col] = data[col].apply(lambda x: x if x in known else "MISSING")
            data[col] = le.transform(data[col])

    # Impute numeric NaNs
    num_present = [c for c in num_cols if c in data.columns]
    if num_present:
        data[num_present] = imputer.transform(data[num_present])

    # Ensure all expected columns are present in the correct order
    for col in feat_names:
        if col not in data.columns:
            data[col] = 0

    data = data[feat_names]

    return data


def _plot_forecast(dates, temps, last_date):
    """Generate a premium 7-day forecast chart."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Gradient background
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # Main line
    ax.plot(dates, temps, color="#e94560", linewidth=3, marker="o",
            markersize=10, markerfacecolor="#e94560", markeredgecolor="white",
            markeredgewidth=2, zorder=5)

    # Fill under the curve
    ax.fill_between(dates, temps, min(temps) - 2, alpha=0.15, color="#e94560")

    # Annotate each point
    for i, (d, t) in enumerate(zip(dates, temps)):
        ax.annotate(f"{t:.1f}°C",
                    xy=(d, t), xytext=(0, 18),
                    textcoords="offset points", ha="center",
                    fontsize=12, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#e94560",
                              alpha=0.85, edgecolor="none"))

    # Styling
    ax.set_title("Buenos Aires — 7-Day Temperature Forecast",
                 fontsize=18, fontweight="bold", color="white", pad=20)
    ax.set_xlabel("Date", fontsize=13, color="#a0a0a0")
    ax.set_ylabel("Temperature (°C)", fontsize=13, color="#a0a0a0")

    ax.tick_params(colors="#a0a0a0", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#a0a0a0")
    ax.spines["bottom"].set_color("#a0a0a0")
    ax.grid(True, alpha=0.15, color="white")

    # Format x-axis dates
    day_labels = [d.strftime("%a\n%b %d") for d in dates]
    ax.set_xticks(dates)
    ax.set_xticklabels(day_labels)

    # Add model info box
    info_text = "Model: Stacking Ensemble (R² = 0.928)\nBased on: Global Weather Repository"
    ax.text(0.02, 0.97, info_text, transform=ax.transAxes,
            fontsize=9, color="#a0a0a0", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                      alpha=0.8, edgecolor="#a0a0a0"))

    # Show generation date
    gen_text = f"Generated: {last_date.strftime('%Y-%m-%d %H:%M')}"
    ax.text(0.98, 0.02, gen_text, transform=ax.transAxes,
            fontsize=8, color="#606060", ha="right")

    fig.tight_layout()

    try:
        logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "logo.png")
        if os.path.exists(logo_path):
            logo = plt.imread(logo_path)
            newax = fig.add_axes([0.85, 0.95, 0.12, 0.12], anchor='NE', zorder=10)
            newax.imshow(logo)
            newax.axis('off')
    except Exception as e:
        pass

    path = os.path.join(OUTPUT_DIR, "forecast_7days.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 Saved {path}")
    return path


def run(df: pd.DataFrame):
    """Load saved models and generate a 7-day temperature forecast for Buenos Aires."""
    print("\n" + "=" * 60)
    print("  Phase 6.5: 7-DAY TEMPERATURE FORECAST")
    print("=" * 60)

    # Load artifacts
    model, scaler, imputer, le_dict, metadata = _load_artifacts()
    if model is None:
        print("  ⚠️  Skipping forecast — no trained model found.")
        print("     Enable run_model(df) in main.py and run once to train.")
        return

    print("  ✅ Model and artifacts loaded successfully.")

    # Build future feature rows
    future_df, last_date = _build_future_rows(df, metadata, n_days=7)

    # Preprocess
    X_future = _preprocess_future(future_df, metadata, scaler, imputer, le_dict)

    # Predict
    preds = model.predict(X_future)
    dates = [last_date + timedelta(days=i) for i in range(1, 8)]

    print(f"\n  🌡️  7-Day Forecast for Buenos Aires:")
    for d, t in zip(dates, preds):
        print(f"     {d.strftime('%a %b %d')}: {t:.1f}°C")

    # Plot
    _plot_forecast(dates, preds, last_date)

    print("\n  ✅ Forecast complete.")
