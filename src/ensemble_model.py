# DONT WORRY IF THE IMPORTS ARE NOT FOUND, THEY EXECUTE INSIDE THE VENV
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
import joblib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


OUTPUT_DIR_FIG = "outputs/figures/model"
OUTPUT_DIR_RPT = "outputs/reports"
OUTPUT_DIR_MDL = "outputs/models"
TARGET = "temperature_celsius"

# Columns to drop (redundant unit conversions, IDs, or target leakage)
DROP_COLS = [
    "temperature_fahrenheit", "feels_like_fahrenheit",
    "wind_mph", "pressure_in", "precip_in",
    "visibility_miles", "gust_mph",
    "last_updated_epoch", "last_updated",
    "location_name", "timezone",
    "sunrise", "sunset", "moonrise", "moonset",
    # feels_like is derived from temperature — target leakage
    "feels_like_celsius",
]


def _save_fig(fig, name):
    os.makedirs(OUTPUT_DIR_FIG, exist_ok=True)
    path = os.path.join(OUTPUT_DIR_FIG, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  📊 Saved {path}")


def preprocess(df: pd.DataFrame):
    """Prepare data for modeling."""
    print("\n  ▸ Preprocessing data...")

    data = df.copy()

    # Drop columns
    data = data.drop(columns=[c for c in DROP_COLS if c in data.columns], errors="ignore")

    # Drop rows without target
    data = data.dropna(subset=[TARGET])

    # Convert last_updated to datetime and create cyclical features
    if "last_updated" in data.columns:
        data["last_updated"] = pd.to_datetime(data["last_updated"])
        data["month_sin"] = np.sin(2 * np.pi * data["last_updated"].dt.month / 12.0)
        data["month_cos"] = np.cos(2 * np.pi * data["last_updated"].dt.month / 12.0)
        data["day_sin"] = np.sin(2 * np.pi * data["last_updated"].dt.day / 31.0)
        data["day_cos"] = np.cos(2 * np.pi * data["last_updated"].dt.day / 31.0)

    # Separate target
    y = data[TARGET]
    X = data.drop(columns=[TARGET])

    # Identify column types
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"     Features: {len(num_cols)} numeric, {len(cat_cols)} categorical")
    print(f"     Samples:  {len(X):,}")

    # Label encode categoricals
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna("MISSING")
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    # Impute remaining NaNs in numeric columns
    imputer = SimpleImputer(strategy="median")
    X[num_cols] = imputer.fit_transform(X[num_cols])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (for SVR and Ridge particularly)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print(f"     Train: {len(X_train):,} | Test: {len(X_test):,}")

    return (X_train, X_test, X_train_scaled, X_test_scaled,
            y_train, y_test, X.columns.tolist(),
            scaler, imputer, le_dict, cat_cols, num_cols)


def build_models():
    """Define all individual models."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0, n_jobs=-1
        )
    else:
        print("  ⚠️  XGBoost not available, skipping")

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1, n_jobs=-1
        )
    else:
        print("  ⚠️  LightGBM not available, skipping")

    # SVR needs scaled data — we'll handle this separately
    models["SVR"] = SVR(kernel="rbf", C=10.0, epsilon=0.1)

    return models


def train_and_evaluate(models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    """Train all models and collect metrics."""
    results = {}
    predictions = {}

    # Models that need scaling
    scale_models = {"SVR", "Linear Regression", "Ridge Regression"}

    for name, model in models.items():
        print(f"\n  ▸ Training {name}...")
        t0 = time.time()

        x_tr = X_train_scaled if name in scale_models else X_train
        x_te = X_test_scaled if name in scale_models else X_test

        # For SVR with large datasets, subsample for speed
        if name == "SVR" and len(x_tr) > 20000:
            sample_idx = x_tr.sample(20000, random_state=42).index
            model.fit(x_tr.loc[sample_idx], y_train.loc[sample_idx])
        else:
            model.fit(x_tr, y_train)

        elapsed = time.time() - t0
        y_pred = model.predict(x_te)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2, "Time (s)": round(elapsed, 2)}
        predictions[name] = y_pred

        print(f"     MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  ({elapsed:.1f}s)")

    return results, predictions


def build_ensemble(models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test,
                   results, predictions):
    """Build Stacking and Voting ensemble methods."""
    scale_models = {"SVR", "Linear Regression", "Ridge Regression"}

    # ── Stacking Regressor ──
    print("\n  ▸ Building Stacking Ensemble...")
    t0 = time.time()

    # Use tree-based models (no scaling needed) as base estimators for stacking
    base_estimators = []
    for name, model in models.items():
        if name not in scale_models:
            base_estimators.append((name.lower().replace(" ", "_"), model.__class__(
                **{k: v for k, v in model.get_params().items()
                   if k not in ["verbose", "verbosity"]}
            )))

    if len(base_estimators) >= 2:
        stacking = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=-1,
        )
        stacking.fit(X_train, y_train)
        y_pred_stack = stacking.predict(X_test)
        elapsed = time.time() - t0

        mae = mean_absolute_error(y_test, y_pred_stack)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_stack))
        r2 = r2_score(y_test, y_pred_stack)

        results["⭐ Stacking Ensemble"] = {"MAE": mae, "RMSE": rmse, "R²": r2, "Time (s)": round(elapsed, 2)}
        predictions["⭐ Stacking Ensemble"] = y_pred_stack
        print(f"     MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  ({elapsed:.1f}s)")

    # ── Voting Regressor ──
    print("\n  ▸ Building Voting Ensemble...")
    t0 = time.time()

    # Use top performing tree-based models
    top_models = sorted(
        [(n, results[n]["R²"]) for n in results if n not in ["⭐ Stacking Ensemble"] and n not in scale_models],
        key=lambda x: x[1], reverse=True
    )[:4]

    if len(top_models) >= 2:
        voting_estimators = []
        for name, _ in top_models:
            m = models[name]
            voting_estimators.append((name.lower().replace(" ", "_"), m.__class__(
                **{k: v for k, v in m.get_params().items()
                   if k not in ["verbose", "verbosity"]}
            )))

        voting = VotingRegressor(estimators=voting_estimators, n_jobs=-1)
        voting.fit(X_train, y_train)
        y_pred_vote = voting.predict(X_test)
        elapsed = time.time() - t0

        mae = mean_absolute_error(y_test, y_pred_vote)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_vote))
        r2 = r2_score(y_test, y_pred_vote)

        results["⭐ Voting Ensemble"] = {"MAE": mae, "RMSE": rmse, "R²": r2, "Time (s)": round(elapsed, 2)}
        predictions["⭐ Voting Ensemble"] = y_pred_vote
        print(f"     MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  ({elapsed:.1f}s)")

    return results, predictions


def plot_model_comparison(results):
    """Bar chart comparing all model metrics."""
    df_results = pd.DataFrame(results).T.sort_values("R²", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle("Model Performance Comparison", fontsize=18, fontweight="bold", y=1.02)

    # Colors: highlight ensemble models
    colors = []
    for name in df_results.index:
        if "⭐" in name:
            colors.append("#2ECC71")
        else:
            colors.append("#3498DB")

    # R²
    axes[0].barh(df_results.index, df_results["R²"], color=colors, edgecolor="white")
    axes[0].set_xlabel("R² Score", fontsize=12)
    axes[0].set_title("R² (higher is better)", fontweight="bold")
    for i, v in enumerate(df_results["R²"]):
        axes[0].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

    # MAE
    df_sorted_mae = df_results.sort_values("MAE", ascending=False)
    colors_mae = ["#2ECC71" if "⭐" in n else "#E74C3C" for n in df_sorted_mae.index]
    axes[1].barh(df_sorted_mae.index, df_sorted_mae["MAE"], color=colors_mae, edgecolor="white")
    axes[1].set_xlabel("MAE", fontsize=12)
    axes[1].set_title("MAE (lower is better)", fontweight="bold")
    for i, v in enumerate(df_sorted_mae["MAE"]):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    # RMSE
    df_sorted_rmse = df_results.sort_values("RMSE", ascending=False)
    colors_rmse = ["#2ECC71" if "⭐" in n else "#F39C12" for n in df_sorted_rmse.index]
    axes[2].barh(df_sorted_rmse.index, df_sorted_rmse["RMSE"], color=colors_rmse, edgecolor="white")
    axes[2].set_xlabel("RMSE", fontsize=12)
    axes[2].set_title("RMSE (lower is better)", fontweight="bold")
    for i, v in enumerate(df_sorted_rmse["RMSE"]):
        axes[2].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    _save_fig(fig, "model_comparison")


def plot_predictions_vs_actual(y_test, predictions):
    """Scatter plot of predicted vs actual values for all models."""
    n_models = len(predictions)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    fig.suptitle("Predictions vs Actual Values", fontsize=18, fontweight="bold", y=1.02)
    axes_flat = axes.flatten() if n_models > 1 else [axes]

    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = axes_flat[i]
        # Sample for performance
        idx = np.random.RandomState(42).choice(len(y_test), min(2000, len(y_test)), replace=False)
        y_t = y_test.values[idx]
        y_p = y_pred[idx]

        color = "#2ECC71" if "⭐" in name else "#3498DB"
        ax.scatter(y_t, y_p, alpha=0.3, s=10, color=color, edgecolors="none")

        # Perfect prediction line
        lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
        ax.plot(lims, lims, "r--", linewidth=2, alpha=0.7)

        r2 = r2_score(y_test, y_pred)
        ax.set_title(f"{name}\nR²={r2:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused
    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, "predictions_vs_actual")


def plot_feature_importance(models, feature_names):
    """Feature importance from tree-based models."""
    # Pick the best tree-based model
    tree_models = {}
    for name in ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"]:
        if name in models and hasattr(models[name], "feature_importances_"):
            tree_models[name] = models[name].feature_importances_

    if not tree_models:
        return

    fig, axes = plt.subplots(1, len(tree_models), figsize=(7 * len(tree_models), 8))
    if len(tree_models) == 1:
        axes = [axes]

    fig.suptitle("Feature Importance — Tree-Based Models", fontsize=16, fontweight="bold", y=1.02)

    for ax, (name, importances) in zip(axes, tree_models.items()):
        indices = np.argsort(importances)[-15:]  # Top 15
        ax.barh(
            [feature_names[i] for i in indices],
            importances[indices],
            color=sns.color_palette("viridis", len(indices)),
            edgecolor="white",
        )
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(name, fontsize=13, fontweight="bold")

    fig.tight_layout()
    _save_fig(fig, "feature_importance")


def plot_residuals(y_test, predictions):
    """Residual plots for the best models."""
    # Pick top 4 models
    top = sorted(predictions.keys(), key=lambda n: ("⭐" not in n, n))[:4]

    fig, axes = plt.subplots(1, len(top), figsize=(6 * len(top), 5))
    if len(top) == 1:
        axes = [axes]
    fig.suptitle("Residual Analysis", fontsize=16, fontweight="bold", y=1.02)

    for ax, name in zip(axes, top):
        residuals = y_test.values - predictions[name]
        color = "#2ECC71" if "⭐" in name else "#3498DB"
        ax.scatter(predictions[name], residuals, alpha=0.2, s=8, color=color, edgecolors="none")
        ax.axhline(y=0, color="red", linewidth=2, linestyle="--")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Residual", fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, "residual_analysis")


def save_report(results):
    """Save model comparison report to text file."""
    os.makedirs(OUTPUT_DIR_RPT, exist_ok=True)
    path = os.path.join(OUTPUT_DIR_RPT, "model_results.txt")

    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append("  ENSEMBLE MODEL — RESULTS REPORT")
    lines.append(sep)

    df_results = pd.DataFrame(results).T.sort_values("R²", ascending=False)
    lines.append(f"\n{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Time':>8}")
    lines.append("-" * 70)
    for name, row in df_results.iterrows():
        lines.append(f"{name:<30} {row['MAE']:>8.3f} {row['RMSE']:>8.3f} {row['R²']:>8.4f} {row['Time (s)']:>7.1f}s")

    best = df_results.index[0]
    lines.append(f"\n🏆 Best model: {best}  (R² = {df_results.loc[best, 'R²']:.4f})")

    report = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n💾 Report saved to {path}")
    print(report)


def save_model_artifacts(stacking_model, scaler, imputer, le_dict,
                         feat_names, cat_cols, num_cols):
    """Persist trained model and preprocessing artifacts to disk."""
    os.makedirs(OUTPUT_DIR_MDL, exist_ok=True)

    joblib.dump(stacking_model, os.path.join(OUTPUT_DIR_MDL, "stacking_ensemble.joblib"))
    joblib.dump(scaler,         os.path.join(OUTPUT_DIR_MDL, "scaler.joblib"))
    joblib.dump(imputer,        os.path.join(OUTPUT_DIR_MDL, "imputer.joblib"))
    joblib.dump(le_dict,        os.path.join(OUTPUT_DIR_MDL, "label_encoders.joblib"))
    joblib.dump({
        "feature_names": feat_names,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "drop_cols": DROP_COLS,
        "target": TARGET,
    }, os.path.join(OUTPUT_DIR_MDL, "feature_metadata.joblib"))

    print(f"\n  💾 Model artifacts saved to {OUTPUT_DIR_MDL}/")


def run(df: pd.DataFrame):
    """Execute the full ensemble modeling pipeline."""
    print("\n" + "=" * 60)
    print("  Phase 6: ENSEMBLE FORECASTING MODEL")
    print("=" * 60)
    print(f"  Target: {TARGET}")

    # Preprocess
    (X_train, X_test, X_train_s, X_test_s,
     y_train, y_test, feat_names,
     scaler, imputer, le_dict, cat_cols, num_cols) = preprocess(df)

    # Build models
    models = build_models()
    print(f"\n  📋 Models to train: {', '.join(models.keys())}")

    # Train and evaluate individual models
    results, predictions = train_and_evaluate(
        models, X_train, X_test, X_train_s, X_test_s, y_train, y_test
    )

    # Build ensembles
    results, predictions = build_ensemble(
        models, X_train, X_test, X_train_s, X_test_s, y_train, y_test,
        results, predictions
    )

    # Visualizations
    print("\n  ▸ Generating model comparison plots...")
    plot_model_comparison(results)
    plot_predictions_vs_actual(y_test, predictions)
    plot_feature_importance(models, feat_names)
    plot_residuals(y_test, predictions)

    # Report
    save_report(results)

    # ── Persist the Stacking Ensemble for inference ──
    # We need a reference to the fitted stacking model.
    # Rebuild it from the ensemble step (it was trained inside build_ensemble).
    # Re-train a fresh one specifically to save (trained on ALL training data).
    print("\n  ▸ Saving best model for inference...")
    scale_models = {"SVR", "Linear Regression", "Ridge Regression"}
    base_estimators = []
    for name, model in models.items():
        if name not in scale_models:
            base_estimators.append((name.lower().replace(" ", "_"), model))

    if len(base_estimators) >= 2:
        stacking = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=-1,
        )
        stacking.fit(X_train, y_train)
        save_model_artifacts(stacking, scaler, imputer, le_dict,
                             feat_names, cat_cols, num_cols)

    print("\n  ✅ Ensemble modeling complete.")
