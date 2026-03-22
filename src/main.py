# DONT WORRY IF THE IMPORTS ARE NOT FOUND, THEY EXECUTE INSIDE THE VENV
import time
import sys
import os

# Ensure src/ is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, run as run_data_loader
from univariate import run as run_univariate
from bivariate import run as run_bivariate
from temporal import run as run_temporal
from geography import run as run_geography
from ensemble_model import run as run_model
from forecast import run as run_forecast
from pdf_report import run as run_pdf_report


def banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🌦️  WEATHER FORECASTING MODEL                            ║
║     Global Weather Repository Analysis & Prediction          ║
║                                                              ║
║     Primary City: Buenos Aires                               ║
║     Comparison:   Cape Town · Canberra · Wellington          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    t_start = time.time()
    banner()

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "GlobalWeatherRepository.csv")
    csv_path = os.path.abspath(csv_path)

    if not os.path.exists(csv_path):
        print(f"❌ ERROR: CSV not found at {csv_path}")
        print("   Run: kaggle datasets download -d hummaamqaasim/world-weather-repository -p data/ --unzip")
        sys.exit(1)

    # Phase 1: Load & Explore
    print(f"⏱️  Loading dataset from {csv_path}")
    df = load_data(csv_path)
    run_data_loader(df)

    # Phase 2: Univariate
    run_univariate(df)

    # Phase 3: Bivariate
    run_bivariate(df)

    # Phase 4: Temporal/Regional
    run_temporal(df)

    # Phase 5: Geography
    run_geography(df)

    # Phase 6: Ensemble Model
    run_model(df)

    # Phase 6.5: 7-Day Forecast (uses saved models, no retraining)
    run_forecast(df)

    # Phase 7: PDF Report
    run_pdf_report()

    elapsed = time.time() - t_start
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ PIPELINE COMPLETE                                        ║
║  Total time: {elapsed:.1f}s                                    ║
║                                                              ║
║  📁 Outputs saved to:                                        ║
║     outputs/figures/  — All generated plots                  ║
║     outputs/reports/  — Data & model reports                 ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
