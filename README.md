# Weather Forecasting Model Pipeline

This is an end-to-end Machine Learning pipeline that automates the extraction, exploration, training, and reporting of global weather data. It trains a Stacking Ensemble model to predict temperatures and generates a comprehensive 7-day forecast.

## System Requirements
- Python 3.9+
- Kaggle API credentials (for downloading the dataset)

##  How to Run the Project (For Recruiters/Reviewers)

Because machine learning models and virtual environments are very heavy, they are not included in this repository. Follow these steps to reproduce the entire pipeline locally from scratch:

### 1- Clone the repository
```bash
git clone https://github.com/your-username/ForecastingModel.git
cd ForecastingModel
```

### 2- Set up the Environment
It is highly recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate it
# On Linux/MacOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install all required libraries
pip install -r requirements.txt
```

### 3- Kaggle API Setup (Dataset Download)
The pipeline automatically downloads the `GlobalWeatherRepository.csv` dataset using the Kaggle API.
To allow this, you must have your `kaggle.json` credentials file placed in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\Username\.kaggle\kaggle.json` (Windows). 
*If you already have the `data/GlobalWeatherRepository.csv` file manually downloaded, you can skip this step.*

### 4- Execute the Pipeline
Run the main orchestrator script. This will automatically execute data loading, exploratory data analysis (EDA), model training, 7-day forecasting, and PDF report generation.

```bash
python src/main.py
```

## Project Structure
- `src/`: Python source code modules.
- `data/`: Raw downloaded CSV data.
- `outputs/`: Auto-generated upon execution.
  - `figures/`: PNG charts from EDA and forecasting.
  - `models/`: Serialized trained models (`.joblib`).
  - `reports/`: The final generated PDF (`weather_report.pdf`).
