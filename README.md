# Weather Forecasting Model Pipeline

This is an end-to-end Machine Learning pipeline that automates the extraction, exploration, training, and reporting of global weather data. It trains a Stacking Ensemble model to predict temperatures and generates a comprehensive 7-day forecast.

## System Requirements
- Python 3.9+
- Kaggle API credentials (for downloading the dataset)

##  How to Run the Project

Because machine learning models and virtual environments are very heavy, they are not included in this repository. Follow these steps to reproduce the entire pipeline locally from scratch:

### 1- Clone the repository
```bash
git clone https://github.com/your-username/ForecastingModel.git
cd ForecastingModel
```

### 2- Easy Setup & Run 
The easiest way to test this project is by using the automated `run.sh` script. This script acts as a "Plug & Play" wrapper that handles everything for you: it creates a virtual environment, installs the dependencies from `requirements.txt`, downloads the dataset, and runs the entire pipeline automatically.

```bash
chmod +x run.sh
./run.sh
```

**⚠️ Important Requirement:** The script automatically downloads the dataset using the Kaggle API. You must have your `kaggle.json` credentials configured (`~/.kaggle/kaggle.json` on Linux/Mac, or `C:\Users\Username\.kaggle\kaggle.json` on Windows). If you already downloaded the dataset manually to `data/GlobalWeatherRepository.csv`, the script will simply skip the download step.

### 3- Manual Execution
If you prefer to run the project manually step-by-step instead of using the automation script, you need to prepare the environment first:
1. Create and activate a virtual environment (`python3 -m venv .venv` and `source .venv/bin/activate`).
2. Install dependencies (`pip install -r requirements.txt`).
3. Ensure the dataset is located at `data/GlobalWeatherRepository.csv`.
4. Execute the orchestrator script directly:

```bash
python src/main.py
```
*Note: The `src/main.py` script contains the core Python logic and assumes your environment and data are already fully prepared.*If you don't want to wait until the model is trained each time you want to run the project,you can simply comment the run_model(df) line of phase 6.


