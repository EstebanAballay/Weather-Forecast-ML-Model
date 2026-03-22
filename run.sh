#!/bin/bash
set -e

echo "============================================="
echo "  Weather Forecasting Model — Setup & Run"
echo "============================================="
echo ""

# ── 1. Create virtual environment if needed ──
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# ── 2. Install dependencies ──
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# ── 3. Download dataset from Kaggle if not present ──
if [ ! -f "data/GlobalWeatherRepository.csv" ]; then
    echo "🌐 Downloading dataset from Kaggle..."
    echo "   (Requires ~/.kaggle/kaggle.json — see README.md)"
    kaggle datasets download -d hummaamqaasim/world-weather-repository -p data/ --unzip
fi

# ── 4. Run the analysis pipeline ──
echo ""
echo "🚀 Starting analysis pipeline..."
echo ""
python src/main.py

echo ""
echo "============================================="
echo "  ✅ Done! Check outputs/ for results."
echo "============================================="
