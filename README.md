# Term Project 2 — Volatility-Volume-based Order Management
## Columbia University | IEOR4703 Monte Carlo Simulation Methods

---

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Data setup
Place the data folders in a `data/` directory in the project root:
data/
├── Gold/
├── EuroStoxx/
├── GBP - British Pound/
├── German Bunds - German Government Bonds/
├── HeatingOil/
├── JPY - Japanese Yen/
└── Nasdaq/

### 3. Run the statistical engine notebooks
Open and run all cells for each market:
```bash
jupyter notebook cleaning_gold.ipynb
jupyter notebook cleaning_eurostoxx.ipynb
jupyter notebook cleaning_gbp.ipynb
jupyter notebook cleaning_bunds.ipynb
jupyter notebook cleaning_heatingoil.ipynb
jupyter notebook cleaning_jpy.ipynb
jupyter notebook cleaning_nasdaq.ipynb
```

### 4. Run the backtests
```bash
python backtest_gold.py
python run_all_backtests.py
```

### 5. Launch the dashboard
```bash
streamlit run dashboard.py
```
Open http://localhost:8501 in your browser.

### 6. View the final notebook
```bash
jupyter notebook TermProject2_Final.ipynb
```

---

## Notes
- Data files are not included due to size — place them in `data/` as shown above
- Run notebooks before backtests, run backtests before the dashboard
- All backtest outputs are saved to `backtest_outputs/` automatically