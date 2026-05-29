import subprocess
import sys
from pathlib import Path

markets = [
    {
        "name": "EuroStoxx",
        "data_dir": "data/EuroStoxx",
        "signals_file": "data/EuroStoxx/ES_signals_with_recommendations.csv",
        "market_file": "data/EuroStoxx/ES_clean.csv",
        "tick_size": 0.50,
        "output_prefix": "ES"
    },
    {
        "name": "GBP",
        "data_dir": "data/GBP - British Pound",
        "signals_file": "data/GBP - British Pound/GBP_signals_with_recommendations.csv",
        "market_file": "data/GBP - British Pound/GBP_clean.csv",
        "tick_size": 0.01,
        "output_prefix": "GBP"
    },
    {
        "name": "German Bunds",
        "data_dir": "data/German Bunds - German Government Bonds",
        "signals_file": "data/German Bunds - German Government Bonds/Bunds_signals_with_recommendations.csv",
        "market_file": "data/German Bunds - German Government Bonds/Bunds_clean.csv",
        "tick_size": 0.01,
        "output_prefix": "Bunds"
    },
    {
        "name": "Heating Oil",
        "data_dir": "data/HeatingOil",
        "signals_file": "data/HeatingOil/HeatingOil_signals_with_recommendations.csv",
        "market_file": "data/HeatingOil/HO_clean.csv",
        "tick_size": 0.01,
        "output_prefix": "HO"
    },
    {
        "name": "JPY",
        "data_dir": "data/JPY - Japanese Yen",
        "signals_file": "data/JPY - Japanese Yen/JPY_signals_with_recommendations.csv",
        "market_file": "data/JPY - Japanese Yen/JPY_clean.csv",
        "tick_size": 0.005,
        "output_prefix": "JPY"
    },
    {
        "name": "Nasdaq",
        "data_dir": "data/Nasdaq",
        "signals_file": "data/Nasdaq/Nasdaq_signals_with_recommendations.csv",
        "market_file": "data/Nasdaq/NQ_clean.csv",
        "tick_size": 0.25,
        "output_prefix": "NQ"
    },
]

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILL_PROB_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def resample_market_to_5min(market, prefix):
    market = market.copy()
    market["datetime"] = pd.to_datetime(market["datetime"])
    market = market.sort_values("datetime").set_index("datetime")
    df_5min = market.resample("5min").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    return df_5min

def run_backtest_for_market(config):
    name = config["name"]
    tick_size = config["tick_size"]
    prefix = config["output_prefix"]
    output_dir = Path("backtest_outputs") / prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*40}")
    print(f"Running backtest for {name}...")

    # Load
    signals = pd.read_csv(config["signals_file"])
    signals["datetime"] = pd.to_datetime(signals["datetime"])
    market_1m = pd.read_csv(config["market_file"])
    market_5m = resample_market_to_5min(market_1m, prefix)

    # Merge
    market_use = market_5m[["datetime", "open", "high", "low", "close"]].copy()
    market_use = market_use.rename(columns={"open": "mkt_open"})
    merged = signals.merge(market_use, on="datetime", how="left")

    # Fill logic
    merged["signal_direction"] = merged["signal_direction"].str.strip().str.lower()
    buy_mask = merged["signal_direction"] == "buy"
    sell_mask = merged["signal_direction"] == "sell"

    merged["filled"] = 0
    merged.loc[buy_mask & (merged["low"] <= merged["limit_price"]), "filled"] = 1
    merged.loc[sell_mask & (merged["high"] >= merged["limit_price"]), "filled"] = 1

    merged["execution_price"] = np.where(
        merged["filled"] == 1, merged["limit_price"], merged["close"]
    )
    merged["benchmark_price"] = merged["open"]

    merged["improvement"] = np.nan
    merged.loc[buy_mask, "improvement"] = (
        merged.loc[buy_mask, "benchmark_price"] - merged.loc[buy_mask, "execution_price"]
    )
    merged.loc[sell_mask, "improvement"] = (
        merged.loc[sell_mask, "execution_price"] - merged.loc[sell_mask, "benchmark_price"]
    )
    merged["better_than_market"] = (merged["improvement"] > 0).astype(int)
    merged["improvement_ticks"] = merged["improvement"] / tick_size
    # Summaries
    overall = pd.DataFrame({
        "metric": ["num_signals", "fill_rate", "avg_improvement", 
                   "median_improvement", "positive_improvement_rate"],
        "value": [len(merged), merged["filled"].mean(),
                  merged["improvement"].mean(), merged["improvement"].median(),
                  merged["better_than_market"].mean()]
    })

    by_direction = merged.groupby("signal_direction").agg(
        num_signals=("datetime", "count"),
        fill_rate=("filled", "mean"),
        avg_improvement=("improvement", "mean"),
        median_improvement=("improvement", "median"),
        positive_improvement_rate=("better_than_market", "mean")
    ).reset_index()

    by_ticks = merged.groupby("ticks_away").agg(
        num_signals=("datetime", "count"),
        fill_rate=("filled", "mean"),
        avg_improvement=("improvement", "mean"),
        median_improvement=("improvement", "median")
    ).reset_index()

    state_tables = {}
    for state_col in ["vol_state", "range_state", "trend_state"]:
        if state_col in merged.columns:
            state_tables[state_col] = merged.groupby(state_col).agg(
                num_signals=("datetime", "count"),
                fill_rate=("filled", "mean"),
                avg_improvement=("improvement", "mean"),
                median_improvement=("improvement", "median")
            ).reset_index()

    # Save
    overall.to_csv(output_dir / "overall.csv", index=False)
    by_direction.to_csv(output_dir / "by_direction.csv", index=False)
    by_ticks.to_csv(output_dir / "by_ticks.csv", index=False)
    merged.to_csv(output_dir / f"{prefix}_backtest_results.csv", index=False)
    for k, v in state_tables.items():
        v.to_csv(output_dir / f"by_{k}.csv", index=False)

    metric_map = dict(zip(overall["metric"], overall["value"]))
    print(f"  Fill rate: {metric_map['fill_rate']:.1%}")
    print(f"  Avg improvement: {metric_map['avg_improvement']:.4f}")
    print(f"  Positive improvement rate: {metric_map['positive_improvement_rate']:.1%}")
    print(f"  Outputs saved to: {output_dir}")

if __name__ == "__main__":
    for config in markets:
        run_backtest_for_market(config)
    print("\nAll backtests complete.")