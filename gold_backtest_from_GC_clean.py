"""
Gold backtest script designed to work with the existing cleaning/recommendation workflow
WITHOUT changing the cleaning file from your partner.

Expected inputs in the same folder (unless paths are changed below):
- GC_signals_with_recommendations.csv
- GC_clean.csv

What this script does:
1. loads the recommendation file and stitched 1-minute market file
2. resamples GC_clean.csv to 5-minute OHLCV bars internally
3. runs sanity checks
4. merges recommendations with realized 5-minute market OHLCV
5. applies fill logic
6. computes execution price and improvement vs market-open benchmark
7. writes detailed and summary outputs

How to use:
- Put this file in the same folder as:
    GC_clean.csv
    GC_signals_with_recommendations.csv
- Run it after the cleaning/statistical notebook has already produced those files.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data" / "Gold"

SIGNALS_FILE = DATA_DIR / "GC_signals_with_recommendations.csv"
MARKET_FILE = DATA_DIR / "GC_clean.csv"

OUTPUT_DIR = BASE_DIR / "backtest_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_SIZE = 0.10  # Gold tick size
RESAMPLE_FREQ = "5min"
FILL_PROB_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ----------------------------
# Helpers
# ----------------------------
def _find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """Return the first matching column from candidates, case-insensitive."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def resample_market_to_5min(market: pd.DataFrame) -> pd.DataFrame:
    """
    Convert stitched 1-minute GC_clean data into 5-minute OHLCV bars.
    This keeps the cleaning file untouched and builds the 5-minute bars here.
    """
    dt_col = _find_col(market, ["datetime", "date_time", "timestamp"])
    open_col = _find_col(market, ["open"])
    high_col = _find_col(market, ["high"])
    low_col = _find_col(market, ["low"])
    close_col = _find_col(market, ["close"])
    vol_col = _find_col(market, ["volume"], required=False)

    market = market.copy()
    market[dt_col] = pd.to_datetime(market[dt_col])
    market = market.sort_values(dt_col).set_index(dt_col)

    agg_map = {
        open_col: "first",
        high_col: "max",
        low_col: "min",
        close_col: "last",
    }
    if vol_col is not None:
        agg_map[vol_col] = "sum"

    df_5min = market.resample(RESAMPLE_FREQ).agg(agg_map).dropna().reset_index()

    rename_map = {open_col: "open", high_col: "high", low_col: "low", close_col: "close"}
    if vol_col is not None:
        rename_map[vol_col] = "volume"
    if dt_col != "datetime":
        rename_map[dt_col] = "datetime"

    df_5min = df_5min.rename(columns=rename_map)
    return df_5min


def load_inputs(signals_file: Path, market_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load recommendation file and stitched 1-minute market file, then resample market to 5-minute."""
    if not signals_file.exists():
        raise FileNotFoundError(f"Signals file not found: {signals_file.resolve()}")
    if not market_file.exists():
        raise FileNotFoundError(f"Market file not found: {market_file.resolve()}")

    signals = pd.read_csv(signals_file)
    market_1m = pd.read_csv(market_file)

    sig_dt = _find_col(signals, ["datetime", "date_time", "timestamp"])
    signals[sig_dt] = pd.to_datetime(signals[sig_dt])
    if sig_dt != "datetime":
        signals = signals.rename(columns={sig_dt: "datetime"})

    market_5m = resample_market_to_5min(market_1m)
    return signals, market_5m


def sanity_checks(signals: pd.DataFrame, tick_size: float) -> dict[str, pd.DataFrame]:
    """Run basic checks on the recommendation file."""
    direction_col = _find_col(signals, ["signal_direction", "direction", "side"])
    open_col = _find_col(signals, ["open"])
    limit_col = _find_col(signals, ["limit_price", "recommended_limit_price"])
    ticks_col = _find_col(signals, ["ticks_away"])
    fill_prob_col = _find_col(signals, ["fill_probability", "est_fill_probability"])

    signals = signals.copy()
    signals[direction_col] = signals[direction_col].astype(str).str.strip().str.lower()

    cols_to_check = ["datetime", direction_col, open_col, limit_col, ticks_col, fill_prob_col]
    missing_summary = signals[cols_to_check].isnull().sum().to_frame("missing_count")

    bad_buys = signals[
        (signals[direction_col] == "buy") & (signals[limit_col] > signals[open_col])
    ].copy()
    bad_sells = signals[
        (signals[direction_col] == "sell") & (signals[limit_col] < signals[open_col])
    ].copy()

    signals["tick_diff_check"] = ((signals[limit_col] - signals[open_col]).abs() / tick_size).round()
    bad_tick_rows = signals[signals["tick_diff_check"] != signals[ticks_col]].copy()

    return {
        "missing_summary": missing_summary,
        "bad_buys": bad_buys,
        "bad_sells": bad_sells,
        "bad_tick_rows": bad_tick_rows,
    }


def merge_signals_with_market(signals: pd.DataFrame, market_5m: pd.DataFrame) -> pd.DataFrame:
    """Merge recommendation rows with realized 5-minute market bars."""
    market_use = market_5m[["datetime", "open", "high", "low", "close"]].copy()
    if "volume" in market_5m.columns:
        market_use["volume"] = market_5m["volume"]

    market_use = market_use.rename(columns={"open": "mkt_open"})
    merged = signals.merge(market_use, on="datetime", how="left")
    return merged


def run_backtest(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Backtest logic:
    - buy fills if low <= limit_price
    - sell fills if high >= limit_price
    - if filled, execution price = limit_price
    - if not filled, execution price = close
    - benchmark = signal-time open from recommendation file
    """
    direction_col = _find_col(merged, ["signal_direction", "direction", "side"])
    open_sig_col = _find_col(merged, ["open"])
    limit_col = _find_col(merged, ["limit_price", "recommended_limit_price"])

    merged = merged.copy()
    merged[direction_col] = merged[direction_col].astype(str).str.strip().str.lower()

    buy_mask = merged[direction_col] == "buy"
    sell_mask = merged[direction_col] == "sell"

    merged["filled"] = 0
    merged.loc[buy_mask & (merged["low"] <= merged[limit_col]), "filled"] = 1
    merged.loc[sell_mask & (merged["high"] >= merged[limit_col]), "filled"] = 1

    merged["execution_price"] = np.where(
        merged["filled"] == 1,
        merged[limit_col],
        merged["close"],
    )

    merged["benchmark_price"] = merged[open_sig_col]

    merged["improvement"] = np.nan
    merged.loc[buy_mask, "improvement"] = (
        merged.loc[buy_mask, "benchmark_price"] - merged.loc[buy_mask, "execution_price"]
    )
    merged.loc[sell_mask, "improvement"] = (
        merged.loc[sell_mask, "execution_price"] - merged.loc[sell_mask, "benchmark_price"]
    )

    merged["better_than_market"] = (merged["improvement"] > 0).astype(int)
    return merged


def build_summaries(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build overall and grouped summary tables."""
    direction_col = _find_col(merged, ["signal_direction", "direction", "side"])
    ticks_col = _find_col(merged, ["ticks_away"])
    fill_prob_col = _find_col(merged, ["fill_probability", "est_fill_probability"])

    overall = pd.DataFrame({
        "metric": [
            "num_signals",
            "fill_rate",
            "avg_improvement",
            "median_improvement",
            "positive_improvement_rate",
        ],
        "value": [
            len(merged),
            merged["filled"].mean(),
            merged["improvement"].mean(),
            merged["improvement"].median(),
            merged["better_than_market"].mean(),
        ],
    })

    by_direction = merged.groupby(direction_col).agg(
        num_signals=("datetime", "count"),
        fill_rate=("filled", "mean"),
        avg_improvement=("improvement", "mean"),
        median_improvement=("improvement", "median"),
        positive_improvement_rate=("better_than_market", "mean"),
    ).reset_index()

    by_ticks = merged.groupby(ticks_col).agg(
        num_signals=("datetime", "count"),
        fill_rate=("filled", "mean"),
        avg_improvement=("improvement", "mean"),
        median_improvement=("improvement", "median"),
    ).reset_index()

    state_tables = {}
    for state_col in ["vol_state", "range_state", "trend_state"]:
        if state_col in merged.columns:
            state_tables[state_col] = merged.groupby(state_col).agg(
                num_signals=("datetime", "count"),
                fill_rate=("filled", "mean"),
                avg_improvement=("improvement", "mean"),
                median_improvement=("improvement", "median"),
            ).reset_index()

    merged = merged.copy()
    merged["fill_prob_bucket"] = pd.cut(
        merged[fill_prob_col],
        bins=FILL_PROB_BINS,
        include_lowest=True,
    )

    calibration = merged.groupby("fill_prob_bucket", observed=False).agg(
        num_signals=("datetime", "count"),
        avg_est_fill_probability=(fill_prob_col, "mean"),
        actual_fill_rate=("filled", "mean"),
    ).reset_index()

    return {
        "overall": overall,
        "by_direction": by_direction,
        "by_ticks": by_ticks,
        "calibration": calibration,
        **{f"by_{k}": v for k, v in state_tables.items()},
    }


def save_tables(checks: dict[str, pd.DataFrame], market_5m: pd.DataFrame, merged: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> None:
    """Write detailed outputs and summaries."""
    checks["missing_summary"].to_csv(OUTPUT_DIR / "GC_recommendation_missing_summary.csv")
    checks["bad_buys"].to_csv(OUTPUT_DIR / "GC_bad_buys.csv", index=False)
    checks["bad_sells"].to_csv(OUTPUT_DIR / "GC_bad_sells.csv", index=False)
    checks["bad_tick_rows"].to_csv(OUTPUT_DIR / "GC_bad_tick_rows.csv", index=False)

    market_5m.to_csv(OUTPUT_DIR / "GC_5min_from_GC_clean.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "GC_backtest_results.csv", index=False)

    for name, df in summaries.items():
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def make_charts(summaries: dict[str, pd.DataFrame]) -> None:
    """Create a few simple charts."""
    if "by_direction" in summaries and not summaries["by_direction"].empty:
        df = summaries["by_direction"]
        direction_col = df.columns[0]
        plt.figure(figsize=(6, 4))
        plt.bar(df[direction_col].astype(str), df["avg_improvement"])
        plt.title("Average Improvement by Direction")
        plt.xlabel("Signal Direction")
        plt.ylabel("Average Improvement")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "GC_avg_improvement_by_direction.png", dpi=200)
        plt.close()

    if "calibration" in summaries and not summaries["calibration"].empty:
        df = summaries["calibration"].dropna(subset=["avg_est_fill_probability", "actual_fill_rate"])
        if not df.empty:
            plt.figure(figsize=(7, 4))
            plt.plot(df["avg_est_fill_probability"], df["actual_fill_rate"], marker="o")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.title("Estimated vs Actual Fill Probability")
            plt.xlabel("Estimated Fill Probability")
            plt.ylabel("Actual Fill Rate")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "GC_fill_probability_calibration.png", dpi=200)
            plt.close()

    if "by_ticks" in summaries and not summaries["by_ticks"].empty:
        df = summaries["by_ticks"]
        ticks_col = _find_col(df, ["ticks_away"])
        plt.figure(figsize=(7, 4))
        plt.bar(df[ticks_col].astype(str), df["fill_rate"])
        plt.title("Fill Rate by Ticks Away")
        plt.xlabel("Ticks Away")
        plt.ylabel("Fill Rate")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "GC_fill_rate_by_ticks.png", dpi=200)
        plt.close()


def write_run_report(checks: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame]) -> None:
    """Write a short text summary of the run."""
    lines = []
    lines.append("Gold Backtest Run Report")
    lines.append("=" * 26)
    lines.append("")
    lines.append("Recommendation sanity checks")
    lines.append("- Missing critical values: see GC_recommendation_missing_summary.csv")
    lines.append(f"- Bad buy rows: {len(checks['bad_buys'])}")
    lines.append(f"- Bad sell rows: {len(checks['bad_sells'])}")
    lines.append(f"- Bad tick rows: {len(checks['bad_tick_rows'])}")
    lines.append("")

    if "overall" in summaries:
        overall = summaries["overall"]
        metric_map = dict(zip(overall["metric"], overall["value"]))
        lines.append("Overall summary")
        lines.append(f"- Number of signals: {metric_map.get('num_signals')}")
        lines.append(f"- Fill rate: {metric_map.get('fill_rate'):.6f}")
        lines.append(f"- Average improvement: {metric_map.get('avg_improvement'):.6f}")
        lines.append(f"- Median improvement: {metric_map.get('median_improvement'):.6f}")
        lines.append(f"- Positive improvement rate: {metric_map.get('positive_improvement_rate'):.6f}")

    (OUTPUT_DIR / "GC_backtest_run_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("Loading inputs...")
    signals, market_5m = load_inputs(SIGNALS_FILE, MARKET_FILE)

    print("Running sanity checks...")
    checks = sanity_checks(signals, TICK_SIZE)

    print("Merging signals with realized 5-minute market bars...")
    merged = merge_signals_with_market(signals, market_5m)

    print("Running backtest...")
    merged = run_backtest(merged)

    print("Building summaries...")
    summaries = build_summaries(merged)

    print("Saving tables...")
    save_tables(checks, market_5m, merged, summaries)

    print("Making charts...")
    make_charts(summaries)

    print("Writing run report...")
    write_run_report(checks, summaries)

    print(f"Done. Outputs written to: {OUTPUT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
