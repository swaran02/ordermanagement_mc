"""
Heating oil recommendation backtest.

Expected inputs:
- HeatingOil_signals_with_recommendations.csv
- HeatingOil_clean.csv if available
- otherwise the raw heating oil contract files in the expected data folder

What this script does:
1. loads the recommendation file and the stitched clean market file if it exists
2. rebuilds the stitched market file from the raw contracts if needed
3. resamples the market data to the signal horizon
4. applies the fill rules to each recommendation
5. writes detailed backtest results, summaries, and charts

How to use:
- run the heating oil cleaning notebook first so it creates the recommendation file
- if no clean market file was saved, keep the raw contract files in the expected data folder
- run the script and review the files written to backtest_outputs
"""

from __future__ import annotations

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# fallback defaults in case the recommendation file does not carry these yet
DEFAULT_TICK_SIZE = 0.01
DEFAULT_TAU_MIN = 5
DAILY_MINUTES = 1380

# these buckets are for comparing estimated fill probability to realized fill rate
FILL_PROB_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# asset-specific locations and patterns taken from the cleaning notebook
DATA_SUBDIR = Path(r"data/HeatingOil")
SIGNALS_FILENAME = "HeatingOil_signals_with_recommendations.csv"
RAW_GLOB = "HO*.csv"
CLEAN_CANDIDATES = ['HeatingOil_clean.csv', 'heatingoil_clean.csv', 'HO_clean.csv']


def _find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    # lets the script survive small column-name differences without rewriting everything
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def _find_existing_path(candidates: list[Path], label: str) -> Path:
    # tries a few likely file locations and returns the first one that exists
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {label}. Tried these paths:\n" +
        "\n".join(str(p.resolve()) for p in candidates)
    )


def infer_tick_size(signals: pd.DataFrame) -> float:
    # if tick_size exists in the recommendation file, use it
    # otherwise just use the asset default
    col = _find_col(signals, ["tick_size"], required=False)
    if col is None:
        return DEFAULT_TICK_SIZE

    vals = pd.to_numeric(signals[col], errors="coerce").dropna().unique()
    if len(vals) == 0:
        return DEFAULT_TICK_SIZE
    return float(vals[0])


def infer_tau_minutes(signals: pd.DataFrame) -> int:
    # same idea for tau / holding period
    col = _find_col(
        signals,
        ["tau", "holding_period_min", "holding_period_minutes"],
        required=False,
    )
    if col is None:
        return DEFAULT_TAU_MIN

    vals = pd.to_numeric(signals[col], errors="coerce").dropna().unique()
    if len(vals) == 0:
        return DEFAULT_TAU_MIN
    return int(vals[0])


def load_and_clean(filepath: str, daily_minutes: int = DAILY_MINUTES) -> pd.DataFrame:
    # this matches the basic cleaning logic used in the notebooks:
    # read raw file, remove weak days, then keep only valid OHLC rows with positive volume
    df = pd.read_csv(
        filepath,
        header=None,
        names=["datetime", "open", "high", "low", "close", "volume"],
    )

    df["contract"] = Path(filepath).stem
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y.%m.%d.%H:%M:%S")
    df["date"] = df["datetime"].dt.date
    df["bars_per_day"] = df.groupby("date")["datetime"].transform("count")

    threshold = int(daily_minutes * 0.90)
    df_clean = df[df["bars_per_day"] >= threshold].copy()

    valid_ohlc = (
        (df_clean["high"] >= df_clean["low"]) &
        (df_clean["open"] >= df_clean["low"]) &
        (df_clean["open"] <= df_clean["high"]) &
        (df_clean["close"] >= df_clean["low"]) &
        (df_clean["close"] <= df_clean["high"]) &
        (df_clean["volume"] > 0)
    )
    df_clean = df_clean[valid_ohlc].copy()
    return df_clean


def build_stitched_market_from_raw(base: Path) -> pd.DataFrame:
    # if no clean stitched file exists yet, rebuild it straight from the raw contract files
    raw_paths = sorted(glob.glob(str(base / RAW_GLOB)))
    if len(raw_paths) == 0:
        raise FileNotFoundError(
            f"No raw contract files found with pattern {RAW_GLOB} under {base.resolve()}"
        )

    all_dfs = [load_and_clean(p) for p in raw_paths]
    df_all = pd.concat(all_dfs, ignore_index=True)

    # pick the most active contract on each date by total daily volume
    daily_volume = (
        df_all.groupby(["date", "contract"], as_index=False)["volume"]
        .sum()
        .sort_values(["date", "volume", "contract"], ascending=[True, False, True])
    )
    active = daily_volume.drop_duplicates(subset=["date"], keep="first")[["date", "contract"]]

    df_stitched = df_all.merge(active, on=["date", "contract"], how="inner")
    df_stitched = df_stitched.sort_values("datetime").reset_index(drop=True)
    return df_stitched


def resample_market(market: pd.DataFrame, tau_min: int) -> pd.DataFrame:
    # the backtest wants bars at the same horizon as the recommendation file
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

    df_tau = market.resample(f"{tau_min}min").agg(agg_map).dropna().reset_index()

    rename_map = {
        open_col: "open",
        high_col: "high",
        low_col: "low",
        close_col: "close",
    }
    if vol_col is not None:
        rename_map[vol_col] = "volume"
    if dt_col != "datetime":
        rename_map[dt_col] = "datetime"

    return df_tau.rename(columns=rename_map)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    # signals file should come from the cleaning notebook output
    # market file can either be:
    # 1. an already-saved stitched clean file
    # 2. rebuilt from the raw contract files if no clean save exists yet
    base = Path(".")

    signal_candidates = [
        base / SIGNALS_FILENAME,
        base / DATA_SUBDIR / SIGNALS_FILENAME,
    ]
    signals_path = _find_existing_path(signal_candidates, "signals file")
    signals = pd.read_csv(signals_path)

    sig_dt = _find_col(signals, ["datetime", "date_time", "timestamp"])
    signals[sig_dt] = pd.to_datetime(signals[sig_dt])
    if sig_dt != "datetime":
        signals = signals.rename(columns={sig_dt: "datetime"})

    tau_min = infer_tau_minutes(signals)

    market_path = None
    market_1m = None

    # first try to find a stitched clean file if one exists
    clean_candidates = []
    for name in CLEAN_CANDIDATES:
        clean_candidates.append(base / name)
        clean_candidates.append(base / DATA_SUBDIR / name)

    for p in clean_candidates:
        if p.exists():
            market_path = p
            market_1m = pd.read_csv(p)
            break

    # if no clean file exists, rebuild from the raw contract files in the data folder
    if market_1m is None:
        raw_base_candidates = [
            base / DATA_SUBDIR,
            base,
        ]
        raw_base = None
        for candidate in raw_base_candidates:
            if len(glob.glob(str(candidate / RAW_GLOB))) > 0:
                raw_base = candidate
                break
        if raw_base is None:
            raise FileNotFoundError(
                f"Could not find a clean market file or raw files with pattern {RAW_GLOB}"
            )

        market_path = raw_base / f"{Path(SIGNALS_FILENAME).stem}_rebuilt_from_raw.csv"
        market_1m = build_stitched_market_from_raw(raw_base)

    market_tau = resample_market(market_1m, tau_min)
    return signals, market_tau, signals_path, market_path


def sanity_checks(signals: pd.DataFrame, tick_size: float) -> dict[str, pd.DataFrame]:
    # these checks are here so we catch obvious recommendation issues before trusting the backtest
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


def merge_signals_with_market(signals: pd.DataFrame, market_tau: pd.DataFrame) -> pd.DataFrame:
    # connects each recommendation row to the realized market bar at that timestamp
    market_use = market_tau[["datetime", "open", "high", "low", "close"]].copy()
    if "volume" in market_tau.columns:
        market_use["volume"] = market_tau["volume"]

    market_use = market_use.rename(columns={"open": "mkt_open"})
    return signals.merge(market_use, on="datetime", how="left")


def run_backtest(merged: pd.DataFrame) -> pd.DataFrame:
    # main fill logic:
    # buy fills if realized low <= limit price
    # sell fills if realized high >= limit price
    # if not filled, force execution at the bar close
    # benchmark is immediate execution at the open from the recommendation row
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
    # these are the summary tables that are most useful for reading results quickly
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

    for label_col in ["vol_state_label", "range_state_label", "trend_state_label"]:
        if label_col in merged.columns:
            state_tables[label_col] = merged.groupby(label_col).agg(
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


def choose_output_dir(signals_path: Path) -> Path:
    # keeps outputs next to the signals file so everything stays in one place
    out = signals_path.parent / "backtest_outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_tables(
    output_dir: Path,
    checks: dict[str, pd.DataFrame],
    market_tau: pd.DataFrame,
    merged: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    tau_min: int,
) -> None:
    # saves diagnostics, the resampled market file used, the row-level backtest file, and grouped summaries
    checks["missing_summary"].to_csv(output_dir / "HeatingOil_recommendation_missing_summary.csv", index=True)
    checks["bad_buys"].to_csv(output_dir / "HeatingOil_bad_buys.csv", index=False)
    checks["bad_sells"].to_csv(output_dir / "HeatingOil_bad_sells.csv", index=False)
    checks["bad_tick_rows"].to_csv(output_dir / "HeatingOil_bad_tick_rows.csv", index=False)

    market_tau.to_csv(output_dir / f"HeatingOil_{tau_min}min_from_clean_or_raw.csv", index=False)
    merged.to_csv(output_dir / "HeatingOil_backtest_results.csv", index=False)

    for name, df in summaries.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)


def make_charts(output_dir: Path, summaries: dict[str, pd.DataFrame]) -> None:
    # just a few quick visuals for the main takeaways
    if "by_direction" in summaries and not summaries["by_direction"].empty:
        df = summaries["by_direction"]
        direction_col = df.columns[0]
        plt.figure(figsize=(6, 4))
        plt.bar(df[direction_col].astype(str), df["avg_improvement"])
        plt.title("HeatingOil Average Improvement by Direction")
        plt.xlabel("Signal Direction")
        plt.ylabel("Average Improvement")
        plt.tight_layout()
        plt.savefig(output_dir / "HeatingOil_avg_improvement_by_direction.png", dpi=200)
        plt.close()

    if "calibration" in summaries and not summaries["calibration"].empty:
        df = summaries["calibration"].dropna(subset=["avg_est_fill_probability", "actual_fill_rate"])
        if not df.empty:
            plt.figure(figsize=(7, 4))
            plt.plot(df["avg_est_fill_probability"], df["actual_fill_rate"], marker="o")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.title("HeatingOil Estimated vs Actual Fill Probability")
            plt.xlabel("Estimated Fill Probability")
            plt.ylabel("Actual Fill Rate")
            plt.tight_layout()
            plt.savefig(output_dir / "HeatingOil_fill_probability_calibration.png", dpi=200)
            plt.close()

    if "by_ticks" in summaries and not summaries["by_ticks"].empty:
        df = summaries["by_ticks"]
        ticks_col = _find_col(df, ["ticks_away"])
        plt.figure(figsize=(7, 4))
        plt.bar(df[ticks_col].astype(str), df["fill_rate"])
        plt.title("HeatingOil Fill Rate by Ticks Away")
        plt.xlabel("Ticks Away")
        plt.ylabel("Fill Rate")
        plt.tight_layout()
        plt.savefig(output_dir / "HeatingOil_fill_rate_by_ticks.png", dpi=200)
        plt.close()


def write_run_report(
    output_dir: Path,
    checks: dict[str, pd.DataFrame],
    summaries: dict[str, pd.DataFrame],
    tick_size: float,
    tau_min: int,
    signals_path: Path,
    market_path: Path,
) -> None:
    # quick text summary so you do not have to open every csv first
    lines = []
    lines.append("HeatingOil Backtest Run Report")
    lines.append("=" * 26)
    lines.append("")
    lines.append("Input metadata")
    lines.append(f"- Signals file: {signals_path.resolve()}")
    lines.append(f"- Market file or source used: {market_path.resolve()}")
    lines.append(f"- Tau minutes used: {tau_min}")
    lines.append(f"- Tick size used: {tick_size}")
    lines.append("")
    lines.append("Recommendation sanity checks")
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

    (output_dir / "HeatingOil_backtest_run_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    print("loading inputs...")
    signals, market_tau, signals_path, market_path = load_inputs()

    tick_size = infer_tick_size(signals)
    tau_min = infer_tau_minutes(signals)
    output_dir = choose_output_dir(signals_path)

    print(f"using tau={tau_min} minute(s), tick_size={tick_size}")
    print("running sanity checks...")
    checks = sanity_checks(signals, tick_size)

    print("merging signals with realized market bars...")
    merged = merge_signals_with_market(signals, market_tau)

    print("running backtest...")
    merged = run_backtest(merged)

    print("building summaries...")
    summaries = build_summaries(merged)

    print("saving tables...")
    save_tables(output_dir, checks, market_tau, merged, summaries, tau_min)

    print("making charts...")
    make_charts(output_dir, summaries)

    print("writing run report...")
    write_run_report(output_dir, checks, summaries, tick_size, tau_min, signals_path, market_path)

    print(f"done. outputs written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
