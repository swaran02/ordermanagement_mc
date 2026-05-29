import pandas as pd

print("=== OVERALL ===")
print(pd.read_csv("backtest_outputs/overall.csv").to_string())

print("\n=== BY DIRECTION ===")
print(pd.read_csv("backtest_outputs/by_direction.csv").to_string())

print("\n=== BY TICKS ===")
print(pd.read_csv("backtest_outputs/by_ticks.csv").to_string())

print("\n=== BY VOL STATE ===")
print(pd.read_csv("backtest_outputs/by_vol_state.csv").to_string())

print("\n=== BY RANGE STATE ===")
print(pd.read_csv("backtest_outputs/by_range_state.csv").to_string())

print("\n=== BY TREND STATE ===")
print(pd.read_csv("backtest_outputs/by_trend_state.csv").to_string())