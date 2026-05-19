# Notebook Walkthrough — Volatility-Volume-based Order Management
### Columbia University IEOR4703 — Term Project 2
### Gold Futures (GC) — Statistical Engine

---

## Overview

This notebook builds the **statistical engine** for the order management system described in the project spec. The goal is not to decide *whether* to trade — that decision comes from the AIAgent signal file — but to decide *how* to trade once a signal is given. Specifically: where to place a limit order to maximize the chance of getting a better price than just hitting the market.

The notebook is organized into 5 steps:
1. **Data Loading & Cleaning** — load raw futures CSVs, filter bad data, stitch contracts
2. **Range Computation** — compute price movement statistics in units of ticks
3. **Empirical PDFs** — estimate probability distributions of price movement
4. **EWMA State Classification** — classify current market regime using exponential moving averages
5. **Conditional ePDFs & Order Placement** — use regime-conditioned distributions to recommend limit order prices

---

## Step 1 — Data Loading & Cleaning

### Cell 0 — `load_and_clean(filepath)` function

```python
import pandas as pd

def load_and_clean(filepath):
    df = pd.read_csv(
        filepath,
        header=None,
        names=["datetime", "open", "high", "low", "close", "volume"]
    )
    
    df["contract"] = filepath.split("/")[-1].replace(".csv", "")

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        format="%Y.%m.%d.%H:%M:%S"
    )

    df["date"] = df["datetime"].dt.date
    df["bars_per_day"] = df.groupby("date")["datetime"].transform("count")
    df_clean = df[df["bars_per_day"] > 1242].copy()

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
```

**What this does, line by line:**

`import pandas as pd`
Imports the pandas library, aliased as `pd`. Pandas is used throughout for all dataframe operations — loading CSVs, filtering rows, grouping, merging, and resampling.

`pd.read_csv(filepath, header=None, names=[...])`
Loads a single futures CSV file. `header=None` tells pandas there is no header row in the file — the raw data starts immediately from row 1. `names=[...]` manually assigns column names since the file doesn't include them. The columns are the standard OHLCV format: `datetime`, `open`, `high`, `low`, `close`, `volume`.

`df["contract"] = filepath.split("/")[-1].replace(".csv", "")`
Creates a new column called `contract` that stores the contract name (e.g. "GCG24"). `filepath.split("/")[-1]` takes the last part of the file path (the filename), and `.replace(".csv", "")` strips the file extension. This is important later when stitching multiple contracts together — you need to know which row came from which contract.

`pd.to_datetime(df["datetime"], format="%Y.%m.%d.%H:%M:%S")`
Converts the datetime column from a raw string (e.g. `"2023.12.05.02:31:00"`) to a proper Python datetime object. The `format` argument tells pandas exactly how the string is structured. Without this, pandas would either fail or guess incorrectly. Once parsed, you can extract components like `.dt.date`, `.dt.hour`, `.dt.minute` etc.

`df["date"] = df["datetime"].dt.date`
Extracts just the calendar date (no time) from the datetime column. This is needed to group by day in the next step.

`df["bars_per_day"] = df.groupby("date")["datetime"].transform("count")`
Counts how many 1-minute bars exist for each calendar date, then assigns that count back to every row belonging to that date. `groupby("date")` groups the dataframe by calendar day. `["datetime"].count()` counts rows in each group. `.transform("count")` is what makes this a broadcast — instead of returning one row per date, it returns the count for every row in that date's group, so each row knows how many bars its day has. This is necessary to filter at the row level in the next step.

**Why 1,242 bars?**
Gold futures trade on CME Globex nearly 24 hours a day, Sunday through Friday. Each trading day runs from 6:00 PM to 5:00 PM ET the next day — a 23-hour session — which equals 1,380 minutes. The project spec (Section 2.1) states that days with many missing data/trades should be discarded, specifically "the number of trading minutes should be more than 90% to be considered." 90% of 1,380 = 1,242. Days below this threshold are considered too sparse to produce reliable statistics.

`df_clean = df[df["bars_per_day"] > 1242].copy()`
Keeps only rows that belong to days with more than 1,242 bars. `.copy()` creates an independent copy of the filtered dataframe rather than a view of the original — this prevents a common pandas warning called `SettingWithCopyWarning` when you later modify `df_clean`.

**OHLCV Sanity Checks:**
```python
valid_ohlc = (
    (df_clean["high"] >= df_clean["low"]) &
    (df_clean["open"] >= df_clean["low"]) &
    (df_clean["open"] <= df_clean["high"]) &
    (df_clean["close"] >= df_clean["low"]) &
    (df_clean["close"] <= df_clean["high"]) &
    (df_clean["volume"] > 0)
)
```
This creates a boolean Series — `True` for every row that passes all checks, `False` for rows that fail any. Each condition is a vectorized comparison across the entire column. The `&` operator combines conditions with logical AND — a row must pass every single condition to survive.

The conditions enforce the fundamental OHLC relationships:
- High must be ≥ Low (a bar where high < low is physically impossible)
- Open must be within [Low, High] — the opening price must fall within the bar's range
- Close must be within [Low, High] — same for closing price
- Volume must be positive — a bar with zero volume means no trades occurred

`df_clean = df_clean[valid_ohlc].copy()`
Applies the boolean mask to keep only valid rows.

`return df_clean`
Returns the cleaned dataframe for this one contract file.

---

### Cell 1 — Load and concatenate all 4 Gold contracts

```python
files = [
    "data/Gold/GCG24.csv",
    "data/Gold/GCJ24.csv",
    "data/Gold/GCM24.csv",
    "data/Gold/GCQ24.csv"
]

all_dfs = [load_and_clean(f) for f in files]
df_all = pd.concat(all_dfs)
df_all = df_all.sort_values("datetime").reset_index(drop=True)

print(df_all["contract"].value_counts())
print("Total liquid days:", df_all["date"].nunique())
```

**What this does:**

`files = [...]`
A list of file paths for all four Gold futures contracts available in the data folder:
- **GCG24** = Gold February 2024 contract (G = February, 24 = 2024)
- **GCJ24** = Gold April 2024 contract (J = April)
- **GCM24** = Gold June 2024 contract (M = June)
- **GCQ24** = Gold August 2024 contract (Q = August)

These contracts overlap in time — as one approaches expiry, traders roll into the next. That's why we need all four.

`[load_and_clean(f) for f in files]`
A list comprehension that calls `load_and_clean()` on each file and collects the results into a list of four dataframes.

`pd.concat(all_dfs)`
Concatenates all four dataframes into one. At this point, some dates will have rows from two different contracts (the overlap period before a roll). This is resolved in the next cell.

`sort_values("datetime").reset_index(drop=True)`
Sorts all rows chronologically. `reset_index(drop=True)` resets the row index to 0, 1, 2, ... because after concatenation the index values may be duplicated or out of order. `drop=True` discards the old index rather than adding it as a column.

`value_counts()` — shows how many bars each contract contributed after cleaning.
`nunique()` — counts distinct calendar dates across all contracts.

---

### Cell 2 — Contract stitching by liquidity

```python
daily_vol = df_all.groupby(["date", "contract"])["volume"].sum().reset_index()

best_contract = daily_vol.loc[
    daily_vol.groupby("date")["volume"].idxmax(),
    ["date", "contract"]
]

df_stitched = df_all.merge(
    best_contract,
    on=["date", "contract"]
).reset_index(drop=True)

print(df_stitched["contract"].value_counts())
print("Total days after stitching:", df_stitched["date"].nunique())
```

**What this does:**

This cell implements the contract roll logic described in Figure 1 of the project spec. Before a contract expires, the next contract becomes more liquid — more volume starts flowing into it. The spec says to switch to the more liquid contract. We implement this by asking: for each day, which contract had more total volume? That contract "wins" and its bars are kept; the other contract's bars for that day are discarded.

`df_all.groupby(["date", "contract"])["volume"].sum()`
Groups by both date AND contract, then sums volume. This gives total daily volume for each (date, contract) pair. `.reset_index()` brings the group keys back as columns.

`daily_vol.groupby("date")["volume"].idxmax()`
For each date, finds the index (row number) of the row with the highest volume. This identifies which contract was most liquid on each day.

`daily_vol.loc[..., ["date", "contract"]]`
Uses those indices to extract just the `date` and `contract` columns — giving you a table of "on date X, use contract Y."

`df_all.merge(best_contract, on=["date", "contract"])`
Merges the full data with the winner table. An inner merge keeps only rows where both the date AND the contract match — effectively filtering out bars from the losing contract on each day.

---

### Cell 3 — Verify non-overlapping date ranges

```python
print(df_stitched.groupby("contract")["date"].agg(["min", "max"]))
```

**What this does:**

Groups by contract and finds the earliest and latest date for each. This is a sanity check — after stitching, each contract should cover a clean, non-overlapping date range. The expected output is:
- GCG24: Nov 2023 → Jan 2024
- GCJ24: Jan 2024 → Mar 2024
- GCM24: Mar 2024 → May 2024
- GCQ24: May 2024 → Jul 2024

If you saw overlapping ranges here, it would mean the stitching failed.

---

### Cell 4 — Final cleanup

```python
df_stitched = df_stitched.drop(columns=["bars_per_day"])
df_stitched = df_stitched.reset_index(drop=True)
print(df_stitched.info())
```

**What this does:**

`drop(columns=["bars_per_day"])`
Removes the `bars_per_day` helper column that was created inside `load_and_clean()`. It served its purpose (filtering low-liquidity days) and is no longer needed.

`reset_index(drop=True)`
Resets the row index to clean sequential integers after all the filtering and merging operations.

`df_stitched.info()`
Prints a concise summary: number of rows, column names, data types, and non-null counts. This is the final check that the clean dataset is complete with no missing values.

**Result:** 181,787 clean 1-minute bars across 132 trading days, covering Nov 27, 2023 → Jul 25, 2024.

---

## Step 2 — Range Computation

### Cells 5–7 — Compute range columns on 1-min data

```python
df_stitched["range_ticks"] = (
    (df_stitched["high"] - df_stitched["low"]) / 0.10
).round().astype(int)

df_stitched["range_up"] = (
    (df_stitched["high"] - df_stitched["open"]) / 0.10
).round().astype(int)

df_stitched["range_dn"] = (
    (df_stitched["open"] - df_stitched["low"]) / 0.10
).round().astype(int)
```

**What this does:**

These three columns implement the core definitions from Section 3 of the project spec.

**Range** (Section 3.3): `R_t,τ = H − L`
The total price movement from low to high within the interval, expressed in ticks (minimum price increments). Dividing by the tick size `ε = 0.10` converts from dollars to number of ticks. For example, if H = 2034.7 and L = 2034.0, Range = 0.7 / 0.10 = 7 ticks.

**RangeUp** (Section 3.4): `R^(U)_t,τ = H − O`
How far the price rose above the open. This represents the maximum upside opportunity within the bar. A limit sell order placed `k` ticks above the open would have been filled if RangeUp ≥ k.

**RangeDn** (Section 3.4): `R^(D)_t,τ = O − L`
How far the price fell below the open. This represents the maximum downside dip within the bar. A limit buy order placed `k` ticks below the open would have been filled if RangeDn ≥ k.

The spec also states: `R_t,τ = R^(U)_t,τ + R^(D)_t,τ`. This is verified in the next cell.

`.round().astype(int)`
Due to floating point arithmetic, `(2034.7 - 2034.0) / 0.10` might give `6.999999` instead of `7`. `.round()` fixes this, and `.astype(int)` converts to integer since tick counts are always whole numbers.

`0.10`
The tick size for Gold futures (GC). This is the minimum price increment — the smallest amount the Gold price can move. It is market-specific: for example, S&P 500 E-Mini (ES) has a tick size of 0.25, Nasdaq (NQ) has 0.25, etc.

**Cell 7 — Range identity check:**
```python
check = (df_stitched["range_up"] + df_stitched["range_dn"] == df_stitched["range_ticks"]).all()
print("Range check passes:", check)
```
Verifies the identity R = RangeUp + RangeDn holds for every single row. `.all()` returns `True` only if every element of the boolean Series is `True`. If this prints `False`, it means there's a floating point rounding error somewhere.

---

### Cell 8 — `resample_ohlcv(df, tau)` function

```python
def resample_ohlcv(df, tau):
    temp = df.copy()
    temp = temp.set_index("datetime")
    df_resampled = temp.resample(f"{tau}min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })
    df_resampled = df_resampled.dropna()
    df_resampled = df_resampled.reset_index()
    return df_resampled
```

**What this does:**

The project spec (Section 3.1) requires analyzing data at multiple holding periods τ: 5, 10, 15, 30, and 60 minutes. Since the raw data is at 1-minute resolution, this function aggregates 1-min bars into τ-minute bars following the OHLCV convention described in Section 2:

- **Open**: the open price of the first 1-min bar in the interval
- **High**: the maximum high across all 1-min bars in the interval
- **Low**: the minimum low across all 1-min bars in the interval
- **Close**: the close price of the last 1-min bar in the interval
- **Volume**: the total volume summed across all bars in the interval

`df.copy()`
Creates a copy so the original `df_stitched` is not modified.

`set_index("datetime")`
Makes datetime the index, which is required for pandas `resample()` to work — it needs to know the time axis.

`resample(f"{tau}min")`
Groups the data into consecutive τ-minute windows. `f"{tau}min"` is an f-string that inserts the value of `tau` — e.g. if `tau=5`, this becomes `"5min"`.

`.agg({...})`
Applies different aggregation functions to different columns simultaneously. This is how you produce a proper OHLCV bar from multiple 1-min bars.

`dropna()`
Removes time windows where no trades occurred (empty bars). These appear as NaN after resampling because there's no data to aggregate.

`reset_index()`
Brings the datetime back from the index into a regular column.

---

### Cells 9–10 — Test resample at τ=5min

```python
df_5min = resample_ohlcv(df_stitched, 5)
print(df_5min.head())
print(df_5min.shape)
```

Applies the resampling function to create 5-minute bars and prints the first few rows and the dimensions. Then range columns are added and the identity check is run again — this time on the 5-min data — to confirm the ranges computed on aggregated bars still satisfy R = RangeUp + RangeDn.

---

### Cells 11–14 — `compute_ranges()` function and final τ comparison

```python
def compute_ranges(df, tick_size):
    df = df.copy()
    df["range_ticks"] = ((df["high"] - df["low"]) / tick_size).round().astype(int)
    df["range_up"] = ((df["high"] - df["open"]) / tick_size).round().astype(int)
    df["range_dn"] = ((df["open"] - df["low"]) / tick_size).round().astype(int)
    return df

TICK_SIZE = 0.10  # Gold
df_5min = compute_ranges(resample_ohlcv(df_stitched, 5), TICK_SIZE)
df_15min = compute_ranges(resample_ohlcv(df_stitched, 15), TICK_SIZE)
```

**What this does:**

Wraps the range calculation into a reusable function that takes `tick_size` as a parameter. This makes the code generic — to run it on Nasdaq instead of Gold, you just change `TICK_SIZE = 0.25`. The function is then used to produce both 5-min and 15-min dataframes in a single readable line each.

`TICK_SIZE = 0.10`
Defined as a named constant (all caps by Python convention) so it appears exactly once and is easy to change for other markets.

Cell 14 prints the average range_ticks for both resolutions:
- 5-min average ≈ 15 ticks
- 15-min average ≈ 27 ticks

This is an important intuition check — wider time windows should capture more price movement, so the 15-min average must be larger than the 5-min average.

---

### Cell 15 — Save cleaned data

```python
df_stitched.to_csv("data/Gold/GC_clean.csv", index=False)
```

Saves the cleaned 1-min stitched data to disk. `index=False` prevents pandas from writing the row numbers as a column. This means you can reload the cleaned data at any time with `pd.read_csv()` without re-running all the cleaning steps.

---

## Step 3 — Empirical Probability Distributions (ePDFs)

### Cell 17 — Build the unconditional ePDF for range_dn

```python
range_dn_counts = df_5min["range_dn"].value_counts()
range_dn_epdf = range_dn_counts / len(df_5min)
range_dn_epdf = range_dn_epdf.sort_index()
print(range_dn_epdf)
```

**What this does:**

This cell implements Section 4 of the project spec — computing `P(R^(D)_t,τ = l)` for l = 0, 1, 2, ...

`value_counts()`
Counts how many times each distinct value of `range_dn` appears in the dataset. For example, if `range_dn = 3` appears 800 times out of 36,432 total bars, the count for 3 is 800.

`/ len(df_5min)`
Divides every count by the total number of bars to convert raw counts to probabilities. This is the definition of an empirical probability — frequency of observed outcomes. For example, 800 / 36,432 ≈ 0.022, meaning P(range_dn = 3) ≈ 2.2%.

`sort_index()`
Sorts by the tick value (0, 1, 2, 3, ...) rather than by frequency. This makes the distribution readable as a proper PDF from left to right.

This is the **unconditional ePDF** — built from all bars regardless of market conditions. It answers: "across all history, what fraction of 5-minute bars saw the price dip exactly k ticks below the open?" This serves as a baseline before we condition on market state.

---

### Cells 18–19 — `compute_epdfs()` function

```python
def compute_epdfs(df):
    epdfs = {}
    for col in ["range_ticks", "range_up", "range_dn"]:
        counts = df[col].value_counts()
        epdfs[col] = (counts / len(df)).sort_index()
    return epdfs

epdfs_5min = compute_epdfs(df_5min)
epdfs_15min = compute_epdfs(df_15min)
```

**What this does:**

Generalizes the single ePDF calculation into a function that computes all three distributions at once (range_ticks, range_up, range_dn) and returns them as a dictionary. Each key maps to a pandas Series indexed by tick value with probability values.

`epdfs["range_dn"]` then gives you P(range_dn = k) for all observed k values.

---

### Cell 20 — Plot unconditional ePDFs

```python
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(epdfs_5min["range_dn"].index, epdfs_5min["range_dn"].values, alpha=0.6, label="5min")
ax.bar(epdfs_15min["range_dn"].index, epdfs_15min["range_dn"].values, alpha=0.6, label="15min")
ax.set_xlim(0, 50)
plt.show()
```

**What this does:**

Plots both distributions overlaid as bar charts — analogous to the bottom-left panel of Figure 2 in the project spec. The x-axis is the number of ticks, the y-axis is probability.

`alpha=0.6`
Sets partial transparency so both distributions are visible where they overlap.

`set_xlim(0, 50)`
Zooms in to the first 50 ticks. The tail extends much further but has negligible probability — showing 0–50 captures the meaningful part of the distribution.

The plot confirms the expected relationship: the 5-min distribution is concentrated at lower tick values (shorter windows = smaller moves), while the 15-min distribution is flatter with a heavier tail (more time = larger potential moves).

---

## Step 4 — EWMA State Classification

### Cell 22 — `compute_ewma(series, m)` function

```python
import numpy as np

def compute_ewma(series, m):
    values = series.astype(float).to_numpy()
    lambda_ = 2 ** (-1 / m)

    ewma = np.zeros(len(values))
    ewmv = np.zeros(len(values))
    sumW = 0.0
    sumWX = 0.0
    sumWSS = 0.0

    ewma[0] = np.nan
    ewmv[0] = np.nan

    for j in range(1, len(values)):
        prev_x = values[j - 1]
        sumW = lambda_ * sumW + 1
        sumWX = lambda_ * sumWX + prev_x
        ewma[j] = sumWX / sumW
        sumWSS = lambda_ * sumWSS + (prev_x - ewma[j]) ** 2
        ewmv[j] = np.sqrt(sumWSS / sumW)

    return pd.DataFrame({"ewma": ewma, "ewmv": ewmv}, index=series.index)
```

**What this does:**

This is a direct implementation of **Algorithm 1** from the project spec. It computes a recursively updated exponentially weighted moving average (EWMA) and exponentially weighted moving standard deviation (EWMV) of any input series. Note: despite the variable being named `ewmv` (following the paper's notation), the code computes the square root of the weighted variance — so it is technically an EW moving standard deviation, not variance. This is the more useful quantity for regime detection since it is in the same units as the original series.

**Why exponential weighting?**
Simple moving averages weight all past observations equally. Exponential weighting gives more importance to recent observations and less to older ones — this is more appropriate for market data because recent conditions are more predictive of the immediate future than conditions from weeks ago.

`lambda_ = 2 ** (-1 / m)`
The decay factor λ. With half-life `m`, after `m` bars an observation's relative weight decays to half. A small `m` (e.g. 5) makes the average very reactive — it responds quickly to new data but is noisy. A large `m` (e.g. 50) makes it smooth but slow to adapt.

`values = series.astype(float).to_numpy()`
Converts the pandas Series to a plain NumPy array for efficient loop access. `.astype(float)` ensures the values are floating point (not integers), which is needed for the arithmetic.

`ewma[0] = np.nan` and `ewmv[0] = np.nan`
The first row has no previous observation (there is no `η_{j-1}` when j=0), so it is set to NaN. This is faithful to Algorithm 1 and avoids initializing with an arbitrary value.

**The recursive loop — following Algorithm 1 exactly:**

`prev_x = values[j - 1]`
Uses `η_{j-1}` — the **previous** bar's value, not the current one. This is critical to avoid forward-looking bias. The EWMA at time j must only use information known before time j. If you used `values[j]` instead, you'd be incorporating the current bar's information into the state used to analyze that same bar — which is cheating.

`sumW = lambda_ * sumW + 1`
Accumulates the sum of weights. Each old weight is decayed by λ, and the current bar contributes a weight of 1. This normalization ensures the EWMA stays properly scaled even in the early bars when you have little history.

`sumWX = lambda_ * sumWX + prev_x`
Accumulates the weighted sum of values. Old contributions decay by λ, and the previous observation is added with weight 1.

`ewma[j] = sumWX / sumW`
The EWMA at step j — the weighted average, normalized by the sum of weights.

`sumWSS = lambda_ * sumWSS + (prev_x - ewma[j]) ** 2`
Accumulates the weighted sum of squared deviations from the mean. This is the numerator of the weighted variance — before taking the square root.

`ewmv[j] = np.sqrt(sumWSS / sumW)`
The exponentially weighted moving **standard deviation** — the square root of the weighted variance. Despite being named `ewmv`, this outputs a standard deviation, not a variance. It is in the same units as the original series (ticks or volume), making it directly interpretable as a volatility estimate.

---

### Cells 23–24 — Apply EWMA to volume and range

```python
volume_state = compute_ewma(df_5min["volume"], m=10)
range_state = compute_ewma(df_5min["range_ticks"], m=10)

df_5min[["ewma_volume", "ewmv_volume"]] = volume_state
df_5min[["ewma_range", "ewmv_range"]] = range_state
```

**What this does:**

Applies the EWMA function to two quantities:
- **Volume** (`df_5min["volume"]`) — how busy is the market? High volume = high activity
- **Range** (`df_5min["range_ticks"]`) — how volatile is the market? High range = large price swings

`m=10` means a half-life of 10 bars. At 5-minute resolution, 10 bars = 50 minutes. This means the EWMA at any given bar is primarily influenced by the past ~50 minutes of data, with older data fading out exponentially.

The two output columns per quantity:
- `ewma_volume` / `ewma_range` — the smoothed level (is it currently high or low?)
- `ewmv_volume` / `ewmv_range` — the exponentially weighted moving **standard deviation** of the level (is it stable or changing rapidly?). Note: named `ewmv` following the paper's notation but contains standard deviation values, not variance.

These four new columns are added directly to `df_5min`.

---

### Cell 25 — Bin EWMA values into discrete states

```python
df_5min["vol_state"] = pd.qcut(df_5min["ewma_volume"], q=3, labels=[1, 2, 3])
df_5min["range_state"] = pd.qcut(df_5min["ewma_range"], q=3, labels=[1, 2, 3])
```

**What this does:**

`pd.qcut(series, q=3, labels=[1, 2, 3])`
Divides the EWMA values into 3 equal-frequency bins based on percentiles, then labels them 1, 2, 3. "Equal-frequency" means exactly one-third of all observations fall in each bin — `qcut` finds the percentile boundaries automatically. This ensures all states are equally populated, which is important for building reliable statistics in each state.

- State 1 = bottom third (low volume or low volatility)
- State 2 = middle third (medium)
- State 3 = top third (high volume or high volatility)

**Important caveat (noted in Cell 26):**
`qcut` computes bin boundaries using all data at once. Strictly speaking, this introduces mild forward-looking bias — the boundary between "low" and "medium" volume is computed using future data. In a production system, you would compute boundaries on a rolling basis using only past data. For this project it is an acceptable approximation and is flagged with a comment.

---

### Cell 26 — Trend state (Δx)

```python
df_5min["delta_x"] = df_5min["open"].diff()
df_5min["trend_state"] = pd.qcut(df_5min["delta_x"], q=3, labels=[1, 2, 3])
```

**What this does:**

`df_5min["open"].diff()`
Computes the bar-to-bar change in open price: `Δx_j = x_j - x_{j-1}`. This is the `Δx_{j-1}` quantity from the project spec (Section 5). It captures the recent price direction — is the market trending up, down, or sideways?

Then `qcut` bins these into 3 states:
- State 1 = downtrend (negative Δx, bottom third)
- State 2 = flat (near-zero Δx, middle third)
- State 3 = uptrend (positive Δx, top third)

The first row is NaN because there's no previous bar to diff against.

Together, `vol_state`, `range_state`, and `trend_state` define the full market regime as described in Section 5 of the spec:
`P(R^(D)_t,τ = l | v_{j-1} ∈ m, σ_{j-1} ∈ n, Δx_{j-1} ∈ k)`

---

## Step 5 — Conditional ePDFs & Order Placement

### Cell 28 — Count observations per state combination

```python
state_counts = df_5min.groupby(
    ["vol_state", "range_state", "trend_state"]
).size().reset_index(name="count")
print(state_counts)
print("Total combinations:", len(state_counts))
```

**What this does:**

With 3 possible values for each of 3 state dimensions, there are 3 × 3 × 3 = **27 possible state combinations**. This cell counts how many 5-min bars fall into each combination.

This is a diagnostic step — some combinations may have very few observations (e.g. low volume but high volatility is a rare regime). Combinations with too few observations cannot produce statistically reliable ePDFs.

---

### Cell 29 — `compute_conditional_epdfs()` function

```python
def compute_conditional_epdfs(df, min_count=30):
    cond_epdfs = {}
    grouped = df.groupby(["vol_state", "range_state", "trend_state"])
    
    for state, group in grouped:
        if len(group) < min_count:
            continue
        cond_epdfs[state] = {
            "range_dn": (group["range_dn"].value_counts() / len(group)).sort_index(),
            "range_up": (group["range_up"].value_counts() / len(group)).sort_index()
        }
    
    print(f"Valid state combinations: {len(cond_epdfs)} / 27")
    return cond_epdfs

cond_epdfs = compute_conditional_epdfs(df_5min)
```

**What this does:**

This is the core of the entire project — building a separate ePDF for each market regime.

`df.groupby(["vol_state", "range_state", "trend_state"])`
Groups the data by all three state dimensions simultaneously. Each group is a subset of bars that share the same vol_state, range_state, and trend_state.

`for state, group in grouped:`
Iterates over each (state_tuple, subset_dataframe) pair. `state` is a tuple like `(1, 2, 3)` and `group` is the dataframe of all bars in that regime.

`if len(group) < min_count: continue`
Skips states with fewer than 30 observations. Building an ePDF from 3 bars would be meaningless — you'd get a distribution that's essentially noise. 30 is a reasonable minimum for any empirical distribution estimate.

`cond_epdfs[state] = {"range_dn": ..., "range_up": ...}`
Stores a dictionary for each state containing **two** ePDFs — one for `range_dn` and one for `range_up`. This is important: buy signals need the `range_dn` distribution (how far the price dips below the open) while sell signals need `range_up` (how far the price rises above the open). Storing both under the same state key means a single lookup gives you whichever direction you need.

Previously this stored only `range_dn` as a plain Series. The updated structure stores a dict so both buy and sell signals can be served from the same `cond_epdfs` object.

`cond_epdfs[(1, 2, 3)]["range_dn"]` gives P(range_dn = k | vol_state=1, range_state=2, trend_state=3).
`cond_epdfs[(1, 2, 3)]["range_up"]` gives P(range_up = k | vol_state=1, range_state=2, trend_state=3).

`cond_epdfs = compute_conditional_epdfs(df_5min)`
This line actually calls the function and assigns the result. The function definition alone does nothing — this call is what populates `cond_epdfs`.

**Result:** 24 out of 27 combinations are valid. The 3 that were skipped (vol_state=1, range_state=3) had only 2–6 observations — they represent the physically unusual combination of low volume but high volatility.

---

### Cell 30 — Compare extreme states

```python
state_low = (1, 1, 1)
state_high = (3, 3, 3)

print("Low state ePDF (first 10 ticks):")
print(cond_epdfs[state_low]["range_dn"].head(10))

print("\nHigh state ePDF (first 10 ticks):")
print(cond_epdfs[state_high]["range_dn"].head(10))

print("Mean range_dn — low state:", 
      np.sum(cond_epdfs[state_low]["range_dn"].index.astype(float) * cond_epdfs[state_low]["range_dn"].values).round(2))
print("Mean range_dn — high state:", 
      np.sum(cond_epdfs[state_high]["range_dn"].index.astype(float) * cond_epdfs[state_high]["range_dn"].values).round(2))
```

**What this does:**

Compares the two most extreme regimes — the calmest possible market vs the most active — to validate that the conditioning is meaningful.

`cond_epdfs[state_low]["range_dn"]`
Now that `cond_epdfs[state]` is a dictionary, you must specify which ePDF you want using the key `"range_dn"` or `"range_up"`. Accessing `cond_epdfs[state_low]` alone gives you the dict, not a Series — so `.head()` or arithmetic directly on it would fail. Always include the key.

`np.sum(index.astype(float) * values)`
Computes the expected value (mean) of the distribution: E[range_dn] = Σ k × P(range_dn = k). This is the dot product of tick values and their probabilities. `.astype(float)` is needed because the index contains integers and numpy needs consistent types for the multiplication.

**Result:**
- Low state (1,1,1): mean range_dn = **4.37 ticks**
- High state (3,3,3): mean range_dn = **12.45 ticks**

Nearly 3× difference. In a calm, low-volume downtrend, Gold moves only about 4 ticks below the open on average in 5 minutes. In a volatile, high-volume uptrend, it moves nearly 13 ticks. This validates that the regime conditioning captures genuinely different market behaviors.

---

### Cell 31 — Plot conditional ePDFs

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(cond_epdfs[state_low]["range_dn"].index, cond_epdfs[state_low]["range_dn"].values)
axes[0].set_xlim(0, 50)
axes[0].set_title("Low Vol/Volume State (1,1,1)")
axes[0].set_xlabel("range_dn (ticks)")
axes[0].set_ylabel("Probability")

axes[1].bar(cond_epdfs[state_high]["range_dn"].index, cond_epdfs[state_high]["range_dn"].values, color="orange")
axes[1].set_xlim(0, 50)
axes[1].set_title("High Vol/Volume State (3,3,3)")
axes[1].set_xlabel("range_dn (ticks)")

plt.suptitle("Conditional ePDF of RangeDn — Gold 5min")
plt.tight_layout()
plt.show()
```

Side-by-side bar charts of the two extreme states. Both access `["range_dn"]` explicitly since `cond_epdfs[state]` is now a dictionary. The low state shows a steep decay concentrated within 0–15 ticks. The high state is flat, with significant probability spread out to 40+ ticks. This visual directly corresponds to Figure 2 in the project spec, showing how ePDFs differ across market segments.

---

### Cell 33 — `load_signals()` function

```python
from datetime import datetime, timedelta

def load_signals(filepath):
    df = pd.read_csv(filepath, header=None,
                     names=["date_serial", "hour", "minute", "price", "signal"])
    
    excel_epoch = datetime(1899, 12, 30)
    df["datetime"] = df.apply(
        lambda r: excel_epoch
                  + timedelta(days=int(r["date_serial"]))
                  + timedelta(hours=int(r["hour"]))
                  + timedelta(minutes=int(r["minute"])),
        axis=1
    )
    
    df["signal_direction"] = df["signal"].apply(
        lambda s: "buy" if s > 0 else ("sell" if s < 0 else "flat")
    )
    
    return df[["datetime", "price", "signal", "signal_direction"]].sort_values("datetime").reset_index(drop=True)
```

**What this does:**

Loads the AIAgent signal file, which tells us *when* to trade. The statistical engine (everything above) tells us *how* to trade — this function loads the *when*.

**AIAgent file format:**
The file has 5 columns: `date_serial, hour, minute, price, signal`

`date_serial` uses Excel's date serial format — a number representing days since December 30, 1899. For example, 45293 = January 2, 2024.

`excel_epoch = datetime(1899, 12, 30)`
The reference point for Excel date serials.

`timedelta(days=int(r["date_serial"]))`
Converts the serial number to a proper date by adding that many days to the epoch.

`timedelta(hours=...) + timedelta(minutes=...)`
Adds the hour and minute to get the full timestamp.

**Signal interpretation:**
The signal column contains integers, not just -1/0/1:
- Positive values (1, 2, 3, ..., 8) → **Buy** signal. Larger magnitude = stronger signal
- Negative values (-1, -2, ..., -7) → **Sell** signal. More negative = stronger signal
- Zero → **Flat** (no trade)

`lambda s: "buy" if s > 0 else ("sell" if s < 0 else "flat")`
A one-line function that maps any positive number to "buy", any negative to "sell", and zero to "flat". The signal magnitude is preserved in the `signal` column for later use if needed.

---

### Cell 34 — Merge signals with market state

```python
active_signals = signals[signals["signal_direction"] != "flat"].copy()

df_signals = active_signals.merge(
    df_5min[["datetime", "vol_state", "range_state", "trend_state", "open"]],
    on="datetime",
    how="inner"
)
```

**What this does:**

`signals[signals["signal_direction"] != "flat"]`
Filters to only actionable signals — flat/no-trade signals are discarded since we only need to place orders when there's a buy or sell decision.

`.merge(..., on="datetime", how="inner")`
Joins the signal data with the 5-min market data on timestamp. An inner join keeps only rows where a signal timestamp exactly matches a 5-min bar timestamp. This attaches the market state (vol_state, range_state, trend_state) and the open price at the time of each signal.

**Result:** 25,284 active signals before merge → 19,636 after. The ~5,600 lost signals fall outside the cleaned data's date range (the AIAgent covers Jan–May 2024 but some of those periods may not have matching clean bars).

---

### Cell 35 — `get_order_recommendation()` function

```python
def get_order_recommendation(signal_direction, open_price, state, 
                              cond_epdfs, tick_size=0.10, target_prob=0.70):
    
    # Pick correct ePDF based on signal direction
    if signal_direction == "buy":
        epdf = cond_epdfs[state].get("range_dn") if isinstance(cond_epdfs[state], dict) else cond_epdfs[state]
        direction = -1  # place below open
    else:
        epdf = cond_epdfs[state].get("range_up") if isinstance(cond_epdfs[state], dict) else cond_epdfs[state]
        direction = 1   # place above open
    
    if epdf is None:
        return None
    
    # Compute survival function P(range >= k)
    cumulative = epdf.sort_index().cumsum()
    survival = 1 - cumulative
    
    # Find k where survival probability >= target
    valid = survival[survival >= target_prob]
    if len(valid) == 0:
        k = 0
    else:
        k = valid.index[-1]
    
    limit_price = open_price + direction * k * tick_size
    
    return {
        "limit_price": round(limit_price, 2),
        "ticks_away": k,
        "fill_probability": round(survival.get(k, 0), 3)
    }
```

**What this does:**

This is the final output of the statistical engine — given a signal, it recommends exactly where to place a limit order.

**Parameters:**
- `signal_direction`: "buy" or "sell"
- `open_price`: the current open price at the time of the signal
- `state`: a tuple (vol_state, range_state, trend_state) identifying the current regime
- `cond_epdfs`: the dictionary of conditional ePDFs built in Cell 29
- `tick_size`: market-specific minimum price increment (0.10 for Gold)
- `target_prob`: the minimum acceptable fill probability (default 70%)

**Why range_dn for buy, range_up for sell:**
- For a **buy** signal: you want to purchase below the open price. You place a limit order k ticks below. It gets filled if the price dips at least k ticks — i.e. if range_dn ≥ k. So you use the range_dn distribution.
- For a **sell** signal: you want to sell above the open price. You place a limit order k ticks above. It gets filled if the price rises at least k ticks — i.e. if range_up ≥ k. So you use the range_up distribution.

`direction = -1` for buy (subtract ticks from open), `direction = 1` for sell (add ticks to open).

`cond_epdfs[state].get("range_dn") if isinstance(cond_epdfs[state], dict) else cond_epdfs[state]`
This defensive pattern checks whether the value stored for this state is a dictionary (the current structure) before calling `.get()`. If it were a plain Series, `.get()` would fail. The `isinstance` check makes the function robust to structural changes and is good defensive coding practice.

**Survival function:**

`cumulative = epdf.sort_index().cumsum()`
The cumulative distribution function (CDF): P(range_dn ≤ k).

`survival = 1 - cumulative`
The survival function. Strictly speaking this gives P(range_dn > k), not P(range_dn ≥ k). Since range values are discrete integers, P(range_dn ≥ k) = P(range_dn > k-1) = 1 - CDF(k-1). The implementation uses 1 - CDF(k) which is slightly conservative — it underestimates the true fill probability by the probability mass exactly at k. For practical purposes (tick values are integers and the difference is one tick) this is acceptable and makes the strategy slightly more conservative, which is preferable to overestimating fill probability.

`valid = survival[survival >= target_prob]`
Finds all tick levels where the fill probability exceeds the target (default 70%).

`k = valid.index[-1]`
Takes the largest k that still meets the target. This is the furthest away you can place the order while still having at least a 70% chance of being filled — i.e. the best possible price improvement while maintaining adequate fill probability.

`limit_price = open_price + direction * k * tick_size`
Converts ticks back to a dollar price.

**Returns a dictionary:**
- `limit_price`: the actual price to submit the limit order at
- `ticks_away`: how many ticks from the open the order is placed
- `fill_probability`: the estimated probability the order gets filled based on historical data

---

### Cell 36 — Test the recommendation engine

```python
row = df_signals[df_signals["signal_direction"] == "buy"].iloc[0]
state = (row["vol_state"], row["range_state"], row["trend_state"])
result = get_order_recommendation("buy", row["open"], state, cond_epdfs)
print("Buy recommendation:", result)

row = df_signals[df_signals["signal_direction"] == "sell"].iloc[0]
state = (row["vol_state"], row["range_state"], row["trend_state"])
result = get_order_recommendation("sell", row["open"], state, cond_epdfs)
print("Sell recommendation:", result)
```

**What this does:**

Tests the full pipeline on real signals. `.iloc[0]` takes the first row matching the filter. The state tuple is constructed from the three state columns. The recommendation function is called and the result is printed.

**Results:**
- Buy: limit at 2083.60 (1 tick below open), 77.2% fill probability
- Sell: limit at 2084.60 (2 ticks above open), 73.4% fill probability

Both exceed the 70% target threshold. This output is what gets handed to the backtesting module — the backtester checks whether each recommended limit order would have actually been filled within the holding period τ, and tracks the resulting price improvement vs. a naive market order.

---

### Cell 37 — Apply recommendations to all signals

```python
df_signals["limit_price"] = None
df_signals["ticks_away"] = None
df_signals["fill_probability"] = None

for idx, row in df_signals.iterrows():
    state = (row["vol_state"], row["range_state"], row["trend_state"])
    if state not in cond_epdfs:
        continue
    result = get_order_recommendation(
        row["signal_direction"], 
        row["open"], 
        state, 
        cond_epdfs
    )
    if result:
        df_signals.at[idx, "limit_price"] = result["limit_price"]
        df_signals.at[idx, "ticks_away"] = result["ticks_away"]
        df_signals.at[idx, "fill_probability"] = result["fill_probability"]

print(df_signals.head(10))
print("Signals with recommendations:", df_signals["limit_price"].notna().sum())
```

**What this does:**

This cell scales the recommendation engine from a single test row to the entire signal dataset. It loops over every row in `df_signals` and computes a limit order recommendation for each one, then stores the results back into the dataframe.

`df_signals["limit_price"] = None` (and same for `ticks_away`, `fill_probability`)
Initializes three new columns with `None`. This ensures the columns exist before the loop tries to write into them, and any signals that don't get a recommendation (e.g. sparse states) remain `None`.

`for idx, row in df_signals.iterrows()`
Iterates over the dataframe row by row. `idx` is the row index, `row` is a pandas Series containing all column values for that row. This is the standard pandas way to loop over rows when you need to write back to the dataframe.

`state = (row["vol_state"], row["range_state"], row["trend_state"])`
Constructs the state tuple for this signal's market conditions at the time of the signal.

`if state not in cond_epdfs: continue`
Skips signals whose state combination was too sparse to build a reliable ePDF (fewer than 30 historical observations). For these, the recommendation columns stay as `None`.

`df_signals.at[idx, "limit_price"] = result["limit_price"]`
Writes the result back into the dataframe at the specific row index. `.at[idx, col]` is the correct pandas method for writing a single value to a specific cell by row index and column name. Using `.loc` or direct assignment inside a loop can trigger warnings or silently fail.

**Result:** 19,630 out of 19,636 signals received recommendations. The 6 that didn't correspond to sparse state combinations.

---

### Cell 38 — Save output for backtesting partner

```python
df_signals.to_csv("data/Gold/GC_signals_with_recommendations.csv", index=False)
print("Saved successfully")
```

**What this does:**

Saves the complete `df_signals` dataframe — now enriched with `limit_price`, `ticks_away`, and `fill_probability` — to a CSV file. This is the handoff file to the backtesting module.

`index=False`
Prevents pandas from writing the integer row index as a column in the CSV. The row index carries no meaningful information and would just add noise to the file.

**What the backtesting partner receives:**

Each row in `GC_signals_with_recommendations.csv` contains:
- `datetime` — when the signal occurred
- `signal_direction` — buy or sell
- `signal` — the raw signal strength (magnitude indicates confidence)
- `open` — the market open price at the time of the signal
- `vol_state`, `range_state`, `trend_state` — the market regime at signal time
- `limit_price` — where to place the limit order
- `ticks_away` — how many ticks from the open the order is placed
- `fill_probability` — estimated probability the order gets filled based on historical data

**The backtester's job** is to take each row, look up what actually happened in the market data during the τ-minute window following `datetime`, and determine:
1. Was the limit order actually filled? (Did the price reach `limit_price` within τ minutes?)
2. If filled, what was the price improvement vs. just taking the market open price?
3. If not filled, what was the cost of missing the trade?

This comparison — limit order strategy vs. naive market order — is the core of the backtest.

---

## How to Extend to Other Markets

The entire pipeline is parameterized. To run on Nasdaq (NQ) instead of Gold:

```python
TICK_SIZE = 0.25  # Nasdaq tick size
files = ["data/Nasdaq/NQH24.csv", "data/Nasdaq/NQM24.csv", ...]
# Everything else is identical
```

The only things that change per market are:
1. The file paths
2. The tick size (`TICK_SIZE`)
3. The expected daily minutes (if different from 1,380)
