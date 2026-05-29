
MARKET_FILES = {
    "Gold":         ["data/Gold/GCG24.csv", "data/Gold/GCJ24.csv", "data/Gold/GCM24.csv", "data/Gold/GCQ24.csv"],
    "EuroStoxx":    ["data/EuroStoxx/VGH22.csv", "data/EuroStoxx/VGM22.csv"],
    "GBP":          ["data/GBP - British Pound/BPM20.csv", "data/GBP - British Pound/BPU20.csv", "data/GBP - British Pound/BPZ20.csv"],
    "German Bunds": ["data/German Bunds - German Government Bonds/RXM25.csv", "data/German Bunds - German Government Bonds/RXU25.csv", "data/German Bunds - German Government Bonds/RXZ25.csv"],
    "Heating Oil":  ["data/HeatingOil/HOF22.csv", "data/HeatingOil/HOG22.csv", "data/HeatingOil/HOH22.csv", "data/HeatingOil/HOJ22.csv", "data/HeatingOil/HOK22.csv", "data/HeatingOil/HOM22.csv", "data/HeatingOil/HON22.csv"],
    "JPY":          ["data/JPY - Japanese Yen/JYU24.csv", "data/JPY - Japanese Yen/JYZ24.csv", "data/JPY - Japanese Yen/JYH25.csv"],
    "Nasdaq":       ["data/Nasdaq/NQH20.csv", "data/Nasdaq/NQM20.csv", "data/Nasdaq/NQU20.csv"],
}

MARKETS = {
    "Gold":         {"tick_size": 0.10,  "daily_minutes": 1380, "signal_file": "data/Gold/GC_signals_with_recommendations.csv"},
    "EuroStoxx":    {"tick_size": 0.50,  "daily_minutes": 840,  "signal_file": "data/EuroStoxx/ES_signals_with_recommendations.csv"},
    "GBP":          {"tick_size": 0.01,  "daily_minutes": 1380, "signal_file": "data/GBP - British Pound/GBP_signals_with_recommendations.csv"},
    "German Bunds": {"tick_size": 0.01,  "daily_minutes": 840,  "signal_file": "data/German Bunds - German Government Bonds/Bunds_signals_with_recommendations.csv"},
    "Heating Oil":  {"tick_size": 0.01,  "daily_minutes": 900,  "signal_file": "data/HeatingOil/HeatingOil_signals_with_recommendations.csv"},
    "JPY":          {"tick_size": 0.005, "daily_minutes": 1380, "signal_file": "data/JPY - Japanese Yen/JPY_signals_with_recommendations.csv"},
    "Nasdaq":       {"tick_size": 0.25,  "daily_minutes": 1380, "signal_file": "data/Nasdaq/Nasdaq_signals_with_recommendations.csv"},
}


import pandas as pd

def load_and_clean(filepath, daily_minutes=1380):
    df = pd.read_csv(filepath, header=None,
                     names=["datetime", "open", "high", "low", "close", "volume"])
    
    df["contract"] = filepath.split("/")[-1].replace(".csv", "")
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

def resample_ohlcv(df, tau):
    
    # Make a copy so original dataframe is untouched
    temp = df.copy()

    # Set datetime as index
    temp = temp.set_index("datetime")

    # Resample OHLCV
    df_resampled = temp.resample(f"{tau}min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    # Drop incomplete/empty intervals
    df_resampled = df_resampled.dropna()

    # Bring datetime back as a column
    df_resampled = df_resampled.reset_index()

    return df_resampled

def compute_ranges(df, tick_size):
    df = df.copy()
    df["range_ticks"] = ((df["high"] - df["low"]) / tick_size).round().astype(int)
    df["range_up"] = ((df["high"] - df["open"]) / tick_size).round().astype(int)
    df["range_dn"] = ((df["open"] - df["low"]) / tick_size).round().astype(int)
    return df

def compute_epdfs(df):
    epdfs = {}
    for col in ["range_ticks", "range_up", "range_dn"]:
        counts = df[col].value_counts()
        epdfs[col] = (counts / len(df)).sort_index()
    return epdfs

import numpy as np

def compute_ewma(series, m):
    values = series.astype(float).to_numpy()
    
    lambda_ = 2 ** (-1 / m)

    ewma = np.zeros(len(values))
    ewmv = np.zeros(len(values))

    sumW = 0.0
    sumWX = 0.0
    sumWSS = 0.0

    # First row has no previous observation to use
    ewma[0] = np.nan
    ewmv[0] = np.nan

    for j in range(1, len(values)):
        # Use eta_{j-1}, not eta_j
        prev_x = values[j - 1]

        sumW = lambda_ * sumW + 1
        sumWX = lambda_ * sumWX + prev_x

        ewma[j] = sumWX / sumW

        sumWSS = lambda_ * sumWSS + (prev_x - ewma[j]) ** 2
        ewmv[j] = np.sqrt(sumWSS / sumW)

    return pd.DataFrame({
        "ewma": ewma,
        "ewmv": ewmv
    }, index=series.index)

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


