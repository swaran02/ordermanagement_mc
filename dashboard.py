import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from utils import (
    load_and_clean, resample_ohlcv, compute_ranges,
    compute_ewma, compute_epdfs, compute_conditional_epdfs,
    get_order_recommendation, load_signals,
    MARKETS, MARKET_FILES
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Order Management Dashboard",
    page_icon="📈",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# Caching — load and process data once per market/tau combination
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_market_data(market, tau):
    config = MARKETS[market]
    files = MARKET_FILES[market]
    tick_size = config["tick_size"]
    daily_minutes = config["daily_minutes"]

    all_dfs = [load_and_clean(f, daily_minutes=daily_minutes) for f in files]
    df_all = pd.concat(all_dfs).sort_values("datetime").reset_index(drop=True)

    daily_vol = df_all.groupby(["date", "contract"])["volume"].sum().reset_index()
    best_contract = daily_vol.loc[
        daily_vol.groupby("date")["volume"].idxmax(), ["date", "contract"]
    ]
    df_stitched = df_all.merge(best_contract, on=["date", "contract"]).reset_index(drop=True)
    df_stitched = df_stitched.drop(columns=["bars_per_day"])

    df_tau = compute_ranges(resample_ohlcv(df_stitched, tau), tick_size)

    volume_state = compute_ewma(df_tau["volume"], m=10)
    range_state = compute_ewma(df_tau["range_ticks"], m=10)
    df_tau[["ewma_volume", "ewmv_volume"]] = volume_state
    df_tau[["ewma_range", "ewmv_range"]] = range_state
    df_tau["vol_state"] = pd.qcut(df_tau["ewma_volume"], q=3, labels=[1, 2, 3])
    df_tau["range_state"] = pd.qcut(df_tau["ewma_range"], q=3, labels=[1, 2, 3])
    df_tau["delta_x"] = df_tau["open"].diff()
    df_tau["trend_state"] = pd.qcut(df_tau["delta_x"], q=3, labels=[1, 2, 3])

    cond_epdfs = compute_conditional_epdfs(df_tau)
    signals = pd.read_csv(config["signal_file"])
    signals["datetime"] = pd.to_datetime(signals["datetime"])

    return df_stitched, df_tau, cond_epdfs, signals



# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Order Management")
st.sidebar.markdown("---")

page = st.sidebar.selectbox("Navigate", [
    "Overview",
    "Range Analysis",
    "ePDFs",
    "EWMA & States",
    "Order Recommendations",
    "Backtest Results"
])

market = st.sidebar.selectbox("Market", list(MARKETS.keys()))
tau = st.sidebar.selectbox("Holding Period τ (min)", [5, 15, 30])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Tick Size:** {MARKETS[market]['tick_size']}")
st.sidebar.markdown(f"**Session:** {MARKETS[market]['daily_minutes']} min/day")

# Load data
with st.spinner(f"Loading {market} data..."):
    df_stitched, df_tau, cond_epdfs, signals = load_market_data(market, tau)

print("Dashboard structure loaded")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title(f"📊 {market} — Market Overview")
    st.markdown("---")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bars (1-min)", f"{len(df_stitched):,}")
    col2.metric("Liquid Days", f"{df_stitched['date'].nunique()}")
    col3.metric("Contracts", f"{df_stitched['contract'].nunique()}")
    col4.metric("Tick Size", f"{MARKETS[market]['tick_size']}")

    st.markdown("---")

    # Contract date ranges
    st.subheader("Contract Date Ranges")
    contract_ranges = df_stitched.groupby("contract")["date"].agg(["min", "max"]).reset_index()
    contract_ranges.columns = ["Contract", "Start Date", "End Date"]
    st.dataframe(contract_ranges, use_container_width=True)

    st.markdown("---")

    # Volume profile over time
    st.subheader("Daily Volume Profile")
    st.markdown("Shows how trading volume evolved over time — useful for identifying the liquid period and contract rolls.")

    daily_vol = df_stitched.groupby("date")["volume"].sum().reset_index()
    daily_vol["date"] = pd.to_datetime(daily_vol["date"])

    fig, ax = plt.subplots(figsize=(12, 4))
    for contract in df_stitched["contract"].unique():
        mask = df_stitched[df_stitched["contract"] == contract]["date"].unique()
        contract_vol = daily_vol[daily_vol["date"].isin(pd.to_datetime(mask))]
        ax.bar(contract_vol["date"], contract_vol["volume"], 
               label=contract, alpha=0.8, width=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Volume")
    ax.set_title(f"{market} — Daily Volume by Contract")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Signal summary
    st.subheader("Signal Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Signals", f"{len(signals):,}")
    col2.metric("Buy Signals", f"{(signals['signal_direction']=='buy').sum():,}")
    col3.metric("Sell Signals", f"{(signals['signal_direction']=='sell').sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — RANGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Range Analysis":
    st.title(f"📏 {market} — Range Analysis")
    st.markdown("---")

    # Formulas
    st.subheader("Definitions")
    st.markdown(r"""
    For a time interval $[t, t+\tau)$, three range quantities are defined:

    $$R_{t,\tau} = H - L \quad \text{(Range)}$$

    $$R^{(U)}_{t,\tau} = H - O \quad \text{(RangeUp)}$$

    $$R^{(D)}_{t,\tau} = O - L \quad \text{(RangeDn)}$$

    where $H$, $L$, $O$ are the High, Low, and Open prices of the interval.
    These satisfy the identity:
    $$R_{t,\tau} = R^{(U)}_{t,\tau} + R^{(D)}_{t,\tau}$$

    All values are expressed in **ticks** (integer multiples of tick size $\epsilon$):
    $$\ell = \frac{H - L}{\epsilon}$$
    """)

    st.markdown("---")

    # Stats
    st.subheader(f"Range Statistics at τ = {tau} min")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Range (ticks)", f"{df_tau['range_ticks'].mean():.1f}")
    col2.metric("Avg RangeUp (ticks)", f"{df_tau['range_up'].mean():.1f}")
    col3.metric("Avg RangeDn (ticks)", f"{df_tau['range_dn'].mean():.1f}")

    st.markdown("---")

    # Range distributions
    st.subheader("Range Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, col, color, label in zip(
        axes,
        ["range_ticks", "range_up", "range_dn"],
        ["steelblue", "green", "red"],
        ["Range", "RangeUp", "RangeDn"]
    ):
        counts = df_tau[col].value_counts()
        epdf = (counts / len(df_tau)).sort_index()
        xlim = int(np.percentile(df_tau[col].values, 95))
        ax.bar(epdf.index, epdf.values, color=color, alpha=0.7)
        ax.set_xlim(0, xlim)
        ax.set_title(f"{label} Distribution")
        ax.set_xlabel("Ticks")
        ax.set_ylabel("Probability")

    plt.suptitle(f"{market} — Range Distributions (τ = {tau} min)", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Comparison across tau values
    st.subheader("Range vs Holding Period τ")
    st.markdown("Average range increases with τ — wider windows capture more price movement.")

    tau_data = []
    for t in [5, 15, 30]:
        df_t = compute_ranges(resample_ohlcv(df_stitched, t), MARKETS[market]["tick_size"])
        tau_data.append({
            "τ (min)": t,
            "Avg Range": df_t["range_ticks"].mean().round(1),
            "Avg RangeUp": df_t["range_up"].mean().round(1),
            "Avg RangeDn": df_t["range_dn"].mean().round(1)
        })

    tau_df = pd.DataFrame(tau_data).set_index("τ (min)")
    st.dataframe(tau_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tau_df.index, tau_df["Avg Range"], marker="o", label="Range")
    ax.plot(tau_df.index, tau_df["Avg RangeUp"], marker="s", label="RangeUp")
    ax.plot(tau_df.index, tau_df["Avg RangeDn"], marker="^", label="RangeDn")
    ax.set_xlabel("τ (minutes)")
    ax.set_ylabel("Average Ticks")
    ax.set_title(f"{market} — Average Range vs τ")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — ePDFs
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ePDFs":
    st.title(f"📉 {market} — Empirical Probability Distributions")
    st.markdown("---")

    # Formulas
    st.subheader("Methodology")
    st.markdown(r"""
    For a given holding period $\tau$, the empirical probability distribution of RangeDn is estimated as:

    $$\hat{P}(R^{(D)}_{t,\tau} = \ell) = \frac{\text{count}(R^{(D)} = \ell)}{N} \quad \text{for } \ell = 0, 1, 2, \ldots$$

    where $N$ is the total number of observed intervals. Similarly for RangeUp and Range.

    The **survival function** (fill probability for a limit order placed $k$ ticks away) is:

    $$S(k) = P(R^{(D)}_{t,\tau} \geq k) = 1 - \sum_{\ell=0}^{k-1} \hat{P}(R^{(D)} = \ell)$$

    A limit buy order placed $k$ ticks below the open is filled if $R^{(D)} \geq k$.
    """)

    st.markdown("---")

    # Unconditional ePDFs
    st.subheader("Unconditional ePDFs")
    st.markdown("Estimated from all bars regardless of market conditions — the baseline distribution.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, color, label in zip(
        axes,
        ["range_ticks", "range_up", "range_dn"],
        ["steelblue", "green", "red"],
        ["Range", "RangeUp", "RangeDn"]
    ):
        counts = df_tau[col].value_counts()
        epdf = (counts / len(df_tau)).sort_index()
        xlim = int(np.percentile(df_tau[col].values, 95))
        ax.bar(epdf.index, epdf.values, color=color, alpha=0.7)
        ax.set_xlim(0, xlim)
        ax.set_title(f"ePDF of {label}")
        ax.set_xlabel("Ticks")
        ax.set_ylabel("Probability")

    plt.suptitle(f"{market} — Unconditional ePDFs (τ = {tau} min)", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Survival function
    st.subheader("Survival Function — Fill Probability")
    st.markdown("P(RangeDn ≥ k) — probability a buy limit order placed k ticks below open gets filled.")

    counts = df_tau["range_dn"].value_counts()
    epdf = (counts / len(df_tau)).sort_index()
    survival = 1 - epdf.cumsum()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(survival.index, survival.values, color="steelblue", linewidth=2)
    ax.axhline(y=0.70, color="red", linestyle="--", label="70% threshold")
    ax.set_xlim(0, int(np.percentile(df_tau["range_dn"].values, 95)))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Ticks below open (k)")
    ax.set_ylabel("Fill Probability S(k)")
    ax.set_title(f"{market} — Survival Function (τ = {tau} min)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Conditional ePDFs
    st.subheader("Conditional ePDFs by Market Regime")
    st.markdown(r"""
    Instead of one global distribution, we estimate separate ePDFs conditioned on the current market regime:

    $$\hat{P}(R^{(D)}_{t,\tau} = \ell \mid v_{j-1} \in m, \sigma_{j-1} \in n, \Delta x_{j-1} \in k)$$

    where $v_{j-1}$ is volume state, $\sigma_{j-1}$ is volatility state, and $\Delta x_{j-1}$ is trend state.
    """)

    col1, col2 = st.columns(2)
    with col1:
        vol = st.selectbox("Volume State", [1, 2, 3], format_func=lambda x: {1:"Low",2:"Medium",3:"High"}[x])
    with col2:
        rng = st.selectbox("Volatility State", [1, 2, 3], format_func=lambda x: {1:"Low",2:"Medium",3:"High"}[x])
    trend = st.selectbox("Trend State", [1, 2, 3], format_func=lambda x: {1:"Down",2:"Flat",3:"Up"}[x])

    state = (vol, rng, trend)

    if state in cond_epdfs:
        epdf_dn = cond_epdfs[state]["range_dn"]
        epdf_up = cond_epdfs[state]["range_up"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].bar(epdf_dn.index, epdf_dn.values, color="red", alpha=0.7)
        axes[0].set_xlim(0, int(np.percentile(df_tau["range_dn"].values, 95)))
        axes[0].set_title(f"RangeDn — State {state}")
        axes[0].set_xlabel("Ticks below open")
        axes[0].set_ylabel("Probability")

        axes[1].bar(epdf_up.index, epdf_up.values, color="green", alpha=0.7)
        axes[1].set_xlim(0, int(np.percentile(df_tau["range_up"].values, 95)))
        axes[1].set_title(f"RangeUp — State {state}")
        axes[1].set_xlabel("Ticks above open")
        axes[1].set_ylabel("Probability")

        mean_dn = np.sum(epdf_dn.index.astype(float) * epdf_dn.values).round(2)
        mean_up = np.sum(epdf_up.index.astype(float) * epdf_up.values).round(2)

        plt.suptitle(f"{market} — Conditional ePDFs (τ = {tau} min)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col1, col2 = st.columns(2)
        col1.metric("Mean RangeDn (ticks)", mean_dn)
        col2.metric("Mean RangeUp (ticks)", mean_up)
    else:
        st.warning(f"State {state} has insufficient data (< 30 observations). Try a different combination.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — EWMA & STATES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "EWMA & States":
    st.title(f"📡 {market} — EWMA & State Classification")
    st.markdown("---")

    # Formula
    st.subheader("Algorithm 1 — Exponentially Weighted Moving Average")
    st.markdown(r"""
    At each step $j$, the EWMA and EWMV are updated recursively using only past data $\eta_{j-1}$:

    $$\lambda = 2^{-1/m} \quad \text{(decay factor, half-life } m \text{)}$$

    $$\text{sumW}_j = \lambda \cdot \text{sumW}_{j-1} + 1$$

    $$\text{sumWX}_j = \lambda \cdot \text{sumWX}_{j-1} + \eta_{j-1}$$

    $$\text{EWMA}_j = \frac{\text{sumWX}_j}{\text{sumW}_j}$$

    $$\text{sumWSS}_j = \lambda \cdot \text{sumWSS}_{j-1} + (\eta_{j-1} - \text{EWMA}_j)^2$$

    $$\text{EWMV}_j = \sqrt{\frac{\text{sumWSS}_j}{\text{sumW}_j}}$$

    Using $\eta_{j-1}$ instead of $\eta_j$ is critical — it ensures no forward-looking bias. 
    The half-life $m=10$ bars means data from 10 bars ago has half the weight of current data.
    """)

    st.markdown("---")

    # EWMA over time
    st.subheader("EWMA of Volume & Range Over Time")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(df_tau["datetime"], df_tau["ewma_volume"], color="steelblue", linewidth=1)
    axes[0].fill_between(df_tau["datetime"], df_tau["ewma_volume"], alpha=0.2, color="steelblue")
    axes[0].set_ylabel("EWMA Volume")
    axes[0].set_title(f"{market} — EWMA Volume (τ = {tau} min, m=10)")

    axes[1].plot(df_tau["datetime"], df_tau["ewma_range"], color="orange", linewidth=1)
    axes[1].fill_between(df_tau["datetime"], df_tau["ewma_range"], alpha=0.2, color="orange")
    axes[1].set_ylabel("EWMA Range (ticks)")
    axes[1].set_title(f"{market} — EWMA Range (τ = {tau} min, m=10)")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # State distributions
    st.subheader("State Distribution")
    st.markdown("Each dimension is binned into 3 equal-frequency states using percentile boundaries.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col, title, colors in zip(
        axes,
        ["vol_state", "range_state", "trend_state"],
        ["Volume State", "Volatility State", "Trend State"],
        [["#a8d8ea", "#57b8d4", "#1a7a99"],
         ["#b8f0b8", "#5cb85c", "#2d7a2d"],
         ["#ffcccb", "#ff6b6b", "#cc0000"]]
    ):
        counts = df_tau[col].value_counts().sort_index()
        labels = {
            "vol_state": {1:"Low", 2:"Medium", 3:"High"},
            "range_state": {1:"Low", 2:"Medium", 3:"High"},
            "trend_state": {1:"Down", 2:"Flat", 3:"Up"}
        }[col]
        ax.bar([labels[i] for i in counts.index], counts.values, color=colors, alpha=0.85)
        ax.set_title(title)
        ax.set_ylabel("Count")

    plt.suptitle(f"{market} — State Distributions (τ = {tau} min)", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Valid state combinations
    st.subheader("State Combination Coverage")
    state_counts = df_tau.groupby(
        ["vol_state", "range_state", "trend_state"]
    ).size().reset_index(name="count")
    state_counts["valid"] = state_counts["count"] >= 30
    state_counts["vol_label"] = state_counts["vol_state"].map({1:"Low", 2:"Med", 3:"High"})
    state_counts["range_label"] = state_counts["range_state"].map({1:"Low", 2:"Med", 3:"High"})
    state_counts["trend_label"] = state_counts["trend_state"].map({1:"Down", 2:"Flat", 3:"Up"})
    state_counts["State"] = (
    state_counts["vol_label"].astype(str) + " Vol / " + 
    state_counts["range_label"].astype(str) + " Range / " + 
    state_counts["trend_label"].astype(str) + " Trend"
)
    valid_count = state_counts["valid"].sum()
    st.metric("Valid State Combinations", f"{valid_count} / 27")
    st.dataframe(state_counts[["State", "count", "valid"]].rename(
        columns={"count": "Observations", "valid": "Valid (≥30)"}
    ), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — ORDER RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Order Recommendations":
    st.title(f"🎯 {market} — Order Recommendations")
    st.markdown("---")

    # Methodology
    st.subheader("Methodology")
    st.markdown(r"""
    Given a signal at time $t$ with open price $O$, the limit order is placed at:

    $$\text{limit\_price} = \begin{cases} O - k \cdot \epsilon & \text{(Buy signal)} \\ O + k \cdot \epsilon & \text{(Sell signal)} \end{cases}$$

    where $k$ is the optimal number of ticks away, chosen as the largest $k$ satisfying:

    $$S(k) = P(R^{(D)}_{t,\tau} \geq k) \geq \text{target probability (default 70\%)}$$

    This maximizes price improvement while maintaining the target fill probability.
    """)

    st.markdown("---")

    # Interactive recommendation tool
    st.subheader("Live Recommendation Tool")
    st.markdown("Enter a signal and market state to get a limit order recommendation.")

    col1, col2 = st.columns(2)
    with col1:
        signal_dir = st.selectbox("Signal Direction", ["buy", "sell"])
        open_price = st.number_input("Open Price", value=float(df_tau["open"].iloc[-1]))
        target_prob = st.slider("Target Fill Probability", 0.50, 0.95, 0.70, 0.05)
    with col2:
        vol_s = st.selectbox("Volume State", [1, 2, 3], format_func=lambda x: {1:"Low",2:"Medium",3:"High"}[x])
        range_s = st.selectbox("Volatility State", [1, 2, 3], format_func=lambda x: {1:"Low",2:"Medium",3:"High"}[x])
        trend_s = st.selectbox("Trend State", [1, 2, 3], format_func=lambda x: {1:"Down",2:"Flat",3:"Up"}[x])

    state = (vol_s, range_s, trend_s)

    if state in cond_epdfs:
        result = get_order_recommendation(
            signal_dir, open_price, state, cond_epdfs,
            tick_size=MARKETS[market]["tick_size"],
            target_prob=target_prob
        )
        if result:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Limit Price", f"{result['limit_price']}")
            col2.metric("Ticks Away", f"{result['ticks_away']}")
            col3.metric("Fill Probability", f"{result['fill_probability']:.1%}")

            # Survival function plot
            epdf_key = "range_dn" if signal_dir == "buy" else "range_up"
            epdf = cond_epdfs[state][epdf_key]
            survival = 1 - epdf.sort_index().cumsum()

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(survival.index, survival.values, color="steelblue", linewidth=2)
            ax.axhline(y=target_prob, color="red", linestyle="--", 
                      label=f"Target: {target_prob:.0%}")
            ax.axvline(x=result['ticks_away'], color="green", linestyle="--",
                      label=f"Optimal k = {result['ticks_away']}")
            ax.set_xlim(0, int(np.percentile(df_tau[epdf_key].values, 95)))
            ax.set_ylim(0, 1)
            ax.set_xlabel("Ticks away from open (k)")
            ax.set_ylabel("Fill Probability S(k)")
            ax.set_title(f"{market} — Survival Function | State {state}")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.warning(f"State {state} has insufficient data. Try a different combination.")

    st.markdown("---")

    # Signal summary table
    st.subheader("Signal Recommendations Summary")
    summary = signals.groupby("signal_direction").agg(
        Count=("limit_price", "count"),
        Avg_Ticks_Away=("ticks_away", "mean"),
        Avg_Fill_Prob=("fill_probability", "mean")
    ).round(3)
    st.dataframe(summary, use_container_width=True)

    st.markdown("---")

    # Sample recommendations
    st.subheader("Sample Recommendations")
    sample = signals[signals["limit_price"].notna()].head(20)[[
        "datetime", "signal_direction", "open", 
        "limit_price", "ticks_away", "fill_probability",
        "vol_state_label", "range_state_label", "trend_state_label"
    ]]
    st.dataframe(sample, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — BACKTEST RESULTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Backtest Results":
    st.title("📈 Backtest Results — All Markets")
    st.markdown("---")

    # Methodology
    st.subheader("Methodology")
    st.markdown(r"""
    Each recommended limit order is evaluated against actual 5-minute market data:

    - **Buy order filled** if: $L \leq \text{limit\_price}$ within the $\tau$-minute window
    - **Sell order filled** if: $H \geq \text{limit\_price}$ within the $\tau$-minute window
    - **Missed orders** fall back to the close price of the interval
    - **Benchmark** = signal open price (naive market order)
    - **Improvement** = benchmark price − execution price (buy) or execution price − benchmark price (sell)
    - **Improvement in ticks** = improvement / tick_size (comparable across markets)
    """)

    st.markdown("---")

    # Load all results
    backtest_configs = {
    "Gold":         {"dir": "backtest_outputs",    "prefix": "GC"},
    "EuroStoxx":    {"dir": "data/EuroStoxx/backtest_outputs", "prefix": "EuroStoxx"},
    "GBP":          {"dir": "data/GBP - British Pound/backtest_outputs", "prefix": "GBP"},
    "German Bunds": {"dir": "data/German Bunds - German Government Bonds/backtest_outputs", "prefix": "Bunds"},
    "Heating Oil":  {"dir": "data/HeatingOil/backtest_outputs", "prefix": "HeatingOil"},
    "JPY":          {"dir": "data/JPY - Japanese Yen/backtest_outputs", "prefix": "JPY"},
    "Nasdaq":       {"dir": "data/Nasdaq/backtest_outputs", "prefix": "Nasdaq"},
}

    # Cross market summary
    st.subheader("Cross-Market Summary")
    summary_rows = []
    for mkt, cfg in backtest_configs.items():
        try:
            overall = pd.read_csv(f"{cfg['dir']}/overall.csv")
            metric_map = dict(zip(overall["metric"], overall["value"]))
            summary_rows.append({
                "Market": mkt,
                "Signals": int(metric_map["num_signals"]),
                "Fill Rate": f"{metric_map['fill_rate']:.1%}",
                "Avg Improvement": round(metric_map["avg_improvement"], 4),
                "Positive Improvement Rate": f"{metric_map['positive_improvement_rate']:.1%}"
            })
        except:
            pass

    summary_df = pd.DataFrame(summary_rows).set_index("Market")
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("---")

    # Fill rate comparison chart
    st.subheader("Fill Rate by Market")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    markets_list = [r["Market"] for r in summary_rows]
    fill_rates = [float(r["Fill Rate"].strip("%")) / 100 for r in summary_rows]
    pos_imp_rates = [float(r["Positive Improvement Rate"].strip("%")) / 100 for r in summary_rows]

    colors = ["steelblue" if f >= 0.70 else "red" for f in fill_rates]
    axes[0].bar(markets_list, fill_rates, color=colors, alpha=0.8)
    axes[0].axhline(y=0.70, color="red", linestyle="--", label="70% target")
    axes[0].set_title("Fill Rate by Market")
    axes[0].set_ylabel("Fill Rate")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticklabels(markets_list, rotation=45, ha="right")
    axes[0].legend()

    axes[1].bar(markets_list, pos_imp_rates, color="orange", alpha=0.8)
    axes[1].set_title("Positive Improvement Rate by Market")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticklabels(markets_list, rotation=45, ha="right")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Per market deep dive
    selected_market = market  # use sidebar selection
    cfg = backtest_configs[market]

    try:
        overall = pd.read_csv(f"{cfg['dir']}/overall.csv")
        by_direction = pd.read_csv(f"{cfg['dir']}/by_direction.csv")
        by_ticks = pd.read_csv(f"{cfg['dir']}/by_ticks.csv")
        by_vol = pd.read_csv(f"{cfg['dir']}/by_vol_state.csv")
        by_range = pd.read_csv(f"{cfg['dir']}/by_range_state.csv")
        by_trend = pd.read_csv(f"{cfg['dir']}/by_trend_state.csv")

        metric_map = dict(zip(overall["metric"], overall["value"]))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Signals", f"{int(metric_map['num_signals']):,}")
        col2.metric("Fill Rate", f"{metric_map['fill_rate']:.1%}")
        col3.metric("Avg Improvement", f"{metric_map['avg_improvement']:.4f}")
        col4.metric("Positive Improvement Rate", f"{metric_map['positive_improvement_rate']:.1%}")

        st.markdown("---")

        # By direction
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(by_direction["signal_direction"], by_direction["fill_rate"],
                    color=["steelblue", "orange"], alpha=0.8)
        axes[0].set_title("Fill Rate by Direction")
        axes[0].set_ylabel("Fill Rate")
        axes[0].set_ylim(0, 1)
        axes[0].axhline(y=0.70, color="red", linestyle="--")

        axes[1].bar(by_direction["signal_direction"], by_direction["avg_improvement"],
                    color=["steelblue", "orange"], alpha=0.8)
        axes[1].set_title("Avg Improvement by Direction")
        axes[1].set_ylabel("Avg Improvement")
        plt.suptitle(f"{selected_market} — Performance by Direction", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # By regime
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        state_labels = {
            "vol_state": {1:"Low", 2:"Med", 3:"High"},
            "range_state": {1:"Low", 2:"Med", 3:"High"},
            "trend_state": {1:"Down", 2:"Flat", 3:"Up"}
        }

        for i, (df, col, title) in enumerate([
            (by_vol, "vol_state", "Volume State"),
            (by_range, "range_state", "Range State"),
            (by_trend, "trend_state", "Trend State")
        ]):
            labels = [state_labels[col][x] for x in df[col]]
            axes[0][i].bar(labels, df["fill_rate"], color="steelblue", alpha=0.8)
            axes[0][i].set_title(f"Fill Rate by {title}")
            axes[0][i].set_ylim(0, 1)
            axes[0][i].axhline(y=0.70, color="red", linestyle="--")

            axes[1][i].bar(labels, df["avg_improvement"], color="orange", alpha=0.8)
            axes[1][i].set_title(f"Avg Improvement by {title}")

        plt.suptitle(f"{selected_market} — Performance by Regime", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # PnL Analysis
        st.markdown("---")
        st.subheader("Simulated PnL Analysis")
        st.markdown("PnL per trade = improvement in ticks. Cumulative PnL shows strategy performance over time.")

        try:
            results = pd.read_csv(f"{cfg['dir']}/{cfg['prefix']}_backtest_results.csv")
            results["datetime"] = pd.to_datetime(results["datetime"])
            tick_size_mkt = MARKETS[selected_market]["tick_size"]
            results["improvement_ticks"] = results["improvement"] / tick_size_mkt
            results = results.sort_values("datetime")
            results["cumulative_pnl"] = results["improvement_ticks"].cumsum()

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Cumulative PnL
            axes[0].plot(results["datetime"], results["cumulative_pnl"],
                        color="steelblue", linewidth=1.5)
            axes[0].axhline(y=0, color="red", linestyle="--")
            axes[0].fill_between(results["datetime"], results["cumulative_pnl"],
                                where=results["cumulative_pnl"] >= 0,
                                color="green", alpha=0.2, label="Positive")
            axes[0].fill_between(results["datetime"], results["cumulative_pnl"],
                                where=results["cumulative_pnl"] < 0,
                                color="red", alpha=0.2, label="Negative")
            axes[0].set_title(f"{selected_market} — Cumulative PnL (ticks)")
            axes[0].set_ylabel("Cumulative Ticks")
            axes[0].legend()

            # Rolling average improvement
            results["rolling_improvement"] = results["improvement_ticks"].rolling(50).mean()
            axes[1].plot(results["datetime"], results["rolling_improvement"],
                        color="orange", linewidth=1.5)
            axes[1].axhline(y=0, color="red", linestyle="--")
            axes[1].set_title(f"{selected_market} — Rolling Avg Improvement (50 trades)")
            axes[1].set_ylabel("Avg Ticks")
            axes[1].set_xlabel("Date")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # PnL summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total PnL (ticks)", f"{results['improvement_ticks'].sum():.1f}")
            col2.metric("Avg PnL per Trade (ticks)", f"{results['improvement_ticks'].mean():.3f}")
            col3.metric("Best Trade (ticks)", f"{results['improvement_ticks'].max():.1f}")
            col4.metric("Worst Trade (ticks)", f"{results['improvement_ticks'].min():.1f}")

        except Exception as e:
            st.warning(f"Could not load detailed results: {e}")

    except Exception as e:
        st.error(f"Could not load results for {selected_market}: {e}")