import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

st.set_page_config(page_title="AVP 2σ VWAP Backtester", layout="wide")

st.title("AVP 2σ VWAP Reversal Backtester")
st.caption("Anchored VWAP + developing volume profile + 2σ rejection strategy")

st.markdown("""
Upload OHLCV data exported from TradingView. The app recreates the original signal logic:
- Long: candle touches/breaches VWAP −2σ, closes back above it, closes bullish, and finishes below developing VAL.
- Short: candle touches/breaches VWAP +2σ, closes back below it, closes bearish, and finishes above developing VAH.

The profile is an approximation: each candle's full volume is assigned to the bin containing its HLC3, matching the Pine script.
""")

with st.sidebar:
    st.header("Strategy settings")
    anchor = st.selectbox("Profile/VWAP anchor", ["Day", "Week", "Month"], index=0)
    row_atr_mult = st.number_input("Profile row size (ATR multiple)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    va_pct = st.slider("Value area (%)", min_value=50, max_value=99, value=70)
    st.subheader("Execution")
    qty = st.number_input("Contracts per trade", min_value=1, max_value=100, value=1, step=1)
    point_value = st.number_input("Point value ($ per point per contract)", min_value=0.01, value=2.0, step=0.25,
                                  help="MNQ = $2 per index point; NQ = $20 per index point.")
    stop_mode = st.selectbox("Stop method", ["ATR", "Signal candle"])
    atr_mult = st.number_input("ATR stop multiplier", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
    rr = st.number_input("Reward:risk target (R)", min_value=0.25, max_value=10.0, value=2.0, step=0.25)
    commission = st.number_input("Commission ($ per contract per side)", min_value=0.0, value=0.0, step=0.1)
    slippage_ticks = st.number_input("Slippage (ticks per fill)", min_value=0.0, value=1.0, step=0.5)
    tick_size = st.number_input("Tick size", min_value=0.000001, value=0.25, step=0.01)
    st.subheader("Backtest")
    start_date = st.date_input("Start date", value=date(2025, 1, 1))
    end_date = st.date_input("End date", value=date(2027, 12, 31))
    close_at_end = st.checkbox("Close open position at end of test", value=True)

uploaded = st.file_uploader("Upload TradingView OHLCV CSV", type=["csv"])

with st.expander("How to export data from TradingView"):
    st.markdown("""
1. Open the desired symbol and timeframe (for example, **MNQ1! · 1 minute**).
2. Load the historical range you want.
3. Use the chart menu → **Export chart data**.
4. Include volume if TradingView offers that option, then upload the CSV here.

Expected columns: `time`/`timestamp`/`date`, `open`, `high`, `low`, `close`, and `volume`.
The app accepts common TradingView column names such as `time`, `Volume`, and `#`.
""")

def find_col(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None

def load_data(file):
    raw = pd.read_csv(file)
    raw.columns = [str(c).strip() for c in raw.columns]
    tcol = find_col(raw.columns, ["time", "timestamp", "date", "datetime"])
    ocol = find_col(raw.columns, ["open", "o"])
    hcol = find_col(raw.columns, ["high", "h"])
    lcol = find_col(raw.columns, ["low", "l"])
    ccol = find_col(raw.columns, ["close", "c"])
    vcol = find_col(raw.columns, ["volume", "vol", "v"])
    missing = [n for n, c in [("time", tcol), ("open", ocol), ("high", hcol),
                              ("low", lcol), ("close", ccol), ("volume", vcol)] if c is None]
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))
    df = raw[[tcol, ocol, hcol, lcol, ccol, vcol]].copy()
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"]).copy()
    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid OHLCV rows found.")
    return df

def anchor_key(ts, mode):
    if mode == "Day":
        return ts.normalize()
    if mode == "Week":
        return ts.to_period("W").start_time
    return ts.to_period("M").start_time

def calculate_indicators(df, anchor_mode, row_mult, va_percent):
    d = df.copy()
    # Wilder ATR (RMA), consistent with ta.atr(14) more closely than a simple rolling mean.
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False, min_periods=1).mean()
    d["hlc3"] = (d["high"] + d["low"] + d["close"]) / 3.0
    d["anchor"] = d["time"].map(lambda x: anchor_key(x, anchor_mode))

    n = len(d)
    cols = {k: np.full(n, np.nan) for k in
            ["vwap", "sd", "upper2", "lower2", "poc", "vah", "val", "row_size"]}
    profile = {}
    current_anchor = None
    cum_v = cum_pv = cum_p2v = 0.0
    first_bin = poc_bin = 0
    bins = []

    for i, row in d.iterrows():
        akey = row["anchor"]
        src = float(row["hlc3"])
        vol = float(row["volume"])
        if current_anchor is None or akey != current_anchor:
            current_anchor = akey
            cum_v = cum_pv = cum_p2v = 0.0
            rs = max(float(row["atr"]) * row_mult if pd.notna(row["atr"]) else abs(float(row["close"])) * 0.001 * row_mult,
                     1e-12)
            first_bin = int(np.floor(src / rs))
            poc_bin = first_bin
            bins = [0.0]
        rs = max(float(row["atr"]) * row_mult if pd.notna(row["atr"]) else 1e-12, 1e-12) if i == 0 else rs
        # Pine fixes rowSize at the anchor's first bar. Keep it fixed until next anchor.
        cum_v += vol
        cum_pv += src * vol
        cum_p2v += src * src * vol
        vwap = cum_pv / cum_v if cum_v > 0 else np.nan
        variance = max(cum_p2v / cum_v - vwap * vwap, 0.0) if cum_v > 0 else np.nan
        sd = np.sqrt(variance) if pd.notna(variance) else np.nan

        current_bin = int(np.floor(src / rs))
        if current_bin < first_bin:
            prepend = first_bin - current_bin
            bins = [0.0] * prepend + bins
            first_bin = current_bin
        if current_bin >= first_bin + len(bins):
            bins.extend([0.0] * (current_bin - (first_bin + len(bins)) + 1))
        idx = current_bin - first_bin
        bins[idx] += vol
        # Pine POC changes only when the just-updated bin exceeds the previous max.
        if bins[idx] > max(bins) - 1e-15:
            # Resolve ties as Pine does: only a strictly larger updated bin changes POC.
            # max_bin is tracked separately below.
            pass
        if i == 0 or akey != d.at[i-1, "anchor"]:
            max_bin_vol = 0.0
        if bins[idx] > max_bin_vol:
            max_bin_vol = bins[idx]
            poc_bin = current_bin

        total = float(sum(bins))
        low_idx = poc_bin - first_bin
        high_idx = low_idx
        va_vol = bins[low_idx]
        target = total * va_percent / 100.0
        while va_vol < target and (low_idx > 0 or high_idx < len(bins)-1):
            low_candidate = bins[low_idx-1] if low_idx > 0 else -1.0
            high_candidate = bins[high_idx+1] if high_idx < len(bins)-1 else -1.0
            if high_candidate > low_candidate:
                high_idx += 1
                va_vol += high_candidate
            else:
                low_idx -= 1
                va_vol += low_candidate

        cols["vwap"][i] = vwap
        cols["sd"][i] = sd
        cols["upper2"][i] = vwap + 2 * sd if pd.notna(sd) else np.nan
        cols["lower2"][i] = vwap - 2 * sd if pd.notna(sd) else np.nan
        cols["poc"][i] = (poc_bin + 0.5) * rs
        cols["val"][i] = (first_bin + low_idx) * rs
        cols["vah"][i] = (first_bin + high_idx + 1) * rs
        cols["row_size"][i] = rs

    for k, v in cols.items():
        d[k] = v
    return d

def run_backtest(d, start, end, stop_mode, atr_mult, rr, qty, point_value,
                 commission, slippage_ticks, tick_size, close_at_end):
    d = d.copy()
    d["bull_reject"] = (
        (d["low"] <= d["lower2"]) &
        (d["close"] > d["lower2"]) &
        (d["close"] > d["open"])
    )
    d["bear_reject"] = (
        (d["high"] >= d["upper2"]) &
        (d["close"] < d["upper2"]) &
        (d["close"] < d["open"])
    )
    d["long_signal"] = d["bull_reject"] & (d["close"] < d["val"])
    d["short_signal"] = d["bear_reject"] & (d["close"] > d["vah"])

    # CSV timestamps are normalized to UTC in load_data(). Make date-picker
    # boundaries UTC-aware too, avoiding aware-vs-naive comparison errors.
    start_ts = pd.Timestamp(start).tz_localize("UTC")
    end_ts = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize("UTC")
    in_range = (d["time"] >= start_ts) & (d["time"] < end_ts)
    slip = slippage_ticks * tick_size

    trades = []
    position = None
    equity = 0.0
    equity_curve = []
    for i in range(len(d)):
        r = d.iloc[i]
        # A position opened at the prior bar's close is first eligible to exit on this bar.
        if position is not None:
            exit_price = None
            reason = None
            o, h, l = float(r.open), float(r.high), float(r.low)
            if position["side"] == "Long":
                if o <= position["stop"]:
                    exit_price, reason = o - slip, "Stop (gap)"
                elif o >= position["target"]:
                    exit_price, reason = position["target"] - slip, "Target (gap)"
                else:
                    hit_stop = l <= position["stop"]
                    hit_target = h >= position["target"]
                    if hit_stop:  # Conservative when both levels trade in one candle.
                        exit_price, reason = position["stop"] - slip, "Stop"
                    elif hit_target:
                        exit_price, reason = position["target"] - slip, "Target"
            else:
                if o >= position["stop"]:
                    exit_price, reason = o + slip, "Stop (gap)"
                elif o <= position["target"]:
                    exit_price, reason = position["target"] + slip, "Target (gap)"
                else:
                    hit_stop = h >= position["stop"]
                    hit_target = l <= position["target"]
                    if hit_stop:
                        exit_price, reason = position["stop"] + slip, "Stop"
                    elif hit_target:
                        exit_price, reason = position["target"] + slip, "Target"
            if exit_price is not None:
                direction = 1 if position["side"] == "Long" else -1
                gross = (exit_price - position["entry"]) * direction * point_value * qty
                fees = commission * qty * 2
                net = gross - fees
                equity += net
                trades.append({
                    "Entry time": position["entry_time"], "Exit time": r.time,
                    "Side": position["side"], "Entry": position["entry"],
                    "Exit": exit_price, "Stop": position["stop"],
                    "Target": position["target"], "Exit reason": reason,
                    "Contracts": qty, "Gross P&L": gross,
                    "Fees": fees, "Net P&L": net,
                    "R multiple": net / position["risk_dollars"] if position["risk_dollars"] else np.nan,
                    "Bars held": i - position["entry_i"]
                })
                position = None

        # Entry signals are evaluated at candle close; no same-candle exit is allowed.
        if position is None and bool(in_range.iloc[i]):
            side = None
            if bool(r.long_signal):
                side = "Long"
            elif bool(r.short_signal):
                side = "Short"
            if side is not None and pd.notna(r.atr):
                entry = float(r.close) + (slip if side == "Long" else -slip)
                if stop_mode == "ATR":
                    risk_points = float(r.atr) * atr_mult
                else:
                    risk_points = float(r.close - r.low) if side == "Long" else float(r.high - r.close)
                if risk_points > 0:
                    if side == "Long":
                        stop = entry - risk_points
                        target = entry + risk_points * rr
                    else:
                        stop = entry + risk_points
                        target = entry - risk_points * rr
                    risk_dollars = risk_points * point_value * qty
                    position = {
                        "side": side, "entry": entry, "entry_time": r.time,
                        "entry_i": i, "stop": stop, "target": target,
                        "risk_dollars": risk_dollars
                    }
        equity_curve.append({"time": r.time, "Equity P&L": equity})

    if position is not None and close_at_end and len(d):
        r = d.iloc[-1]
        exit_price = float(r.close) - slip if position["side"] == "Long" else float(r.close) + slip
        direction = 1 if position["side"] == "Long" else -1
        gross = (exit_price - position["entry"]) * direction * point_value * qty
        fees = commission * qty * 2
        net = gross - fees
        equity += net
        trades.append({
            "Entry time": position["entry_time"], "Exit time": r.time,
            "Side": position["side"], "Entry": position["entry"],
            "Exit": exit_price, "Stop": position["stop"],
            "Target": position["target"], "Exit reason": "End of test",
            "Contracts": qty, "Gross P&L": gross, "Fees": fees,
            "Net P&L": net,
            "R multiple": net / position["risk_dollars"] if position["risk_dollars"] else np.nan,
            "Bars held": len(d)-1-position["entry_i"]
        })
        if equity_curve:
            equity_curve[-1]["Equity P&L"] = equity

    return d, pd.DataFrame(trades), pd.DataFrame(equity_curve)

if uploaded is not None:
    try:
        data = load_data(uploaded)
        data = calculate_indicators(data, anchor, row_atr_mult, va_pct)
        tested, trades, curve = run_backtest(
            data, start_date, end_date, stop_mode, atr_mult, rr, int(qty),
            point_value, commission, slippage_ticks, tick_size, close_at_end
        )
        st.success(f"Loaded {len(data):,} bars from {data.time.min()} to {data.time.max()}.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Long signals", f"{int(tested.long_signal.sum()):,}")
        c2.metric("Short signals", f"{int(tested.short_signal.sum()):,}")
        c3.metric("Completed trades", f"{len(trades):,}")

        if len(trades):
            wins = trades[trades["Net P&L"] > 0]
            losses = trades[trades["Net P&L"] < 0]
            gross_profit = wins["Net P&L"].sum()
            gross_loss = abs(losses["Net P&L"].sum())
            pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
            win_rate = len(wins) / len(trades) * 100
            avg_win = wins["Net P&L"].mean() if len(wins) else 0.0
            avg_loss = losses["Net P&L"].mean() if len(losses) else 0.0
            expectancy = trades["Net P&L"].mean()
            eq = curve["Equity P&L"]
            dd = eq - eq.cummax()
            max_dd = abs(dd.min()) if len(dd) else 0.0
            total_net = trades["Net P&L"].sum()
            kpis = st.columns(5)
            kpis[0].metric("Net P&L", f"${total_net:,.2f}")
            kpis[1].metric("Win rate", f"{win_rate:.1f}%")
            kpis[2].metric("Profit factor", "∞" if np.isinf(pf) else f"{pf:.2f}")
            kpis[3].metric("Expectancy / trade", f"${expectancy:,.2f}")
            kpis[4].metric("Max drawdown", f"${max_dd:,.2f}")

            kpis2 = st.columns(4)
            kpis2[0].metric("Winning trades", len(wins))
            kpis2[1].metric("Losing trades", len(losses))
            kpis2[2].metric("Average winner", f"${avg_win:,.2f}")
            kpis2[3].metric("Average loser", f"${avg_loss:,.2f}")

            st.subheader("Equity curve")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=curve["time"], y=curve["Equity P&L"], mode="lines", name="Cumulative P&L"))
            fig_eq.update_layout(height=320, xaxis_title="Time", yaxis_title="P&L ($)", hovermode="x unified")
            st.plotly_chart(fig_eq, use_container_width=True)

            st.subheader("Performance breakdown")
            left, right = st.columns(2)
            with left:
                by_side = trades.groupby("Side")["Net P&L"].agg(["count", "sum", "mean"])
                st.write("By direction")
                st.dataframe(by_side, use_container_width=True)
            with right:
                by_reason = trades.groupby("Exit reason")["Net P&L"].agg(["count", "sum", "mean"])
                st.write("By exit reason")
                st.dataframe(by_reason, use_container_width=True)

            trades["Entry hour"] = pd.to_datetime(trades["Entry time"]).dt.hour
            hourly = trades.groupby("Entry hour")["Net P&L"].agg(["count", "sum", "mean"]).reset_index()
            st.write("By entry hour (timestamp timezone as exported)")
            st.dataframe(hourly, use_container_width=True)

            st.subheader("Trade log")
            st.dataframe(trades.sort_values("Entry time", ascending=False), use_container_width=True)
            st.download_button("Download trade log CSV", trades.to_csv(index=False), "avp_trade_log.csv", "text/csv")
        else:
            st.warning("No completed trades in this date range. Check the signal counts above, the selected dates, and whether the uploaded file includes enough bars.")
            st.info("If signals are nonzero but trades are zero, confirm the start/end dates include those signal timestamps. The Python engine does not apply futures margin rules; it simulates the selected contract count.")
        st.subheader("Price and strategy levels")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=tested["time"], open=tested["open"], high=tested["high"], low=tested["low"], close=tested["close"], name="Price"))
        for col, label in [("vwap","Anchored VWAP"),("upper2","+2σ"),("lower2","−2σ"),("poc","POC"),("vah","VAH"),("val","VAL")]:
            fig.add_trace(go.Scatter(x=tested["time"], y=tested[col], mode="lines", name=label))
        buys = tested[tested["long_signal"]]
        sells = tested[tested["short_signal"]]
        fig.add_trace(go.Scatter(x=buys["time"], y=buys["low"], mode="markers", name="Long signal", marker_symbol="triangle-up", marker_size=9))
        fig.add_trace(go.Scatter(x=sells["time"], y=sells["high"], mode="markers", name="Short signal", marker_symbol="triangle-down", marker_size=9))
        fig.update_layout(height=650, xaxis_rangeslider_visible=False, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("Download indicator data CSV", tested.to_csv(index=False), "avp_indicator_data.csv", "text/csv")
    except Exception as e:
        st.error(f"Could not process this CSV: {e}")
else:
    st.info("Upload a TradingView CSV to begin. The app will calculate the indicators and run the backtest.")
    st.markdown("""
**Contract settings defaults**
- MNQ: $2 per index point, 0.25 tick
- NQ: $20 per index point, 0.25 tick

Use the matching point value for the instrument you exported. P&L is reported in USD and excludes any exchange, regulatory, or broker fees not entered above.
""")
