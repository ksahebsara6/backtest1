# AVP 2σ VWAP Reversal Backtester

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## TradingView data
Export chart data as CSV with time, open, high, low, close, and volume columns.
Upload the CSV in the app.

## Important modeling notes
- Volume profile is approximate: each candle's full volume is assigned to the HLC3 price bin.
- The bin size is set from ATR on the first bar of each anchor period and remains fixed until the next anchor.
- VWAP standard deviation is volume-weighted population standard deviation.
- Signals are evaluated at candle close; a newly opened position can first exit on the following bar.
- If stop and target are both touched in one candle, the engine assumes the stop was hit first (conservative).
- Slippage is applied to entries and exits; commission is charged per contract per side.
- Data timestamps retain the timezone/interpretation provided by the CSV. Verify the exported timezone before comparing RTH vs overnight results.
- This is a research backtester, not a broker execution simulator. Validate results with finer-resolution data and realistic fees.
