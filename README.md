***Daily Expense Forecasting: From Statistical Baselines to Seasonal Models***

**Project Overview:**
This project focuses on predicting daily personal expenditures using time-series econometrics. By moving from a baseline ARIMA to a SARIMA architecture, this study demonstrates the process of identifying seasonal artifacts and handling high-volatility financial data. The final product is a deployed Streamlit dashboard that provides users with not just a point-estimate forecast, but a probabilistic range using 95% confidence intervals, allowing for better risk-adjusted financial planning.

**Data Engineering & EDA**
**Source**: Longitudinal daily transaction data (2020–2025).

**Preprocessing**: Implemented a robust pipeline to handle missing dates, standardize currency formats, and perform temporal aggregation.

**Key Discovery**: EDA revealed non-stationarity and strong annual seasonality (spikes during Q4/Holiday periods), indicating that a simple mean-reverting model would be insufficient.

**The Modeling Journey**
*Phase 1: The ARIMA(1, 1, 1) Baseline*

I began with a standard Autoregressive Integrated Moving Average model.

**Logic**: p=1,d=1,q=1 was chosen to handle the trend and provide a simple "memory" of the previous day's spend.

**Outcome**: The model captured the mean expenditure but resulted in a "flat" forecast.

**Analysis**: The high RMSE (1693) relative to MAE (1184) suggested that the model was significantly penalized by large, unpredicted outliers (spikes).

*Phase 2: SARIMA (Seasonal Upgrade)*

To address the "flat-line" issue, I implemented SARIMAX to incorporate seasonal lags.

**Refinement**: Added a seasonal order to account for the periodic cycles identified in the training data.

**Result**: The forecast transformed from a flat line to a rhythmic oscillation, successfully mimicking the "pulse" of the historical data.

**Metric Insight**: Even though the MAE remained stable (≈1188), the Structural Fit improved significantly, proving the model now understands when spending increases, even if it cannot predict the exact magnitude of random "shocks."

**Critical Analysis**
**The Residual Gap**: The persistent gap between RMSE and MAE indicates that the dataset contains Exogenous Shocks (one-time large purchases). In a professional setting, these would be addressed using "Holiday Effect" markers or a hybrid Anomaly Detection model.

**Stationarity**: I utilized the 'I' (Integration) component to transform the non-stationary spending trend into a stationary series, a prerequisite for stable forecasting.

**Future Roadmap**
1. Exogenous Variables: Integrate "Payday" and "Holiday" binary flags to reduce RMSE.

2. Modern Architectures: Compare SARIMA results against Meta’s Prophet and LSTM (Long Short-Term Memory) neural networks.

3. Deployment: Wrap the model in a Streamlit dashboard for real-time financial tracking.
