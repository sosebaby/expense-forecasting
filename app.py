import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("💰 Personal Finance Forecaster")
st.write("Forecasting your daily expenses with SARIMA — ready for visualization!")

# 1️⃣ Load Data
df = pd.read_csv("data/transactions.csv")
df.columns = df.columns.str.strip().str.lower()
df.rename(columns={"transaction date":"date", "amount":"amount"}, inplace=True)
df['date'] = pd.to_datetime(df['date'])

# 2️⃣ Aggregate daily expenses
daily = df.groupby('date')['amount'].sum().reset_index()
daily.rename(columns={'amount':'daily_expense'}, inplace=True)
daily.set_index('date', inplace=True)

# 3️⃣ Sidebar - forecast settings
st.sidebar.header("Forecast Settings")
days_to_predict = st.sidebar.slider("Days to Forecast", 7, 90, 30)

# 4️⃣ Train SARIMA
model = SARIMAX(daily['daily_expense'], order=(1,1,1), seasonal_order=(1,1,1,7))
results = model.fit()
forecast = results.get_forecast(steps=days_to_predict)

# 5️⃣ Forecast index
forecast_index = pd.date_range(start=daily.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)

# 6️⃣ Plotting - LinkedIn ready
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(14,6))

# Last 60 days of actuals
ax.plot(daily.index[-60:], daily['daily_expense'][-60:], label="Actual", color="#1f77b4", linewidth=2)

# Forecast
ax.plot(forecast_index, forecast.predicted_mean, label="Forecast", color="#ff7f0e", linewidth=2)

# Create three nice columns for the dashboard metrics
m1, m2, m3 = st.columns(3)
m1.metric("Algorithm", "SARIMAX")
m2.metric("Seasonal Period", "7 Days (Weekly)")
m3.metric("Last Daily Spend", f"${daily['daily_expense'].iloc[-1]:,.2f}")

st.divider() 

# Confidence interval
ci = forecast.conf_int()
ax.fill_between(forecast_index, ci['lower daily_expense'], ci['upper daily_expense'],
                color="#ff7f0e", alpha=0.25, label="Confidence Interval")

# Titles and labels
ax.set_title("Daily Expenses Forecast — Last 60 Days + Next 30 Days", fontsize=16, fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Daily Expense ($)")
ax.legend(loc="upper left")
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)
st.success(f"Forecast for next {days_to_predict} days generated successfully!")