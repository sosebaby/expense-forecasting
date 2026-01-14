import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("💰 Personal Finance Forecaster")
st.write("Forecasting your daily expenses with SARIMA — ready for visualization!")

df = pd.read_csv("data/transactions.csv")
df.columns = df.columns.str.strip().str.lower()
df.rename(columns={"transaction date":"date", "amount":"amount"}, inplace=True)
df['date'] = pd.to_datetime(df['date'])

daily = df.groupby('date')['amount'].sum().reset_index()
daily.rename(columns={'amount':'daily_expense'}, inplace=True)
daily.set_index('date', inplace=True)

st.sidebar.header("Forecast Settings")
days_to_predict = st.sidebar.slider("Days to Forecast", 7, 90, 30)

model = SARIMAX(daily['daily_expense'], order=(1,1,1), seasonal_order=(1,1,1,7))
results = model.fit()
forecast = results.get_forecast(steps=days_to_predict)


forecast_index = pd.date_range(start=daily.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)

sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(14,6))

# Last 60 days of actuals
ax.plot(daily.index[-60:], daily['daily_expense'][-60:], label="Actual", color="#1f77b4", linewidth=2)

# Forecast
ax.plot(forecast_index, forecast.predicted_mean, label="Forecast", color="#ff7f0e", linewidth=2)

# Create dynamic metrics based on the forecast
m1, m2, m3 = st.columns(3)

# 1. Total Projected Spend for the selected period
total_projected = forecast.predicted_mean.sum()
m1.metric("Projected Total", f"${total_projected:,.2f}")

# 2. Highest Expected Spike in the forecast
max_spike = forecast.predicted_mean.max()
m2.metric("Expected Peak", f"${max_spike:,.2f}")

# 3. Average Daily Forecast
avg_forecast = forecast.predicted_mean.mean()
m3.metric("Avg. Forecast", f"${avg_forecast:,.2f}")

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



st.divider() 
st.subheader("📅 Forecasted Breakdown")
st.write("Below are the exact predicted values and confidence ranges for the selected period.")

forecast_df = pd.DataFrame({
    'Date': forecast_index,
    'Predicted Spend ($)': forecast.predicted_mean.values,
    'Lower Bound ($)': ci['lower daily_expense'].values,
    'Upper Bound ($)': ci['upper daily_expense'].values
})


forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')

st.dataframe(
    forecast_df.style.format({
        'Predicted Spend ($)': '{:,.2f}',
        'Lower Bound ($)': '{:,.2f}',
        'Upper Bound ($)': '{:,.2f}'
    }), 
    use_container_width=True
)

csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Forecast as CSV",
    data=csv,
    file_name='expense_forecast_results.csv',
    mime='text/csv',
)