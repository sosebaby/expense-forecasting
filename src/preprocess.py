import pandas as pd

def load_and_clean(path):
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    daily = df.groupby("date")["amount"].sum().reset_index()
    daily.rename(columns={"amount": "daily_expense"}, inplace=True)

    return daily
    #here: datetime handling, aggregation logic, why time-series needs ordering