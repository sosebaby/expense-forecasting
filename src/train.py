from statsmodels.tsa.arima.model import ARIMA

def train_arima(train_series, order=(1,1,1)):
    """
    Train an ARIMA model on a pandas series.
    Args:
     train_series: pandas Series with datetime index .
     order: tuple, ARIMA order (p,d,q).
    Returns:
     model_fit: fitted ARIMA model.

    """
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    return model_fit