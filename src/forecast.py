def forecast_model(model_fit, steps):
    """
    Forecast using a fitted ARIMA model.
    
    Args:
        model_fit: fitted ARIMA model
        steps: int, number of steps to forecast
        
    Returns:
        forecast: pandas Series of forecasted values
    """
    forecast = model_fit.forecast(steps=steps)
    return forecast