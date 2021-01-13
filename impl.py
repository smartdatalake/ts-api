# I M P O R T S
# -------------

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose


# F O R E C A S T I N G - F U N C T I O N S
# -----------------------------------------

def forecast(data, start, end, forecast_range, period):
    """'forecast' method is responsible for forecasting time series data within
    the set start, end and forecast range.

    Parameters
    ----------
    data: list
        The provided list with the time series data
    start: int
        The start point of our predictions
    end: int
        The end point of our predictions
    forecast_range: int
        The number of points to predict at each step
    period: int
        The seasonal parameter

    Returns
    -------
    lists of lists
        A list containing the predictions. Each prediction i in [start,end] is
        an list containing M = forecas_range values, which are the forecasted
        values starting from point i.

    """

    # Convert Data into pandas dataframe form
    d = pd.DataFrame(data)

    # Initialise the array that holds the predictions for each timestamp
    predictions = []

    # Initialise the arrays that holds the optimised parameters used for the
    # predicitons of each timestamp
    best_configurations = []
    seasonality = []
    trend_type = []
    seasonal_type = []

    # Set up the parameter space for Grid Search
    cfg_list = holt_winters_configs(seasonal=period)

    # Iterate through the selected timestamps
    for i in range(start, end):

        # Grid Search to tune the period - trend - seasonal parameters according to the performance
        # of the model on the near past. [ith time - forecast range --> ith time] This will be used to
        # find the best parameters that they will also be used for predicting
        # [ith --> ith + forecast range]

        t, s, p = grid_search_HWES(
            cfg_list, d.iloc[0:i - forecast_range], d.iloc[i - forecast_range:i], forecast_range)

        # Train split
        df_train = d.iloc[0:i + 1]

        # Forecasting
        model = HWES(df_train, seasonal_periods=p, trend=t, seasonal=s)
        fitted = model.fit()
        forecasted_values = fitted.forecast(steps=forecast_range)

        # Append the data into the arrays
        predictions.append(forecasted_values.tolist())
        best_configurations.append([p, t, s])
        seasonality.append(p)
        trend_type.append(t)
        seasonal_type.append(s)

    return predictions, seasonality, trend_type, seasonal_type


# G R I D  S E A R C H -  O P T I M I S A T I O N  F U N C T I O N S
# ------------------------------------------------------------------

def holt_winters_configs(seasonal=[None]):
    """'This method sets up the Holt Winter's parameter space for the Grid Search

    Parameters
    ----------
    data: list
        Seasonal values

    Returns
    -------
    list
        parameter space
    """
    models = list()
    # define config lists
    t_params = ['add', 'mul']
    s_params = ['add', 'mul']
    p_params = seasonal
    # create config instances
    for t in t_params:
        for s in s_params:
            for p in p_params:
                cfg = [t, s, p]
                models.append(cfg)
    return models


def grid_search_HWES(cfg_list, train, test, forecast_range):
    """'This is the Grid Search method which is used to find the best parameter set for the
    Holt Winter algorithm for the near past in order to be used for our near future predictions

    Parameters
    ----------
    cfg_list: list
        Parameter space
    train: list
        Training data
    test: list
        Testing data to tune the performance of the model
    forecast_range: int
        The number of points to predict at each step

    Returns
    -------
    list
        The best parameter set [trend, seasonal, period]
    """
    best_RMSE = np.inf
    best_config = []
    for i in range(len(cfg_list)):
        try:
            t, s, p = cfg_list[i]
            model = HWES(train, seasonal_periods=p, trend=t, seasonal=s)
            fitted = model.fit()
            forecasted_values = fitted.forecast(steps=forecast_range)
            rmse = np.sqrt(mean_squared_error(test, forecasted_values))
            if rmse < best_RMSE:
                best_RMSE = rmse
                best_config = cfg_list[i]
        except BaseException:
            continue
    return best_config
