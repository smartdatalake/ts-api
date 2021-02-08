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
    lists of lists:
        A list containing the predictions. Each prediction i in [start,end] is
        an list containing M = forecas_range values, which are the forecasted
        values starting from point i.
    int:
        Best period parameter
    String:
        Best trend type
    String:
        Best seasonal type
    float:
        z-score of the best config
        

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
    z_scores = []

    # Set up the parameter space for Grid Search
    cfg_list = holt_winters_configs(seasonal=period)

    # Iterate through the selected timestamps
    for i in range(start, end):

        # Grid Search to tune the period - trend - seasonal parameters according to the performance
        # of the model on the near past. [ith time - forecast range --> ith time] This will be used to
        # find the best parameters that they will also be used for predicting
        # [ith --> ith + forecast range]

        best_config, z_score = grid_search_HWES(
            cfg_list, d.iloc[0:i - forecast_range], d.iloc[i - forecast_range:i], forecast_range)
     
        t,s,p = best_config
        
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
        z_scores.append(z_score)

    return predictions, seasonality, trend_type, seasonal_type, z_scores




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
    t_params = ['add', 'mul',None]
    s_params = ['add', 'mul',None]
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
    list:
        The best parameter set [trend, seasonal, period]
    float:
        The Z-score of the best config towards the other possible combinations
    """
    best_RMSE = np.inf
    best_config = []
    scores = []
    print(cfg_list)
    for i in range(len(cfg_list)):
        try:
            t, s, p = cfg_list[i]
            print('trend_type: '+t)
            print('seasonal_type: '+s)
            print('period: '+str(p))
            model = HWES(train, seasonal_periods=p, trend=t, seasonal=s)
            fitted = model.fit()
            print(fitted)
            forecasted_values = fitted.forecast(steps=forecast_range)
            print(forecasted_values)
            rmse = np.sqrt(mean_squared_error(test, forecasted_values))
            
            print('rmse: '+str(rmse))
            scores.append(rmse)
            if rmse < best_RMSE:
                best_RMSE = rmse
                best_config = cfg_list[i]
        except BaseException:
            continue

    # Z-score calculation
    mean = np.mean(scores)
    std = np.std(scores)
    z_score = (best_RMSE - mean) / std
   
    return best_config, z_score




# C O R R E L A T I O N S - F U N C T I O N 
# ------------------------------------------

def correlate(data, start, window_size, step_size, steps, correlation_method):
    """This method computes pairwise correlations in a set of given time series within
    a sliding time window.

    Parameters
    ----------
    data: List of lists
        The provided list with the time series data
    start: int
        The start point of our calculations, assuming that the first timestamp of each
        time series is 0.
    window_size: int
        The length of the window
    step_size: int
        The length of each step
    steps: int
        Number of steps
    correlation_method: String
        Methods of correlation {‘pearson’, ‘kendall’, ‘spearman’} 
        

    Returns
    -------
    lists of lists:
        A list containing the correlation matrices 
    """

    # Convert data into pandas dataframe form
    d = pd.DataFrame(data)
    d = d.transpose()
    
    # Initialise the array that holds the correlation matrices for each window
    correlations = []
    
    # Iterate through the selected windows
    s = start
    
    for i in range(0,steps):
        m = d[s:s+window_size]
        
        # Calculate correlation matrix
        if correlation_method in ['pearson','kendall','spearman']:
            m= m.corr(correlation_method)
        else:
            m= m.corr() 
         
        # Transform correlation matrix from pandas to list
        m = m.values.tolist()
        
        # Append correlation matrix into correlations
        correlations.append(m)
        
        # Update start index
        s = s + step_size
        
    return correlations   
