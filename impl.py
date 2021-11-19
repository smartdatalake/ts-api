# I M P O R T S
# -------------

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import copy
from scipy import stats
from statistics import mean

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
     
        try:
            t,s,p = best_config
        except:
            continue
        
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
    t_params = ['add', 'mul', None]
    s_params = ['add', 'mul', None]
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
            print('trend_type: '+str(t))
            print('seasonal_type: '+str(s))
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


# D E C O M P O S E - F U N C T I O N 
# -----------------------------------

def get_gain_index(trend,seasonality,model,real_values):
    
    """This method takes as input the trend, seasonality the model type
    as well as the real values of the time series and calculates the gain
    index. The gain index expresses the contribution of the seasonality to the actual time series. 
    In order to calculate it we calculate thepercentage difference between the percentage error of 
    trend vs actual time series and percentage error of trend + seasonality vs actual time series.
    The higher this index is the more likely is for the time series to have a strong seasonality component.
    
    Parameters
    ----------
    trend: List
        Trend component
    
    seasonality: List
        Seasonal component
    
    model: String
        The type of the model, 'additive' or 'seasonal'
        
    real_values: List
        Real values of the time series
 
    Returns
    -------
    float:
        The gain index, percentage which expresses the contribution of the seasonal component

    """
    
    error1 = [] # error between trend vs actual values
    error2 = [] # error between trend + seasonality vs actual values
    
    for i in range(0,len(real_values)):
        if trend[i] != None and real_values[i] != 0:
            
            if model == 'additive':
                predicted = trend[i] 
                real = real_values[i]
                error1.append(abs(real-predicted)/abs(real))
                
                predicted = trend[i] + seasonality[i]
                real = real_values[i]
                error2.append(abs(real-predicted)/abs(real))
                
                
                
            if model == 'multiplicative':
                predicted = trend[i] 
                real = real_values[i]
                error1.append(abs(real-predicted)/abs(real))
                
                predicted = trend[i] * seasonality[i]
                real = real_values[i]
                error2.append(abs(real-predicted)/abs(real))
    
    
    error =  (mean(error1) - mean(error2)) / mean(error1)          
            
            
    return error


def extract_best_period(ts,dates, periods,model):
    
    """This method decomposes the provided time series using all the provided periods
     and selects the one that gives the highest gain index. It also returns the calculated 
     gain indexes for all the tested periods. The gain index expresses the contribution of 
     the seasonality to the actual time series. In order to calculate it we calculate the
     percentage difference between the percentage error of trend vs actual time series and 
     percentage error of trend + seasonality vs actual time series. The higher this index is
     the more likely is for the time series to have a strong seasonality component.

    Parameters
    ----------
    ts: List
        The provided list with the time series values
    
    periods: List
        The provided list with the tested periods
    
    model: String
        The type of the model, 'additive' or 'seasonal'
 
    Returns
    -------
    int:
        The selected period (the one that gives the highest gain index)
    json:
        A json with the {period:gain_index} for all the tested periods
    """

    # Define gain_index and period
    gain_index = float('-inf') 
    p = -1 # best period
    periods_gain_indexes = {} 
    m = model
    
    # Apply z-normalisation to the time series
    ts = stats.zscore(ts)
    
    # Check if dates are provided and create a df
    if dates != 'None':
        
        # Create df
        d = {'DATE':dates,'values':ts}
        ts = pd.DataFrame(d)
        
        # create datetime index passing the datetime series
        datetime_index =ts['DATE']
        
        # Complete the call to convert the date column
        datetime_index =  pd.to_datetime(datetime_index,format='%m/%d/%Y')
        ts=ts.set_index(datetime_index)

        # we don't need the column anymore
        ts.drop('DATE',axis=1,inplace=True)
        
        ts = ts['values']
        
        
    
    # Find the period with the smallest mean_abs error
    for i in periods:
        
        # Seasonal decomposition
        result = seasonal_decompose(ts, period=i,model=m)
        
        trend = np.where(np.isnan(result.trend), None, result.trend)
        seasonality = np.where(np.isnan(result.seasonal),None,result.seasonal)
        res_error = np.where(np.isnan(result.resid), None, result.resid)
        
        
        #g = get_gain_index(trend,seasonality,model=m,real_values=ts['values'])
        g = get_gain_index(trend,seasonality,model=m,real_values=ts)
        
        if g > gain_index:
            gain_index = g
            p = i
            
        periods_gain_indexes.update({i:g})
        
    
    return p,periods_gain_indexes


# C H A N G E - D E T E C T I O N 
# --------------------------------

def rate_change(data, changepoints):
    
    """This method takes as input the data, the changing points and calculates the absolute and directional rate change.Rate change is a metric that expresses the significance of a changing point. In order to calculate it we take the absolute percentage difference between the average values of two consecutive segments given a selected window. 

    Parameters
    ----------
    data: List
        The provided list with the time series values
    
    changepoints: List
        The discovered changing points
 
    Returns
    -------
    Dictionary:
        The changing points among with their absolute rate change (changing point:rate_change value)
        
    Dictionary:
        The changing points among with their direcitonal rate change (changing point: directional rate_change value)
    """
    
    # Initialise a dictionary that holds the changing points and their rate of change
    rate_changes = {}
    dir_rate_changes = {}
        
    # Convert the data into pandas format
    df = pd.DataFrame(data,columns=['values'])
    
    for i in range(len(changepoints)-2):
        
        # Take the corresponding segments
        segment1 = df[changepoints[i]:changepoints[i+1]]
        segment2 = df[changepoints[i+1]:changepoints[i+2]]
        
        # Calculate their average value
        mean1 = segment1.mean().values[0]
        mean2 = segment2.mean().values[0]

        # Calculate their absolute percentage difference
       
        abs_percentage_diff = abs((mean2 - mean1) / ((mean1+mean2)/2))
       
        dir_percentage_diff = ((mean2 - mean1) / ((mean1+mean2)/2))
        
        # Update rate_changes
        rate_changes[changepoints[i+1]] = abs_percentage_diff 
        dir_rate_changes[changepoints[i+1]] = dir_percentage_diff
        
       
    return rate_changes, dir_rate_changes
