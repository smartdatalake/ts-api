## time_series API


This is a service for Time Series Forecasting and Analysis. Currently, it includes three basic functions

#### Decompose: 
This method performs the Triple Time Series Decomposition. The service takes as input a time series, the corresponding model type ("Multiplicative" or "Additive"), the period and the corresponding locale if applicable.The user could select to insert the path of the selected data or provide them in an array form. The provided time series is decomposed into three distinct components according to the selected model and period:

-   Trend: the increasing - decreasing value in the series
-   Seasonality: the repeating short term cycle in the series
-   Residual Error: the random variation in the series

An additive model suggests that the components are added toghether as follows:

-   y(t) = Trend + Seasonality + Residual Error

While a multilicative model suggests that components are multiplied together as follows:

-   y(t) = Trend * Seasonality * Residual Error

This implementation uses the "statsmodels.tsa.seasonal.seasonal_decompose" from the statsmodels library.

<br>
<br>

#### Forecast
This method performs the Holt Winter Forecasting. The service takes as input a time series as well as the start, the end, the forecast range, the period and the corresponding locale if applicable. The user can provide the input time series (in the form of an array containing its values) or by specifying the path to a file in the SFTP server. The user is also able to provide a specified period parameter or give a set of possible periods and let our oprimisation algorithm to select the best configuration of the possible periods and types of trend and seasonal components. After providing the corresponding inputs, for each point i in the range [start,end] our algorithm computes and returns a forecast for the next N timestamps, as it is defined by the forecast range parameter.

This implementation uses the "statsmodels.tsa.holtwinters.ExponentialSmoothing" from the statsmodels library.

<br>
<br>

#### Catalog
This method returns all the available stock data files on the SFTP server.

<br>
<br>

#### Correlate

This method computes pairwise correlations in a set of given time series within a sliding time window. The service takes as input a list of different time series as well as the start, the window size, the step size, the number of steps and the corresponding locale if applicable. After providing the corresponding inputs, for each step i our service computes and returns the corresponding correlation matrix.

This implementation uses the "pandas.DataFrame.corr" from the pandas library.

<br>
<br>
<br>

### Usage

In order to start the service, the user should run the "python api.py" command at the command line.

<br>
<br>
<br>

### Documentation 

Once the service is running, API documentation is available at http://localhost:5001. 