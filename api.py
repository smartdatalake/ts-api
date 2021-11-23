# I M P O R T S
# -------------

from flask import Flask
from flask_restx import Api, Resource, fields
from datetime import datetime
import locale
import logging
import pysftp
import sys
import math
import json
import requests
from impl import *
from config import *
from datetime import datetime
import ruptures as rpt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import copy


# S F T P - S E R V E R - C O N N E C T I O N
# -------------------------------------------

cnopts = pysftp.CnOpts()
cnopts.hostkeys = None



# A P P L I C A T I O N   U S E R   I N T E R F A C E
# ---------------------------------------------------

# Initialise Application
# ----------------------
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
api = Api(app)
name_space = api.namespace(
    'time_series',
    description='This is a sevice for Time Series Forecasting and Analysis.')


# Set up the structure of the model for the Forecast Method
# ---------------------------------------------------------
a_forecast = api.model(
    'Forecast input values', {
        'start': fields.Integer(
            description='The start point of our predictions'),
        'end': fields.Integer(description='The end point of our predictions'),
        'forecast_range': fields.Integer(description='The number of points to predict at each step'),
        'period': fields.List(fields.Integer(description='The seasonal parameter or set of seasonal parameters')),
        'data': fields.List(fields.String(description='The provided list with the time series data or string which represents the path for the data')),
        'locale': fields.String(description='The corresponding locale, None if it does not exist', default='None', required=False), 
        'api_key': fields.String(description='API key')})


# Set up the structure of the model for the Decompose Method
# ----------------------------------------------------------
a_timeserie = api.model(
    'Decompose input values', {
        'data': fields.List(fields.String(description='The provided list with the time series data or string which represents the path for the data')), 
        'dates': fields.List(fields.String(description='The provided list with the timestamps of the provided time series, None if does not exist')),
        'model': fields.String(description='additive or multiplicative'), 
        'periods': fields.List(fields.Integer(description='The seasonal parameters')),
        'locale': fields.String(description='The corresponding locale, None if it does not exist', default='None', required=False), 
        'api_key': fields.String(description='API key')})



# Set up the structure of the model for the Get Data Method
# ----------------------------------------------------------
a_getData = api.model(
    'GetData input values', {
        'path': fields.String(description='Name of the file'), 
        'start': fields.String(description='Start Date format %Y%m%d 2014-06-18, None if you want to take the whole time series.'), 
        'end': fields.String(description='End Date format %Y%m%d 2014-06-18, None if you want to take the whole time series.'),
        'api_key': fields.String(description='API key')})

# Set up the structure of the model for the Catalog method
# ----------------------------------------------------------
a_catalog = api.model(
    'Catalog input values', {
        'api_key': fields.String(description='API key')})


# Set up the structure of the model for the Correlate Method
# ----------------------------------------------------------
a_correlate = api.model(
    'Correlate input values', {
        'data': fields.List(
            fields.List(
                fields.String(
                description='The provided list with the time series data or string which represents the path for the data'))),
        'start': fields.Integer(
            description='The start point of our calculations'),
        'window_size': fields.Integer(
                description='The length of the window'), 
        'step_size': fields.Integer(
                    description='The length of each step'),
        'steps': fields.Integer(
                        description='Number of steps'),
        'correlation_method': fields.String(
                        description='Methods of correlation {‘pearson’, ‘kendall’, ‘spearman’} s'),
        
        'locale': fields.String(description='The corresponding locale, None if it does not exist', default='None', required=False),
        
        'api_key': fields.String(description='API key')})



# Set up the structure of the model for the buildAndStoreIndex Method
# -------------------------------------------------------------------
a_buildAndStoreIndex = api.model(
    'BuildAndStoreIndex input values', {
        
        'dataPath': fields.String(description='Path to a remote (e.g., on an SFTP server) or local location containing input time aligned time series'),
        
        'doNormalize': fields.String(description='Whether to apply z-normalization or not'),
        
        'doPAA': fields.String(description='The length of the window'), 
        
        'doSmoothing': fields.String(description='Whether to apply LOESS smoothing or not'),
        
        'indexFile': fields.String(description='Path to a remote index file (e.g., on an SFTP server) or local index file'),
        
        
           'noSeg': fields.String(description='If PAA is applied, this is the number of segments'),
           
           'smoothBandwidth': fields.String(description='If smoothing is applied, the percentage (from 0.0 to 1.0) of closest timestamps to be considered in LOESS'),
        
        'api_key': fields.String(description='API key')})



# Set up the structure of the model for the discoverBundles Method
# ----------------------------------------------------------------
a_discoverBundles = api.model(
    'DiscoverBundles input values', {
        
        'delta': fields.String(description='The delta threshold for bundle discovery'),
        
        'doStore': fields.String(description='Whether to store the results or not'),
        
        'epsilon': fields.String(description='The epsilon threshold for bundle discovery'), 
        
        'indexFile': fields.String(description='Path to a remote index file (e.g., on an SFTP server) or local index file'),
        
        'mu': fields.String(description='The mu threshold for bundle discovery'),
        
           'outputFile': fields.String(description='Path to the file where the results will be stored'),
           
           'smoothBandwidth': fields.String(description='If smoothing is applied, the percentage (from 0.0 to 1.0) of closest timestamps to be considered in LOESS'),
        
        'api_key': fields.String(description='API key')})



# Set up the structure of the model for the selfJoin Method
# ---------------------------------------------------------
a_selfJoin = api.model(
    'DiscoverBundles input values', {
        
        'delta': fields.String(description='The delta threshold for pair discovery'),
        
        'doStore': fields.String(description='Whether to store the results or not'),
        
        'epsilon': fields.String(description='The epsilon threshold for pair discovery'), 
        
        'indexFile': fields.String(description='Path to a remote index file (e.g., on an SFTP server) or local index file'),
        
           'outputFile': fields.String(description='Path to the file where the results will be stored'),
        
        'mu': fields.String(description='The mu threshold for bundle discovery'),
        
        'api_key': fields.String(description='API key')})


# Set up the structure of the model for the Find_changing_points Method
# ---------------------------------------------------------------------
a_changepoint = api.model(
    'Change point detection input values', {
        
        'data': fields.List(fields.List(fields.String(description='The provided list with the time series data or string which represents the path for the data'))),
        
        'locale': fields.String(description='The corresponding locale, None if it does not exist', default='None', required=False),
        
        'api_key': fields.String(description='API key')})





# Set up logging system
# ---------------------
ns1 = api.namespace('api/v1', description='test')
fh = logging.FileHandler("v1.log")
ns1.logger.addHandler(fh)






# Initialise the Change point Class
# ---------------------------------
@name_space.route('/change_points')
class Change_points(Resource):

    @api.expect(a_changepoint)
    def post(self):
        """ Time series changing points detection

        Description:
        -----------
        This method identifies the changing points within a collection of time series, ranks them and distinguishes them between global and local changes. In order to find the changing points, our implementation uses Pelt approach (rupture library) and calculates their rate change metric. Subsequently, using DBSCAN, it creates some clusters which include the changing points that are part of the same global change. Our implementation returns, a dictionary including all the identified changing points among with their rate change scores and global-local change labels as well as a dictionary with the corresponding clusters (global changes).
        
        ----------

        Parameters:
        -----------

        data: List of lists
        -----
              The provided list with the time series data.
              
        dates: List of lists
        -------
               The provided list with the timestamps of the provided time series, None if does not exist. 
               
        min_size:
        --------
                Minimum number of samples between two change points (ruptures).
                
        min_samples:
        ------------
                The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself (dbscan).
                
        eps:
        ----
                The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster (dbscan).

        locale: String
        -----
            The corresponding locale or 'None' to use the default.

        api_key: String
        -----
            API user's key.

        stocks_format: Boolean
        ------
            A boolean representing whether the input data follows the stocks-specific format.

        ------

        Returns:
        -------

        change_points: Dictionary
        ------------
            A dictionary containing the timestamps/dates of the identified change points, the name or id of the corresponding time series, the rate change of the change points, local-global cluster label (-1 stands for local changes)
            
        cluters: Dictionary
        ------------
            A dictionary containing all the identified clusters (global changes), their aggregate and absolute aggregate rate changes and corresponding cluster scores, their starting and ending date or timestamp and the number of members of each cluster.
  
        """
        # API key check
        k = api.payload['api_key']
        if k in keys:

            # Logging request
            ns1.logger.info('- Change points Request')
            ns1.logger.info(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            ns1.logger.info(api.payload)
            
            # Parsing data
            data = api.payload['data']
            l = api.payload['locale']
            dates = api.payload['dates']
            min_size = api.payload['min_size']
            min_samples = api.payload['min_samples']
            eps = api.payload['eps']
            stocks_format = api.payload['stocks_format']
            

            # Create Final Dataset
            final_data = pd.DataFrame(columns=['timestamps','rate_change','dir_rate_change','name'])
            num_of_stock = 1
            sftp_v = False
            
            # Iterate through the provided list with the stocks
            for i in range(0, len(data)):
                curr_data = data[i]
                values = [] # closing values
                stock_dates = [] # dates of the specific stock

                # Check whether we have a stocks format
                if stocks_format == True:
                    j = 0
                    num_of_stock = i
                    for k in curr_data:
                        if j > 0:
                            x = k.split(',')
                            stock_dates.append(x[0])
                            values.append(x[5])
                        j = j + 1

                    d = values
                    entry = str(num_of_stock)
                    
                elif isinstance(curr_data, str):
                    with pysftp.Connection(host=hostname, username=username, password=password, private_key=".ppk", cnopts=cnopts) as sftp:
                        with sftp.cd(path):
                            sftp_v = True
                            with sftp.open(curr_data,) as remote_file:                            
                                j = 0 
                                num_of_stock = curr_data
                                for line in remote_file:
                                    if j > 0:
                                        x = line.split(",")
                                        values.append(x[2])
                                        stock_dates.append(x[0])
                                    j = j + 1

                    d = values
                    entry = str(num_of_stock[0])
                else:
                    entry = num_of_stock
                    num_of_stock = num_of_stock+1
                    d = data[i]
                    
                # Locale manipulation
                if l != "None":
                    locale.setlocale(locale.LC_ALL, l)
                    d = [locale.atof(item) for item in d]
                else:
                    # Convert data into floats
                    d = np.array(d, dtype=np.float32)
                    
                    
                # Set up parameters for ruptures
                # ------------------------------
                model = "l1"  # “l2”, “normal”, “rbf”, “linear”, “ar” 
                #min_size = 10
                jump = 5
                signal = d
                
                
                # Fit the model
                # -------------
                algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal)
                my_bkps = algo.predict(pen=3)
                
                
                # Calculate the rate of changes 
                # -----------------------------
                my_bkps.insert(0, 0)
                results,dir_results = rate_change(signal,my_bkps)
                
                
                # Create the final dataset
                # -----------------------
                df = pd.DataFrame(list(results.items()),columns = ['timestamps','rate_change'])                 
                dir_df = pd.DataFrame(list(dir_results.items()),columns = ['timestamps','dir_rate_change'])               
                dir_rate_values = dir_df['dir_rate_change'].tolist()         
                df['dir_rate_change'] = dir_rate_values
                df['name'] = entry   
                
                
                # Add dates if exist
                # ------------------
                if sftp_v == True:                  
                    changepoints = df['timestamps'].tolist()
                    changepoints_dates = []
                    for i in changepoints:
                        changepoints_dates.append(stock_dates[i])
                    df.insert(2, 'date', changepoints_dates)
                if dates != 'None' and sftp_v==False:
                    changepoints = df['timestamps'].tolist()
                    changepoints_dates = []
                    for j in changepoints:
                        changepoints_dates.append(dates[i][j])
                    df.insert(2, 'date', changepoints_dates)      
                final_data = pd.concat([final_data, df], ignore_index=True)
                
            # Sort the changing points according to rate change index
            # -------------------------------------------------------
            final_data  = final_data.sort_values(by='rate_change', ascending=False,ignore_index=True)
            
            
            # DBSCAN for identifying Global - Local changes
            # ----------------------------------------------
            
            # Τake the extracted timestamps of the changing points from the whole collection
            timestamps = final_data['timestamps'].values
            X = timestamps.reshape(-1,1)
            
            # Call the DBSCAN algorithm
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            
            # Add labels to the data
            final_data['cluster label'] = labels
            
            
            # Cluster datarame
            # ----------------
            
            # Create a new df for the cluster dataframe
            data_changes = copy.deepcopy(final_data)
            
            # convert timestamps column to floats
            data_changes['timestamps'] = data_changes['timestamps'].astype(float)
            
  
            
            # get mean rate_change
            data_clusters = data_changes.groupby('cluster label', as_index=False) ['rate_change','dir_rate_change'].mean()
            
            # get min-max date of each cluster
            data_clusters['min_timestamp'] = data_changes.groupby('cluster label', as_index=False)['timestamps'].min()['timestamps']
            
            data_clusters['max_timestamp'] = data_changes.groupby('cluster label', as_index=False)['timestamps'].max()['timestamps']
            
            # add an extra column that will hold info regarding with the number of changing points at each cluster
            data_clusters['counts'] = data_clusters['cluster label'].map(data_changes["cluster label"].value_counts())

            # add score column
            data_clusters['cluster_abs_score'] = data_clusters['rate_change'] * data_clusters['counts']

            # add  dir score column
            data_clusters['cluster_dir_score'] = data_clusters['dir_rate_change'] * data_clusters['counts']
            
            # Check if dates exist and add the min max date to the cluster dataframe
            if 'date' in data_changes.columns:
                print('add dates')
                min_dates = []
                max_dates = []
                for index, row in data_clusters.iterrows():
                    min_dates.append(data_changes.loc[(data_changes['cluster label'] == row['cluster label']) & (data_changes['timestamps'] == row['min_timestamp'])]['date'].reset_index(drop=True)[0])
                    max_dates.append(data_changes.loc[(data_changes['cluster label'] == row['cluster label']) & (data_changes['timestamps'] == row['max_timestamp'])]['date'].reset_index(drop=True)[0])
       
                
                data_clusters['min_dates'] = min_dates
                data_clusters['max_dates'] = max_dates

            # Remove the local changes cluster 
            data_clusters = data_clusters.iloc[1: , :]

            # Sort the df according to the score 
            data_clusters = data_clusters.sort_values(by=['cluster_abs_score'],ascending=False)
            data_clusters = data_clusters.reset_index(drop=True)
        
      
            dates = final_data['timestamps'].apply(lambda x: stock_dates[x])
            final_data['date'] = dates
            final_data = final_data.drop(['timestamps', 'rate_change'], axis=1)
            final_data.columns = ['rate_change', 'id', 'cluster_label', 'date']

            dates_min = data_clusters['min_timestamp'].apply(lambda x: stock_dates[int(x)])
            dates_max = data_clusters['max_timestamp'].apply(lambda x: stock_dates[int(x)])
            data_clusters['min_date'] = dates_min
            data_clusters['max_date'] = dates_max
            data_clusters = data_clusters.drop(['min_timestamp', 'max_timestamp'], axis=1)
            data_clusters.columns = ['cluster_label', 'aggr_rate_change_abs', 'aggr_rate_change', 'members_count', 'cluster_score_abs', 'cluster_score', 'min_date', 'max_date']


            changes = final_data.to_dict()
            clusters = data_clusters.to_dict()
            
            return {'changing_points':changes,'clusters':clusters}, 201  
        else:
            print("Wrong api key")



# Initialise the Decompose Class
# ------------------------------
@name_space.route('/decompose')
class Decompose(Resource):

    @api.expect(a_timeserie)
    def post(self):
        """ Trend - Sesonality - Residual Time Series Decomposition.

        Description:
        -----------
        This method performs the Triple Time Series Decomposition. The service takes as input a time series, the corresponding model type ("Multiplicative" or "Additive"), a list of periods parameters and the corresponding locale if applicable.The user could select to insert the path of the selected data or provide them in an array form. The provided time series is decomposed into three distinct components according to the selected model and period. If the user provides more than 1 period parameters, our system selects the best one according to the best gain index. The gain index is a metric (x %) which expresses the contribution of the seasonality to the actual time series. In order to find it we calculate the percentage difference between the percentage error of trend vs actual time series and percentage error of trend + seasonality vs actual time series. The higher this index is the more likely is for the time series to have a strong seasonality component.
        
        - Trend: the increasing - decreasing value in the series
        - Seasonality: the repeating short term cycle in the series
        - Residual Error: the random variation in the series

        An additive model suggests that the components are added toghether as follows:
        - y(t) = Trend + Seasonality + Residual Error

        While a multilicative model suggests that components are multiplied together as follows:
        - y(t) = Trend * Seasonality * Residual Error

        This implementation uses the "statsmodels.tsa.seasonal.seasonal_decompose" from the statsmodels library.

        ----------

        Parameters:
        -----------
        data: List of floats or string (data path)
        -----
            The provided list with the time series data or string which represents the path for the data.
            
        dates:
        ------
            The provided list with the timestamps of the provided time series, None if does not exist. 

        model: String
        -----
            'multiplicative' for multiplicative model, 'additive' for additive model.

        periods: List
        ------
            A list with the tested seasonal parameters.

        locale: String
        -----
            The corresponding locale or 'None' to use the default.

        api_key: String
        -----
            API user's key.

        stocks_format: Boolean
        ------
            A boolean representing whether the input data follows the stocks-specific format.

        ------

        Returns:
        -------
        
        best_period: The selected best period based on the minimum mean absolute value of the residual error component
        ----------
             The selected best period based on the minimum mean absolute value of the residual error component
        
        gain_values:
        -----------
             Expresses the significance of the difference between the mean_abs value of the res_error component from each period vs the mean of the mean_abs values of the res_error componenet from the rest periods.
  
        trend: list
        -----
            A list containing the data of the trend component.

        seasonality: list
        -----------
            A list containing the data of the seasonal component.

        residual error: list
        ---------------
            A list containing the data of the residual error component.
            
        dates: List
        ----------
            A list containing the corresponding dates if exist.
            

        """
        # API key check
        k = api.payload['api_key']
        if k in keys:

            # Logging request
            ns1.logger.info('- Decompose Request')
            ns1.logger.info(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            ns1.logger.info(api.payload)
            # Parsing Data
            data = api.payload['data']
            dates = api.payload['dates']
            m = api.payload['model']
            p = api.payload['periods']
            l = api.payload['locale']
            stocks_format = api.payload['stocks_format']
            
            # Check whether we have a stocks format
            if stocks_format == True:
                data_ = []
                dates = []
                for i in data:
                    x = i.split(',')
                    dates.append(x[0])
                    data_.append(x[5])
                data = data_
                    

            # Check whether we got a sftp request or raw data
            elif isinstance(data, str) and spring_format==False:
                print('sftp')
                with pysftp.Connection(host=hostname, username=username, password=password, private_key=".ppk", cnopts=cnopts) as sftp:
                    with sftp.cd(path):
                        with sftp.open(data) as remote_file:
                            i = 0
                            data = []
                            dates = []
                            for line in remote_file:
                                if i > 0:
                                    x = line.split(",")
                                    data.append(x[2])
                                    dates.append(x[0])
                                i = i + 1
           
            # Locale Manipulation
            if l != "None":
                locale.setlocale(locale.LC_ALL, l)
                data = [locale.atof(item) for item in data]
            else:
                # Convert data into floats
                data = np.array(data, dtype=np.float32)
                
            # Time Series Decomposition into Trend Seasonality and Residual
            best_period,gain_indexes = extract_best_period(data,dates,p,model=m)  
            result = seasonal_decompose(data, period=best_period, model=m) 

            # Replace NaN values with None
            trend = np.where(np.isnan(result.trend), None, result.trend)
            seasonality = np.where(
                np.isnan(
                    result.seasonal),
                None,
                result.seasonal)
            res_error = np.where(np.isnan(result.resid), None, result.resid)
        
            
            
            return {'best_period':best_period,'gain_indexes':gain_indexes,'trend': list(trend), 'seasonality': list(
                seasonality), 'residual error': list(res_error), 'dates':dates}, 201
        else:
            print("Wrong api key")


# Initialise the Catalog Class
# ------------------------------
@name_space.route('/catalog')
class Catalog(Resource):

    @api.expect(a_catalog)
    def post(self):
        """ Return the available data files.

        Description:
        -----------
        This method returns all the available stock data files on the SFTP server.

        Returns:
        -------

        data: list
        -----
            A list containing the available data on the SFTP server.

        """
        # API key check
        k = api.payload['api_key']
        if k in keys:

            # Logging request
            ns1.logger.info('- Catalog Request')
            ns1.logger.info(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

            # Connect to the SFTP and get the file names
            available_data = []
            with pysftp.Connection(host=hostname, username=username, password=password, private_key=".ppk", cnopts=cnopts) as sftp:
                with sftp.cd(path):
                    print("Connection succesfully stablished ... ")
                    directory_structure = sftp.listdir_attr()
                    for attr in directory_structure:
                        with sftp.open(attr.filename) as remote_file:
                            d = pd.read_csv(remote_file,names=['Date','Time','Open', 'High', 'Low', 'Close', 'Volume'])
                            start=d['Date'].values[0]
                            start = start.replace("/", "-")
                            end=d['Date'].values[-1]
                            end = end.replace("/", "-")
                        available_data.append(([attr.filename,str(len(d)),start,end]))
                        #a = {attr.filename :{ "length": 1531,"start": "03-30-2013","end": "07-12-2019"}}
                        #available_data.append(a)
                        
            return {'filename_length_start_end': available_data}, 201
        else:
            print("Wrong api key")
    
    
# Initialise the getData Class
# ------------------------------
@name_space.route('/getData')
class GetData(Resource):

    @api.expect(a_getData)
    def post(self):
        """ Return the available raw data of a specified stock.

        Description:
        -----------
        This method returns the raw data of the specified interval of a selected stock.

        ----------

        Parameters:
        -----------
        path: Name of the stock
        ----
            The name of the file of the selected stock from the sftp server.
           
        start: String
        -----
            Start Date format %Y%m%d 2014-06-18, 'None' if you want to take the whole time series.
        
        end: String
        -----
            End Date format %Y%m%d 2014-06-18, 'None' if you want to take the whole time series.

        api_key: String
        -----
            API user's key.

        Returns:
        -------

        data: list
        -----
            A list containing the requested raw data.

        """

        # API key check
        k = api.payload['api_key']
        if k in keys:

            # Logging request
            ns1.logger.info('- getData Request')
            ns1.logger.info(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            ns1.logger.info(api.payload)
            
            # Parsing data
            data_name = api.payload['path']
            start = api.payload['start']
            end = api.payload['end']   

            # Connection with sftp
            with pysftp.Connection(host=hostname, username=username, password=password, private_key=".ppk", cnopts=cnopts) as sftp:
                    with sftp.cd(path):
                        with sftp.open(data_name) as remote_file:
                            #colnames = ['Date','Time','Open', 'High', 'Low', 'Close', 'Volume']
                            colnames = ['Date','Time', 'Close']
                            dateparse = lambda x: datetime.strptime(x,'%m/%d/%Y')
                            
                            d = pd.read_csv(remote_file, header=None, parse_dates=[0], date_parser=dateparse,names=colnames)

                            if start == 'None' and end == 'None':
                                df = d['Close'].values.tolist()
                                
                               
                                dates = d['Date'].astype(str).values.tolist()
                                
                            else:
                                
                                df = d[(d['Date'] >= start) & (d['Date'] <= end)]
                                dates = df['Date'].astype(str).values.tolist()
                                df = df['Close'].values.tolist()
                                
        else:
            print('Wrong API key')

        return {'data': df, 'dates':dates},201          


    
    
# Initialise the Forecast opt Class
# -----------------------------
@name_space.route('/forecast')
class Forecast(Resource):

    @api.expect(a_forecast)
    def post(self):
        """ Holt Winters Forecasting.

        Description:
        -----------
        This method performs the Holt Winter Forecasting. The service takes as input a time series as well as the start, the end, the forecast range, the period and the corresponding locale if applicable. The user can provide the input time series (in the form of an array containing its values) or by specifying the path to a file in the SFTP server. The user is also able to provide a specified period            parameter or give a set of possible periods and let our oprimisation algorithm to select the best configuration of the possible periods and types of trend and seasonal components. After providing the corresponding inputs, for each point i in the range [start,end] our algorithm computes and returns a forecast for the next N timestamps, as it is defined by the forecast range parameter.

        This implementation uses the "statsmodels.tsa.holtwinters.ExponentialSmoothing" from the statsmodels library.

        ----------

        Parameters:
        -----------

        start: Ιnteger
        -----
            The start point of our predictions.

        end: Integer
        ----
            The end point of our predictions.

        forecast_range: Integer
        --------------
            The number of points to predict at each step.

        period: List of integers
        ------
            The seasonal parameter or set of seasonal parameters.

        data: List of floats or string (data path)
        -----
            The provided list with the time series data or string which represents the path for the data.

        locale: String
        -----
            The corresponding locale or 'None' to use the default.

        api_key: String
        -----
            API user's key.

        stocks_format: Boolean
        ------
            A boolean representing whether the input data follows the stocks-specific format.

        ------

        Returns:
        -------

        predictions, best_configurations: Lists of lists
        -----------
            A list containing the prediction sets accompanied by a list which contains the corresponding configuration sets (seasonality, trend type, seasonal type).Each prediction set i within the [start,end] range is a list containing M=forecast_range values, which are the the forecasted values starting from point i.Each configuration set i within the [start, end] range is a list containing four other lists with the best parameters (period, type of trend component, type of seasonal component) calculated by our optimisation algorithm for each prediction set as well as the best configuration's rmse z-score. Z-score metric gives us an intuition about how better Holt-Winters works when using the best configuration set instead of all the other possible combinations of the provided periods and trend - seasonal type.

        """
        # API key check
        k = api.payload['api_key']
        if k in keys:

            # Logging request
            ns1.logger.info('- Forecast Request')
            ns1.logger.info(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            ns1.logger.info(api.payload)
            # Parsing data
            start = api.payload['start']
            end = api.payload['end']
            forecast_range = api.payload['forecast_range']
            period = api.payload['period']
            data = api.payload['data']
            l = api.payload['locale']
            stocks_format = api.payload['stocks_format']

            # Check whether we have a stocks format
            if stocks_format == True:
                data_ = []
                dates = []
                for i in data:
                    x = i.split(',')
                    dates.append(x[0])
                    data_.append(x[5])
                data = data_


            # Check whether we got a sftp request or raw data
            elif isinstance(data, str):
                print('sftp')
                with pysftp.Connection(host=hostname, username=username, password=password, private_key=".ppk", cnopts=cnopts) as sftp:
                    with sftp.cd(path):
                        with sftp.open(data) as remote_file:
                            i = 0
                            data = []
                            for line in remote_file:
                                if i > 0:
                                    x = line.split(",")
                                    data.append(x[2])
                                i = i + 1

            # Locale manipulation
            if l != "None":
                locale.setlocale(locale.LC_ALL, l)
                data = [locale.atof(item) for item in data]
            else:
                # Convert data into floats
                data = np.array(data, dtype=np.float32)

            predictions, seasonality, trend_type, seasonal_type, z_score = forecast(
                data, start, end, forecast_range, period)
            return {'predictions': predictions, 'best_configurations': {
                'seasonality': seasonality, 'trend_type': trend_type, 'seasonal_type': seasonal_type, 'z_score':z_score}}, 201
        else:
            print("Wrong api key")
            
          
        
# Initialise the Correlate Class
# -----------------------------
@name_space.route('/correlate')
class Correlate(Resource):

    @api.expect(a_correlate)
    def post(self):
        """ Time series correlations

        Description:
        -----------
        This method computes pairwise correlations in a set of given time series within a sliding time window. The service takes as input a list of different time series as well as the start, the window size, the step size, the number of steps and the corresponding locale if applicable. After providing the corresponding inputs, for each step i our service computes and returns the corresponding correlation matrix.

        This implementation uses the "pandas.DataFrame.corr" from the pandas library.

        ----------

        Parameters:
        -----------

        data: List of lists
        -----
              The provided list with the time series data.

        start: Integer
        -----
            The start point of our calculations, assuming that the first timestamp of each time series is 0.

        window_size: Integer
        -----------
            The length of the window.

        step_size: Integer
        ---------
            The length of each step.

        steps: Integer
        -----
            Number of steps.
            
        correlation_method: String
        ------------------
            Methods of correlation {‘pearson’, ‘kendall’, ‘spearman’} 
            
        locale: String
        -----
            The corresponding locale or 'None' to use the default.

        api_key: String
        -----
            API user's key.

        stocks_format: Boolean
        ------
            A boolean representing whether the input data follows the stocks-specific format.

        ------

        Returns:
        -------

        correlations: Lists of lists
        ------------
            A list containing a correlation matrix for each window.
        """
        # API key check
        k = api.payload['api_key']
        if k in keys:

            # Logging request
            ns1.logger.info('- Correlate Request')
            ns1.logger.info(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            ns1.logger.info(api.payload)
            
            # Parsing data
            data = api.payload['data']
            start = api.payload['start']
            window_size = api.payload['window_size']
            step_size = api.payload['step_size']
            steps = api.payload['steps']
            correlation_method = api.payload['correlation_method']
            l = api.payload['locale']
            stocks_format = api.payload['stocks_format']

            # Iterate through the provided list with the stocks
            final_data = []
            for i in range(0, len(data)):
                curr_data = data[i]
                values = [] # closing values

                # Check whether we have a stocks format
                if stocks_format == True:
                    j = 0
                    for k in curr_data:
                        if j > 0:
                            x = k.split(',')
                            values.append(x[5])
                        j = j + 1

                    d = values
                    
                elif isinstance(curr_data, str):
                    with pysftp.Connection(host=hostname, username=username, password=password, private_key=".ppk", cnopts=cnopts) as sftp:
                        with sftp.cd(path):
                            sftp_v = True
                            with sftp.open(curr_data,) as remote_file:                            
                                j = 0 
                                for line in remote_file:
                                    if j > 0:
                                        x = line.split(",")
                                        values.append(x[2])
                                    j = j + 1

                    d = values
            
                # Locale manipulation
                if l != "None":
                    locale.setlocale(locale.LC_ALL, l)
                    d = [locale.atof(item) for item in d]
                else:
                    # Convert data into floats
                    d = np.array(d, dtype=np.float32)
                    
                final_data.append(d)
            
            correlations = correlate(final_data, start, window_size, step_size, steps,correlation_method)
            ns1.logger.info(correlations)
            return {'correlations': correlations}, 201
        else:
            print("Wrong api key")


# Initialise the buildAndStoreIndex Class
# ----------------------------------------
@name_space.route('/buildAndStoreIndex')
class BuildAndStoreIndex(Resource):

    @api.expect(a_buildAndStoreIndex)
    def post(self):
        """ Builds the index used to speed up self-join and bundle discovery.

        Description:
        -----------
        
        This function builds an index on the given co-evolving time series, which is used to execute the self-join (pair discovery) and bundle discovery. To reduce the candidate pairs that need to be checked at each timestamp during self-join, we discretize the values of all time series in bins of size ε. The time series index is essentially a tree-map (i.e., sorted keys), where, for each timestamp, we store a hash-map containing the generated bins as keys, each containing the corresponding time series values. Time series with values within the same bin at any timestamp form candidate pairs. To avoid false negatives, we need to check adjacent bins for additional candidate pairs whose values differ by at most ε. Time series having values at non-adjacent bins are certainly farther than ε at that specific timestamp, so we can avoid these checks. However, using the above procedure, the calculation of self-joins would require a new index to be built for different ε threshold values. To overcome this, we build the index using an initial ε_0 value and then, we can compute any self-join process with different ε thresholds, using the same computed and loaded index. We can do this by simply considering the following two cases during calculation: a) - ε≤ε_0: In this case, we can only find pairs in adjacent bins, so the procedure takes place as previously.b) - ε>ε_0 : In this case, we may find results also in non-adjacent bins, so, it suffices to check for candidate pairs within the next ⌈ε⁄ε_0 ⌉ bins. A trade-off of this procedure can be detected in the case that the given ε threshold is slightly larger by a multiple of ε_0, when a larger number of candidates will have to be checked. However, this ensures no false negatives and the slightly larger computation time is negligible compared to the time needed to rebuild the index for each new self-join operation with a different ε threshold value.

        ----------

        Parameters:
        -----------

        dataPath: String
        -----
            Path to a remote (e.g., on an SFTP server) or local location containing input time aligned time series

        doNormalize: String
        -----------
            Whether to apply z-normalization or not

        doPAA: String
        -----
            Whether to apply Piecewise Aggregate Approximation (PAA) or not

        doSmoothing: String
        -----------
            Whether to apply LOESS smoothing or not

        indexFile: String
        ---------
            Path to a remote index file (e.g., on an SFTP server) or local index file
            
        noSeg: String
        -----
            If PAA is applied, this is the number of segments

        smoothBandwidth: String
        ---------------
            If smoothing is applied, the percentage (from 0.0 to 1.0) of closest timestamps to be considered in LOESS
            
        api_key: String
        -----
            API user's key.

        ------
        
        """
        
        # Logging request
        ns1.logger.info('- buildAndStoreIndex Request')
        ns1.logger.info(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        ns1.logger.info(api.payload)
        
        # Parsing data
        dataPath = api.payload['dataPath']
        doNormalize = api.payload['doNormalize']
        doPAA = api.payload['doPAA']
        doSmoothing = api.payload['doSmoothing']
        indexFile = api.payload['indexFile']
        noSeg = api.payload['noSeg']
        smoothBandwidth = api.payload['smoothBandwidth']
        api_key = api.payload['api_key']  
        
        # API call
        config = dict()
        config['username'] = username
        config['password'] = password
        config['dataPath'] = dataPath
        config['indexFile'] = indexFile
        config['doPAA'] = doPAA
        config['noSeg'] = noSeg
        config['doSmoothing'] = doSmoothing
        config['smoothBandwidth'] = smoothBandwidth
        config['doNormalize'] = doNormalize

        config = json.dumps(config)
        config = json.loads(config)
        headers = {'apiKey' : api_key}
        r = requests.post('http://sdl-ts.magellan.imsi.athenarc.gr/buildAndStoreIndex', json=config, headers=headers)
        
        return r.json(), 201
    

   
# Initialise the discoverBundles Class
# -------------------------------------
@name_space.route('/discoverBundles')
class DiscoverBundles(Resource):

    @api.expect(a_discoverBundles)
    def post(self):
        """ Performs bundle discovery on the read input co-evolving time series.

        Description:
        -----------
        This function performs the bundle discovery operation on the given co-evolving time series dataset. Given the ε (value threshold), δ (duration threshold) and μ (membership threshold) parameters as input, the process calculates and returns all the groups (bundles) having at least μ locally similar time series (i.e., value difference of at most ε for at least δ timestamps), along with the corresponding similarity intervals.

        ----------

        Parameters:
        -----------

        delta: String
        -----
            The delta threshold for bundle discovery

        doStore: String
        -------
            Whether to store the results or not

        epsilon: String
        -------
            The epsilon threshold for bundle discovery

        indexFile: String
        ---------
            Path to a remote index file (e.g., on an SFTP server) or local index file
            
        mu: String
        ---
            The mu threshold for bundle discovery
            
        outputFile: String
        ----------
            Path to the file where the results will be stored

        api_key: String
        -----
            API user's key.

        ------

        Returns:
        -------
        
        Αll the groups (bundles) having at least μ locally similar time series (i.e., value difference of at most ε for at least δ timestamps), along with the corresponding similarity intervals.

  
        """
        
        # Logging request
        ns1.logger.info('- discoverBundles Request')
        ns1.logger.info(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        ns1.logger.info(api.payload)
        
        # Parsing data
        index_file = api.payload['indexFile']
        doStore = api.payload['doStore']
        output_file = api.payload['outputFile']
        epsilon = api.payload['epsilon']
        delta = api.payload['delta']
        mu = api.payload['mu']
        api_key = api.payload['api_key']
  
        
        # API call
        config = dict()
        config['username'] = username
        config['password'] = password
        config['indexFile'] = index_file
        config['doStore'] = 'true'
        config['outputFile'] = output_file
        config['epsilon'] = '0.01'
        config['delta'] = '10'
        config['mu'] = '3'

        config = json.dumps(config)
        config = json.loads(config)
        headers = {'apiKey' : api_key}
        r = requests.post('http://sdl-ts.magellan.imsi.athenarc.gr/discoverBundles', json=config, headers=headers)
        
        return r.json(), 201
        

   
# Initialise the selfJoin Class
# ------------------------------
@name_space.route('/selfJoin')
class SelfJoin(Resource):

    @api.expect(a_selfJoin)
    def post(self):
        """ Performs self-join on the read input co-evolving time series..

        Description:
        -----------
        This function performs the self-join operation on the given co-evolving time series dataset. Given the ε (value threshold) and δ (duration threshold) parameters as input, the process calculates and returns for each time series within the given dataset, all the locally similar time series (i.e., value difference of at most ε for at least δ timestamps) among the rest, along with the corresponding similarity intervals.

        ----------

        Parameters:
        -----------

        delta: String
        -----
            The delta threshold for pair discovery

        doStore: String
        -------
            Whether to store the results or not

        epsilon: String
        -------
            The epsilon threshold for bundle discovery

        indexFile: String
        ---------
            Path to a remote index file (e.g., on an SFTP server) or local index file
            
        outputFile: String
        ----------
            Path to the file where the results will be stored

        mu: String
        ---
            The mu threshold for bundle discovery
            
        api_key: String
        -----
            API user's key.

        ------

        Returns:
        -------
        
        Αll the locally similar time series along with the corresponding similarity intervals.

        """
        
        # Logging request
        ns1.logger.info('- discoverBundles Request')
        ns1.logger.info(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        ns1.logger.info(api.payload)
        
        # Parsing data
        index_file = api.payload['indexFile']
        doStore = api.payload['doStore']
        output_file = api.payload['outputFile']
        epsilon = api.payload['epsilon']
        delta = api.payload['delta']
        mu = api.payload['mu']
        api_key = api.payload['api_key']
  
        
        # API call
        config = dict()
        config['username'] = username
        config['password'] = password
        config['indexFile'] = index_file
        config['doStore'] = 'true'
        config['outputFile'] = output_file
        config['epsilon'] = '0.01'
        config['delta'] = '10'
        config['mu'] = '3'

        config = json.dumps(config)
        config = json.loads(config)
        headers = {'apiKey' : api_key}
        r = requests.post('http://sdl-ts.magellan.imsi.athenarc.gr/selfJoin', json=config, headers=headers)
        
        return r.json(), 201


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(port_num))