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
from impl import *
from config import *


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
            description='The start point of our predictions'), 'end': fields.Integer(
                description='The end point of our predictions'), 'forecast_range': fields.Integer(
                    description='The number of points to predict at each step'), 'period': fields.List(fields.Integer(
                        description='The seasonal parameter or set of seasonal parameters')), 'data': fields.List(
                            fields.String(
                                description='The provided list with the time series data or string which represents the path for the data')), 'locale': fields.String(
                                    description='The corresponding locale, None if it does not exist', default='None', required=False), 'api_key': fields.String(
                                        description='API key')})


# Set up the structure of the model for the Decompose Method
# ----------------------------------------------------------
a_timeserie = api.model(
    'Decompose input values', {
        'data': fields.List(
            fields.String(
                description='The provided list with the time series data or string which represents the path for the data')), 'model': fields.String(
                    description='additive or multiplicative'), 'period': fields.Integer(
                        description='The seasonal parameter'), 'locale': fields.String(
                            description='The corresponding locale, None if it does not exist', default='None', required=False), 'api_key': fields.String(
                                description='API key')})


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

        'locale': fields.String(
            description='The corresponding locale, None if it does not exist', default='None', required=False),


        'api_key': fields.String(
            description='API key')})


# Set up logging system
# ---------------------
ns1 = api.namespace('api/v1', description='test')
fh = logging.FileHandler("v1.log")
ns1.logger.addHandler(fh)


# Initialise the Decompose Class
# ------------------------------
@name_space.route('/decompose')
class Decompose(Resource):

    @api.expect(a_timeserie)
    def post(self):
        """ Trend - Sesonality - Residual Time Series Decomposition.

        Description:
        -----------
        This method performs the Triple Time Series Decomposition. The service takes as input a time series, the corresponding model type ("Multiplicative" or "Additive"), the period and the corresponding locale if applicable.The user could select to insert the path of the selected data or provide them in an array form. The provided time series is decomposed into three distinct components according to the selected model and period:

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

        model: String
        -----
            'multiplicative' for multiplicative model, 'additive' for additive model.

        period: Integer
        ------
            The seasonal parameter.

        locale: String
        -----
            The corresponding locale or 'None' to use the default.

        api_key: String
        -----
            API user's key.

        ------

        Returns:
        -------

        trend: list
        -----
            A list containing the data of the trend component.

        seasonality: list
        -----------
            A list containing the data of the seasonal component.

        residual error: list
        ---------------
            A list containing the data of the residual error component.

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
            m = api.payload['model']
            p = api.payload['period']
            l = api.payload['locale']

            # Check whether we got a sftp request or raw data
            if isinstance(data, str):
                print('sftp')
                with pysftp.Connection(host=hostname, username=username, password=password, private_key=".ppk", cnopts=cnopts) as sftp:
                    with sftp.cd(path):
                        with sftp.open(data) as remote_file:
                            i = 0
                            data = []
                            for line in remote_file:
                                if i > 0:
                                    x = line.split(",")
                                    data.append(x[5])
                                i = i + 1

            # Locale Manipulation
            if l != "None":
                locale.setlocale(locale.LC_ALL, l)
                data = [locale.atof(item) for item in data]
            else:
                # Convert data into floats
                data = np.array(data, dtype=np.float32)
            # Time Series Decomposition into Trend Seasonality and Residual
            # Error
            result = seasonal_decompose(data, period=p, model=m)

            # Replace NaN values with None
            trend = np.where(np.isnan(result.trend), None, result.trend)
            seasonality = np.where(
                np.isnan(
                    result.seasonal),
                None,
                result.seasonal)
            res_error = np.where(np.isnan(result.resid), None, result.resid)
            return {'trend': list(trend), 'seasonality': list(
                seasonality), 'residual error': list(res_error)}, 201
        else:
            print("Wrong api key")


# Initialise the Catalog Class
# ------------------------------
@name_space.route('/catalog')
class Catalog(Resource):

    def get(self):
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
                    available_data.append(attr.filename)

        return {'available data': available_data}, 201


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

            # Check whether we got a sftp request or raw data
            if isinstance(data, str):
                print('sftp')
                with pysftp.Connection(host=hostname, username=username, password=password, private_key=".ppk", cnopts=cnopts) as sftp:
                    with sftp.cd(path):
                        with sftp.open(data) as remote_file:
                            i = 0
                            data = []
                            for line in remote_file:
                                if i > 0:
                                    x = line.split(",")
                                    data.append(x[5])
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
            return {'predictions': predictions, 'best_configurations': {'seasonality': seasonality,
                                                                        'trend_type': trend_type, 'seasonal_type': seasonal_type, 'z_score': z_score}}, 201
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

        ------

        Returns:
        -------

        correlations: Lists of lists
        ------------
            A list containing the correlation matrices.
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

            final_data = []
            for d in data:
                # Check whether we got a sftp request or raw data
                dt = []
                if isinstance(d, str):
                    print('sftp')
                    with pysftp.Connection(host=hostname, username=username, password=password, private_key=".ppk", cnopts=cnopts) as sftp:
                        with sftp.cd(path):
                            with sftp.open(d) as remote_file:
                                i = 0
                                for line in remote_file:
                                    if i > 0:
                                        x = line.split(",")
                                        dt.append(x[5])
                                    i = i + 1
                    d = dt

                # Locale manipulation
                if l != "None":
                    locale.setlocale(locale.LC_ALL, l)
                    d = [locale.atof(item) for item in d]
                else:
                    # Convert data into floats
                    d = np.array(d, dtype=np.float32)

                final_data.append(d)

            correlations = correlate(
                final_data,
                start,
                window_size,
                step_size,
                steps,
                correlation_method)
            return {'correlations': correlations}, 201
        else:
            print("Wrong api key")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(port_num))
