import pandas as pd
import os
from math import radians, cos, sin, asin, sqrt

# Inspired from https://stackoverflow.com
#    /questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


# Inspired form the feature extractor that comes with starting_kit.
class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df

        
        # Fetches external data from external_data.csv
        path = os.path.dirname(__file__)
        ext_data = pd.read_csv(os.path.join(path, 'external_data.csv'))
        X_ext_data = ext_data[['Date', 'AirPort', 'Max TemperatureC',
                               'PaidHoliday', 'FederalHoliday', 'Latitude', 'Longitude']]
        
        # Merges (left join) fetched external data with base data
        X_encoded = pd.merge(
            X_encoded, X_ext_data,
            how='left',
            left_on=['DateOfDeparture', 'Arrival'],
            right_on=['Date', 'AirPort'],
            sort=False)
        X_encoded = X_encoded.rename(
            columns={'Latitude': 'A_lat', 'Longitude': 'A_lon'})
        
        # Merges (left join) data relevant to departure airport coordinates
        X_ext_data_short = X_ext_data[['Date', 'AirPort', 'Latitude', 'Longitude']]
        X_encoded = pd.merge(
            X_encoded, X_ext_data_short,
            how='left',
            left_on=['DateOfDeparture', 'Departure'],
            right_on=['Date', 'AirPort'],
            sort=False)
        X_encoded = X_encoded.rename(
            columns={'Latitude': 'D_lat', 'Longitude': 'D_lon'})
        
        # Creates one hot encoding for Departure, then drop the original feature
        X_encoded = X_encoded.join(pd.get_dummies(
            X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        
        # Creates one hot encoding for Arrival, then drop the original feature
        X_encoded = X_encoded.join(pd.get_dummies(
            X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Arrival', axis=1)
        
        #FIXME : for some reason, these treatment raise an error in the regressor like this:
        #ValueError: Number of features of the model must match the input.
        #Model n_features is 137 and input n_features is 135
        #
        # Creates one hot encoding for meteo Events, then drop the original feature
        #X_encoded['Events'] = X_encoded['Events'].fillna('None')
        #X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Events'], prefix='mevent'))
        #X_encoded = X_encoded.drop('Events', axis=1)
        #
        # Creates one hot encoding for Holiday Events, then drop the original feature
        #X_encoded['Event'] = X_encoded['Event'].fillna('Ordinary')
        #X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Event'], prefix='h'))
        #X_encoded = X_encoded.drop('Event', axis=1)
        
        # Creates one hot encoding for time period likely to catch seasonality
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.drop('weekday', axis=1)
        
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
        X_encoded = X_encoded.drop('week', axis=1)
        
        # Drops DateOfDeparture
        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
        
        # Computes the distances thanks to Haversine formula
        X_encoded['Distances'] = haversine(X_encoded.D_lon,X_encoded.D_lat,X_encoded.A_lon,X_encoded.A_lat)
        
        # Return the values
        X_array = X_encoded.values
        return X_array