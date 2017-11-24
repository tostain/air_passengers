import pandas as pd
import os
from pandas.tseries.holiday import Holiday, USMemorialDay, AbstractHolidayCalendar, nearest_workday, MO
from copy import deepcopy

# Define a mapped version of the Haversine formula to compute distances between airports. 
# Inspired from https://stackoverflow.com
#    /questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
from math import radians, cos, sin, asin, sqrt

def haversine(row):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # we map the rows
    lon1 = row['D_lon']
    lat1 = row['D_lat']
    lon2 = row['A_lon']
    lat2 = row['A_lat']
    
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
        
        # Splits the external dataset in two subsets
        ext_coords = ext_data.get(['Airport', 'Latitude', 'Longitude'])
        ext_census = ext_data.drop(['Latitude', 'Longitude'], axis=1)
        
        # Merges (left join) census data with base data for the departure point
        ext_census_d = deepcopy(ext_census)
        ext_census_d.columns = "D_" + ext_census_d.columns
        X_encoded = pd.merge(
            X_encoded, ext_census_d,
            how='left',
            left_on=['Departure'],
            right_on=['D_Airport'],
            sort=False)
        X_encoded = X_encoded.drop(['D_Airport'], axis=1)
        
        # Merges (left join) census data with base data for the arrival point
        ext_census_a = deepcopy(ext_census)
        ext_census_a.columns = "A_" + ext_census_a.columns
        X_encoded = pd.merge(
            X_encoded, ext_census_a,
            how='left',
            left_on=['Arrival'],
            right_on=['A_Airport'],
            sort=False)
        X_encoded = X_encoded.drop(['A_Airport'], axis=1)
        
        # Performs a first merge to import departure coordinates.
        X_encoded = pd.merge(
            X_encoded, ext_coords,
            how='left',
            left_on=['Departure'],
            right_on=['Airport'],
            sort=False)
        X_encoded = X_encoded.drop(['Airport'], axis=1)

        X_encoded = X_encoded.rename(
            columns={'Latitude': 'D_lat', 'Longitude': 'D_lon'})

        # We perform a second merge to import arrival coordinates.
        X_encoded = pd.merge(
            X_encoded, ext_coords,
            how='left',
            left_on=['Arrival'],
            right_on=['Airport'],
            sort=False)
        X_encoded = X_encoded.drop(['Airport'], axis=1)

        X_encoded = X_encoded.rename(
            columns={'Latitude': 'A_lat', 'Longitude': 'A_lon'})

        # And we apply the Haversine formula.
        X_encoded['Distance'] = X_encoded.apply(lambda row: haversine(row), axis=1)
        
        # Creates one hot encoding for Departure, then drop the original feature
        X_encoded = X_encoded.join(pd.get_dummies(
            X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        
        # Creates one hot encoding for Arrival, then drop the original feature
        X_encoded = X_encoded.join(pd.get_dummies(
            X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Arrival', axis=1)
        
        # Adds the Federal Holidays
        cal = pd.tseries.holiday.USFederalHolidayCalendar()
        holiday_dates = cal.holidays(min(X_encoded['DateOfDeparture']), max(X_encoded['DateOfDeparture']))
        X_encoded['Holiday'] = X_encoded['DateOfDeparture'].isin(holiday_dates)
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Holiday'], prefix='hd'))
        X_encoded = X_encoded.drop(['Holiday'], axis=1)
        
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
        
        # Return the values
        X_array = X_encoded.values
        return X_array