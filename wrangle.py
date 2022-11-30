import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

import env

###################################################
################## ACQUIRE DATA ###################
###################################################

def get_db_url(db, user=env.username, password=env.password, host=env.host):
    '''
    This function uses the imported host, username, password from env file, 
    and takes in a database name and returns the url to access that database.
    '''

    return f'mysql+pymysql://{user}:{password}@{host}/{db}' 

def new_zillow_data():
    '''
    This reads the zillow 2017 properties data from the Codeup db into a df.
    '''
    # Create SQL query.
    sql_query='''
        SELECT prop.parcelid,
            predictions_2017.logerror,
            bathroomcnt AS bathrooms,
            bedroomcnt AS bedrooms,
            calculatedfinishedsquarefeet AS sqft,
            fips,
            latitude,
            longitude,
            lotsizesquarefeet,
            yearbuilt,
            predictions_2017.logerror,
            fireplacecnt AS fireplace,
            decktypeid AS deck, 
            poolcnt AS pool, 
            garagecarcnt AS garage,
            hashottuborspa AS hottub
        FROM (
        SELECT parcelid, MAX(transactiondate) AS max_transactiondate
        FROM predictions_2017
        GROUP BY parcelid
        ) pred 
        JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                    AND pred.max_transactiondate = predictions_2017.transactiondate
        LEFT JOIN properties_2017 prop ON pred.parcelid = prop.parcelid
        LEFT JOIN propertylandusetype land USING(propertylandusetypeid)
        WHERE propertylandusedesc = "Single Family Residential"
        AND transactiondate <= '2017-12-31'
        AND prop.longitude IS NOT NULL
        AND prop.latitude IS NOT NULL
        AND yearbuilt IS NOT NULL;
        '''

    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url(db = 'zillow'))

    return df

def acquire_zillow_data(new = False):
    ''' 
    Checks to see if there is a local copy of the data, 
    if not or if new = True then go get data from Codeup database
    '''
    
    filename = 'zillow.csv'
    
    #if we don't have cached data or we want to get new data go get it from server
    if (os.path.isfile(filename) == False) or (new == True):
        df = new_zillow_data()
        #save as csv
        df.to_csv(filename,index=False)

    #else used cached data
    else:
        df = pd.read_csv(filename)
          
    return df

def give_county_names(df):
    df['county'] = df.fips.replace({6037:'LA', 6059:'Orange', 6111:'Ventura'})
    df.drop(columns='fips', inplace=True)
    return df

def create_age(df):
    df["yearbuilt"] = df["yearbuilt"].astype(int)
    df["2017_age"] = 2017 - df.yearbuilt
    df["2017_age"] = df["2017_age"].astype(int)
    df.drop(columns='yearbuilt', inplace=True)
    return df

def nulls_to_zeros(df, columns):
    '''
    Takes in df and and a list of column names and replaces nulls with 0
    returns data frame
    ''' 
    for feature in columns:
        df[feature]=df[feature].replace(r"^\s*$", np.nan, regex=True)     
        # fill optional features with 0 assumption that if it was not mark it did not exist
        df[feature] = df[feature].fillna(0)
    return df

def clean_zillow(df, features):
    df = nulls_to_zeros(df, features)
    df = give_county_names(df)
    df = create_age(df)
    df.dropna(inplace=True)
    return df

def split_data(df, test_size=0.15):
    '''
    Takes in a data frame and the train size
    It returns train, validate , and test data frames
    with validate being 0.05 bigger than test and train has the rest of the data.
    '''
    train, test = train_test_split(df, test_size = test_size , random_state=27)
    train, validate = train_test_split(train, test_size = (test_size + 0.05)/(1-test_size), random_state=27)
    
    return train, validate, test

def scale_zillow(train, validate, test):
    '''
    Takes in train, validate, test and a list of features to scale
    and scales those features.
    Returns df with new columns with scaled data
    '''
    scale_features=list(train.select_dtypes(include=np.number).columns)
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    minmax = pre.MinMaxScaler()
    minmax.fit(train[scale_features])
    
    train_scaled[scale_features] = pd.DataFrame(minmax.transform(train[scale_features]),
                                                  columns=train[scale_features].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[scale_features] = pd.DataFrame(minmax.transform(validate[scale_features]),
                                               columns=validate[scale_features].columns.values).set_index([validate.index.values])
    
    test_scaled[scale_features] = pd.DataFrame(minmax.transform(test[scale_features]),
                                                 columns=test[scale_features].columns.values).set_index([test.index.values])
    
    return train_scaled, validate_scaled, test_scaled