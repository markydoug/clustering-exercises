import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

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
        SELECT *
        FROM predictions_2017 
        LEFT JOIN properties_2017 USING (parcelid)
        LEFT JOIN airconditioningtype USING (airconditioningtypeid)
        LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN buildingclasstype USING (buildingclasstypeid)
        LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN storytype USING (storytypeid)
        LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
        WHERE YEAR(transactiondate) = 2017
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL;
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

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    percent_missing = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': percent_missing})
    
    return cols_missing.sort_values(by='num_rows_missing', ascending=False)

def handle_missing_values(df, prop_required_columns=0.5, prop_required_rows=0.75):
    column_threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=column_threshold)
    row_threshold = int(round(prop_required_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=row_threshold)
    return df

def drop_id_columns(df):
    ids_to_drop = ['id','id.1','airconditioningtypeid','architecturalstyletypeid','buildingclasstypeid','heatingorsystemtypeid','propertylandusetypeid','storytypeid','typeconstructiontypeid']
    df.drop(columns=ids_to_drop, inplace=True)
    return df

def keep_single_unit_properties(df):
    df = df[(df.propertylandusedesc=='Manufactured, Modular, Prefabricated Homes') | 
        (df.propertylandusedesc=='Single Family Residential') |
        (df.propertylandusedesc=='Condominium') |
        (df.propertylandusedesc=='Cluster Home') |
        (df.propertylandusedesc=='Mobile Home') |
        (df.propertylandusedesc=='Townhouse')]
    return df

def drop_dup_parcelids(df):
    dups = df[df.duplicated(subset='parcelid', keep='last')].index
    df.drop(dups, inplace=True)
    return df

def wrangle_zillow_2():
    df = acquire_zillow_data()
    df = drop_dup_parcelids(df)
    df = drop_id_columns(df)
    df = keep_single_unit_properties(df)
    df = handle_missing_values(df)
    return df