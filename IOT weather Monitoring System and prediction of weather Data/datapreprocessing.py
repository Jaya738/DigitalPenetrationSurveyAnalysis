#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 08:37:51 2018

@author: srikanth
"""

#http://api.wunderground.com/api/API_KEY/history_YYYYMMDD/q/STATE/CITY.json 
from datetime import datetime, timedelta  
import time  
from collections import namedtuple  
import pandas as pd  
import requests  
#import matplotlib.pyplot as plt 
records = [] 
def derive_nth_day_feature(df, feature, N):  
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements
def extract_weather_data(url, api_key, target_date, days):  
    records = []
    for _ in range(days):
        request = BASE_URL.format(API_KEY, target_date.strftime('%Y%m%d'))
        response = requests.get(request)
        if response.status_code == 200:
            data = response.json()['history']['dailysummary'][0]
            records.append(DailySummary(
                date=target_date,
                meantempm=data['meantempm'],
                meandewptm=data['meandewptm'],
                meanpressurem=data['meanpressurem'],
                maxhumidity=data['maxhumidity'],
                minhumidity=data['minhumidity'],
                maxtempm=data['maxtempm'],
                mintempm=data['mintempm'],
                maxdewptm=data['maxdewptm'],
                mindewptm=data['mindewptm'],
                maxpressurem=data['maxpressurem'],
                minpressurem=data['minpressurem'],
                precipm=data['precipm']))
        time.sleep(6)
        target_date += timedelta(days=1)
    return records
API_KEY = '7052ad35e3c73564'  
BASE_URL = "http://api.wunderground.com/api/{}/history_{}/q/NE/Lincoln.json"   
target_date = datetime(2016, 5, 16)  
features = ["date", "meantempm", "meandewptm", "meanpressurem", "maxhumidity", "minhumidity", "maxtempm",  
            "mintempm", "maxdewptm", "mindewptm", "maxpressurem", "minpressurem", "precipm"]
DailySummary = namedtuple("DailySummary", features)  

records += extract_weather_data(BASE_URL, API_KEY, target_date, 500)  
df = pd.DataFrame(records, columns=features).set_index('date')  
tmp = df[['meantempm', 'meandewptm']].head(10)  

N = 1
feature = 'meantempm'
rows = tmp.shape[0]
nth_prior_measurements = [None]*N + [tmp[feature][i-N] for i in range(N, rows)]
col_name = "{}_{}".format(feature, N)  
tmp[col_name] = nth_prior_measurements  

for feature in features:  
    if feature != 'date':
        for N in range(1, 4):
            derive_nth_day_feature(df, feature, N)

df = df.apply(pd.to_numeric, errors='coerce')  
spread = df.describe().T
IQR = spread['75%'] - spread['25%']
spread['outliers'] = (spread['min']<(spread['25%']-(3*IQR)))|(spread['max'] > (spread['75%']+3*IQR))
spread.ix[spread.outliers,]  
for precip_col in ['precipm_1', 'precipm_2', 'precipm_3']:  
    # create a boolean array of values representing nans
    missing_vals = pd.isnull(df[precip_col])
    df[precip_col][missing_vals] = 0
df = df.dropna()  