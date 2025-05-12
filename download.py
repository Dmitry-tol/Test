import requests
from urllib.parse import urlencode
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator 
from pathlib import Path
import os
from datetime import timedelta
from train_model import train

def download_data():
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/3YBohlrOKEtkkg'   
 
    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']
 
    download_response = requests.get(download_url)
    with open('downloaded_file.csv', 'wb') as f:    
        f.write(download_response.content)
     

def clear_data():
    taxiDB = pd.read_csv('downloaded_file.csv', index_col=0)
    taxiDB['pickup_datetime'] = pd.to_datetime(taxiDB['pickup_datetime'])
    taxiDB['dropoff_datetime'] = pd.to_datetime(taxiDB['dropoff_datetime'])
    taxiDB['trip_duration'] = (taxiDB['dropoff_datetime'] - taxiDB['pickup_datetime']).dt.total_seconds()

    taxiDB = taxiDB.drop('dropoff_datetime', axis=1)
    taxiDB['vendor_id'] = taxiDB['vendor_id'] - 1
 
    taxiDB['store_and_fwd_flag'] = taxiDB['store_and_fwd_flag'].apply(lambda x: 0 if x=='N' else 1)
    taxiDB['pickup_datetime'] = taxiDB['pickup_datetime'].astype(str)
    allLat  = list(taxiDB['pickup_latitude']) + list(taxiDB['dropoff_latitude'])
    medianLat  = sorted(allLat)[int(len(allLat)/2)]

    latMultiplier  = 111.32

    taxiDB['pickup_latitude']   = latMultiplier  * (taxiDB['pickup_latitude']   - medianLat)
    taxiDB['dropoff_latitude']   = latMultiplier  * (taxiDB['dropoff_latitude']  - medianLat)
    allLong = list(taxiDB['pickup_longitude']) + list(taxiDB['dropoff_longitude'])

    medianLong  = sorted(allLong)[int(len(allLong)/2)]

    longMultiplier = np.cos(medianLat*(np.pi/180.0)) * 111.32
 
    taxiDB['pickup_longitude']   = longMultiplier  * (taxiDB['pickup_longitude']   - medianLong)
    taxiDB['dropoff_longitude']   = longMultiplier  * (taxiDB['dropoff_longitude']  - medianLong)

 
    taxiDB['distance_km'] = ((taxiDB['dropoff_longitude'] -taxiDB['pickup_longitude'])**2 + (taxiDB['dropoff_latitude'] - taxiDB['pickup_latitude'])**2)**0.5

    taxiDB = taxiDB.drop(['pickup_longitude', 'dropoff_longitude',
                      'pickup_latitude', 'dropoff_latitude'], axis=1)

 
    taxiDB['passenger_count'] = taxiDB['passenger_count'].map(taxiDB.groupby(['passenger_count'])['trip_duration'].mean())


    taxiDB = taxiDB.rename({'passenger_count':'category_encoded'}, axis=1)

    result_columns = [
    
        'vendor_id',
        'pickup_datetime',
        'category_encoded',
        'store_and_fwd_flag',
        'trip_duration',
        'distance_km'
    ]

    taxiDB = taxiDB[result_columns]

    taxiDB['pickup_datetime'] = pd.to_datetime(taxiDB['pickup_datetime'])
    
     
    taxiDB.to_csv('df_clear.csv')
    return True

dag1 = DAG(
    dag_id="my_project",
    start_date=datetime(2025, 15, 12),
    concurrency=4,
    schedule_interval=timedelta(minutes=5),
#    schedule="@hourly",
    max_active_runs=1,
    catchup=False,
)
download_task = PythonOperator(python_callable=download_data, task_id = "download_data", dag = dag1)
clear_task = PythonOperator(python_callable=clear_data, task_id = "clear_data", dag = dag1)
train_task = PythonOperator(python_callable=train, task_id = "train_data", dag = dag1)
download_task >> clear_task >> train_task
