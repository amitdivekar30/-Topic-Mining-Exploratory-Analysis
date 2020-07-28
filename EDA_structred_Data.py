# Exploratory Data Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('structred_data.csv')
dataset.head()
dataset.columns

df = dataset.iloc[:, 1: ]
df.columns

# removing duplicate data

country= df['Country_Name'].value_counts()
region= df['Region'].value_counts()
city= df['City'].value_counts()

df.describe()
df.dtypes

from datetime import datetime,time

date= []
time= []
for i in range(0, 127826):
    a=i
    aa = df.timestamp[i][0:10]
    date.append(aa)
    ab = df.timestamp[i][11:-1]
    time.append(ab)

date.append(df.timestamp[125825][0:10])
time.append(df.timestamp[125825][11:-1])

# Creating a Date column to store the actual Date format for the given Month column
df["Date"] = date
df["time"] = time


df["Date"]= pd.to_datetime(df.Date)
df["time"]= pd.to_datetime(df.time)



df["Date1"] = pd.to_datetime(df.Date,format="%b-%y")

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

df["month"] = df.Date1.dt.strftime("%b") # month extraction
df["Day"] = df.Date1.dt.strftime("%d") # Day extraction
df["wkday"] = df.Date1.dt.strftime("%A") # weekday extraction
df["year"] = df.Date1.dt.strftime("%Y") # year extraction

df.columns

new_df= df[['Date', 'time', 'Date1', 'month','Day', 'wkday', 'year', 'Unread','IP', 
            'Country_Code','Country_Name', 'Region', 'City','User_Agent', 
            'Platform', 'Browser']]

new_df.to_csv('cleaned_structred_data.csv')
