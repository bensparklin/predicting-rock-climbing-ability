#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:31:17 2022

@author: Ben
"""

#process and clean data using numpy, pandas
#build model 
#visualize using matplotlib/seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data
ascents = pd.read_csv('ascents.csv')
grades = pd.read_csv('grades.csv')
method = pd.read_csv('method.csv')
user = pd.read_csv('user.csv')


#exploring the ascents data - first 10 rows
ascents.head(10)

#ascents data types
ascents.dtypes

#asecnts info
ascents.info

#ascents shape - (4111877, 28)
ascents.shape

#mean,mean,min max for each column
ascents.describe() 

#exploring the user data - first 10 rows
user.head(10)

#user data types
user.dtypes

#user info
user.info

#user shape - (62593, 23)
user.shape

#mean,mean,min max for each column
user.describe() 


#clean the data

#how many unique users? 62593
user['id'].nunique()
#each row is a unique user 62593 climbers

#restrict to have those that have a birth date since age will be an important
#feature I use
user_clean = user[user['birth'].notna()]
user_clean['birth'].describe()
#we have users who are over 100 years, most likley a typo so 
#will restrict to only have birth years after 1940
#(maybe restrict users by age at climb?)
user_clean = user_clean[user_clean['birth'] > '1940-01-01']
user_clean['birth'].max()

#compare male and female
user_clean.groupby('sex').size()
#there are many more males than females in this dataset

#clean weight data
plt.hist(user_clean['weight'])

#weight is unlikley to be less than 20 kg
user_clean = user_clean[user_clean['weight'] > 20]
plt.hist(user_clean['weight'])
user_clean['weight'].describe()
#looks better

#clean height data
plt.hist(user_clean['height'])
user_clean['height'].describe()

#to remove the potential for any data entry errors I will restrict height 
#to be between filter 4-7.5 feet 
user_clean = user_clean[user_clean['height'].isin(range(120,228))]
#convert from centimeters to meters
user_clean['height'] = user_clean['height']/100

#calculate bmi for users
plt.hist(user_clean['height'])
user_clean['height'].describe()

user_clean['bmi'] = user_clean['weight']/(user_clean['height']**2)

user_clean['bmi'].describe()
plt.hist(user_clean['bmi'])
#some BMI's appear to be unlikely, so I will filter BMIs to be between 15-40
user_clean = user_clean[user_clean['bmi'].isin(range(15,40))]
user_clean['bmi'].describe()
plt.hist(user_clean['bmi'])

#define years of climbing experience


#compare number of boulders and sport climbs
ascents.groupby(['climb_type']).size()
#many more sport climbs than bouldering, but I will focus on bouldering data
#(1.2 million boulders logged)

#filter to only include bouldering
ascents_bouldering = ascents.loc[ascents['climb_type'] == 1]







