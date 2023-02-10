#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:31:17 2022

@author: Ben
"""

#process and clean data using numpy, pandas
#build model 
#visualize using matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#pd.set_option('display.max_rows', 10)
pd.set_option('display.max_rows', None)


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
#we have users who are over 100 years old, most likley a typo so 
#will restrict to only have birth years after 1940
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
user_clean = user_clean[user_clean['height'] > 120]
user_clean = user_clean[user_clean['height'] < 228]
#convert from centimeters to meters
user_clean['height'] = user_clean['height']/100

#calculate bmi for users
plt.hist(user_clean['height'])
user_clean['height'].describe()

user_clean['bmi'] = user_clean['weight']/(user_clean['height']**2)

user_clean['bmi'].describe()
plt.hist(user_clean['bmi'])
#some BMI's appear to be unlikely, so I will filter BMIs to be between 15-40
user_clean = user_clean[user_clean['bmi'] > 15]
user_clean = user_clean[user_clean['bmi'] < 40]
user_clean['bmi'].describe()
plt.hist(user_clean['bmi'])


#compare number of boulders and sport climbs
ascents.groupby(['climb_type']).size()
#many more sport climbs than bouldering, but I will focus on bouldering data
#since that is what I'm more familiar with
#(1.2 million boulders logged)

#filter to only include bouldering
ascents_bouldering = ascents.loc[ascents['climb_type'] == 1]

#merge users and ascents
user_ascent = pd.merge(user_clean, ascents_bouldering, 
                       left_on='id', right_on='user_id', how='inner', 
                       suffixes=("_user", "_ascents"))

user_ascent.dtypes
user_ascent['id_user'].nunique()
user_ascent['id_ascents'].nunique()
#after merging we have 9470 users with 739583 climbs

#number of logged ascents per user
user_ascent['logged_ascents'] = user_ascent.groupby('user_id').size()
plt.hist(user_ascent['logged_ascents'])

#age at climb = ascent date - birth date date,this will be different at
#each ascent

#remove ascents with years that are before 1900
user_ascent['year'].describe()
user_ascent = user_ascent[user_ascent['year'] > 1900]
user_ascent['year'].describe()

#convert to pandas datetime to access birth year
user_ascent['birth'] = pd.to_datetime(user_ascent['birth'])
user_ascent['birth_year'] = user_ascent['birth'].dt.year

user_ascent['ascent_age'] = user_ascent['year'] - user_ascent['birth_year']
user_ascent['ascent_age'].describe()

#several climbers completed climbs before they were born, 
#so I will remove any rows where ascent age is less than 5
user_ascent[['id_user','birth_year', 'year', 'ascent_age']].sort_values('ascent_age')
user_ascent = user_ascent[user_ascent['ascent_age'] > 5]
user_ascent[['id_user','birth_year', 'year', 'ascent_age']].sort_values('ascent_age')
plt.hist(user_ascent['ascent_age'])


#years of experience = ascent date - started, this will be different at each 
#ascent
user_ascent['started'].describe()
#remove users that started climbing before 1900
user_ascent = user_ascent[user_ascent['started'] > 1900]
user_ascent['started'].describe()

user_ascent['years_exp'] = user_ascent['year'] - user_ascent['started']
user_ascent['years_exp'].describe()
#remove users with negative climbing experience
user_ascent = user_ascent[user_ascent['years_exp'] > 0]
user_ascent['years_exp'].describe()
plt.hist(user_ascent['years_exp'])

#remove users where years of experience are greater than ascent age
user_ascent[['id_user','birth_year', 'year', 'ascent_age', 'years_exp']].sort_values('ascent_age')
user_ascent = user_ascent[user_ascent['years_exp'] < user_ascent['ascent_age']]

#merge in grades based on grade id from grades
user_ascent_grade = pd.merge(user_ascent, grades[['id', 'usa_boulders']], left_on='grade_id',
                             right_on='id', how='inner', suffixes=('_ascents', '_grades'))

user_ascent_grade[['user_id', 'id_ascents', 'id', 'usa_boulders']].head(50)
user_ascent_grade.dtypes

#merge in methods based on method id from methods
user_ascent_grade_method = pd.merge(user_ascent_grade, 
                                    method[['id', 'name']], left_on='method_id',
                                    right_on='id', how='inner', suffixes=('_grades','_methods'))

#check results of merge
user_ascent_grade_method[['user_id', 'id_ascents', 'id_grades', 'usa_boulders',
                          'id_methods', 'name_methods']]
user_ascent_grade_method.dtypes

#remove na from grades
user_ascent_grade_method = user_ascent_grade_method[user_ascent_grade_method['usa_boulders'].notna()]


#convert grades with multiple to be one grade, using the lower grade
user_ascent_grade_method['usa_boulders_new'] = np.where(user_ascent_grade_method['usa_boulders'] == 'V0-',
             'VB', np.where(user_ascent_grade_method['usa_boulders'] == 'V3/4',
                      'V3', np.where(user_ascent_grade_method['usa_boulders'] == 'V4/V5',
                               'V4', np.where(user_ascent_grade_method['usa_boulders'] == 'V5/V6',
                                        'V5', np.where(user_ascent_grade_method['usa_boulders'] == 'V8/9',
                                                 'V8', user_ascent_grade_method['usa_boulders'])))))

#create numeric version of grades
user_ascent_grade_method['usa_boulders_numeric'] = user_ascent_grade_method['usa_boulders_new'].str.replace("V", "")
user_ascent_grade_method['usa_boulders_numeric'] = user_ascent_grade_method['usa_boulders_numeric'].str.replace("B", "-1")
user_ascent_grade_method['usa_boulders_numeric']  = user_ascent_grade_method['usa_boulders_numeric'].astype(int)


user_ascent_grade_method['usa_boulders'].unique()
user_ascent_grade_method['usa_boulders_new'].unique()
user_ascent_grade_method['usa_boulders_numeric'].unique()


#set order of grades
grades_order = ["VB", "V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", 
                "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17"]


#how many ascents for each grade?
grades_summary = pd.DataFrame(user_ascent_grade_method.groupby('usa_boulders_new').size())
#move rownames into column 
grades_summary.index.name = 'usa_boulders_new'
grades_summary.reset_index(inplace=True)
#plot histogram
grades_summary.set_index('usa_boulders_new').loc[grades_order].plot(kind="bar")


#how are ascents completed?
methods_summary = user_ascent_grade_method.groupby('name_methods').size()
methods_summary.plot(kind='bar')



#1. identify max grade completed per user
user_ascent_grade_method['max_grade'] = user_ascent_grade_method.groupby('user_id')['usa_boulders_numeric'].max()
plt.hist(user_ascent_grade_method.groupby('user_id')['usa_boulders_numeric'].max(), 
         color='skyblue', edgecolor='k')
plt.xlabel('Max grade')
plt.ylabel('Number of climbers')
plt.show()

#what countries are the climbers from?
countries =  pd.DataFrame(user_ascent_grade_method.drop_duplicates(subset=['id_user']).groupby('country_user').size())
#move rownames into column 
countries.index.name = 'country_user'
countries.reset_index(inplace=True)
#rename second column
countries = countries.rename(columns={countries.columns[1]: 'number_of_users'})

#combine countries with less than 10 into "other" category
countries['country_user'] = np.where(countries['number_of_users'] < 50, 'Other', countries['country_user'])
countries = countries.groupby('country_user')['number_of_users'].sum().sort_values(ascending=False)
cmap = plt.cm.tab10
colors = cmap(np.arange(len(countries)) % cmap.N)
countries.plot(kind='bar', color=colors, xlabel='Country', ylabel='Number of users' )

#plot correlation between features (bmi vs max, years exp vs max)
no_duplicate_users = user_ascent_grade_method.drop_duplicates(subset=['id_user'])
no_duplicate_users.dtypes
plt.scatter(no_duplicate_users['bmi'], no_duplicate_users['max_grade'], alpha=0.5)
plt.scatter(no_duplicate_users['bmi'], no_duplicate_users['years_exp'], alpha=0.5)
plt.scatter(no_duplicate_users['max_grade'], no_duplicate_users['ascent_age'], alpha=0.5)
plt.scatter(no_duplicate_users['logged_ascents'], no_duplicate_users['max_grade'], alpha=0.5)



#ascent comments
#what are the most and least popular routes, using sentiment analysis about ascent difficulty
user_ascent_grade_method[user_ascent_grade_method['comment'].notna()]['comment'].head(1000)




#max flash - completed first try, but have seen others do it or were told how to do it

#max redpoint per user - lead climb after having practiced

#max onsight per user - flashed without seeing any else do it

#calculate progression time per grade per user

