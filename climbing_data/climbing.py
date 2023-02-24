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
from matplotlib import pyplot

import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegressionfrom 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier





pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)

#pd.set_option('display.max_rows', None)


#import data
ascents = pd.read_csv('ascents.csv')
grades = pd.read_csv('grades.csv')
method = pd.read_csv('method.csv')
user = pd.read_csv('user.csv')


#exploring the ascents data - first 10 rows
ascents.head(10)

#ascents data types
ascents.dtypes

#ascnts info
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
#after merging we have 9469 users with 739554 climbs

#number of logged ascents per user
user_ascent['ascent_count'] = user_ascent.groupby('id_user')['id_user'].transform('count')
plt.hist(user_ascent['ascent_count'])

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

#identify max grade completed per user
user_ascent_grade_method['max_grade'] = user_ascent_grade_method['usa_boulders_numeric'].groupby(user_ascent_grade_method['id_user']).transform('max')
user_ascent_grade_method['max_grade_combined'] = np.where(user_ascent_grade_method['max_grade'].isin([-1,0,1,2,3,4,5,6,7]), 'Low', 'High')

#check it worked
user_ascent_grade_method[['max_grade', 'max_grade_combined']].head(10)

#identify years of exp when climber completed the max grade
#if there is a tie for a user by max grade, select the ascent that was completed first
user_ascent_grade_method['years_exp_max_grade']=user_ascent_grade_method.sort_values(by=['usa_boulders_numeric', 'years_exp'],ascending=[False, True]).groupby('id_user')['years_exp'].transform('first')

##identify age when climber completed the max grade
user_ascent_grade_method['age_max_grade']=user_ascent_grade_method.sort_values(by=['usa_boulders_numeric', 'years_exp'],ascending=[False, True]).groupby('id_user')['ascent_age'].transform('first')

#check number ascents at time max grade was completed
user_ascent_grade_method['ascents_max_grade'] = user_ascent_grade_method[user_ascent_grade_method['years_exp'] <= user_ascent_grade_method['years_exp_max_grade']].groupby('id_user')['id_user'].transform('count')

#calculate number of redpoints completed at time of max grade
user_ascent_grade_method[['max_grade', 'years_exp', 'years_exp_max_grade', 'ascents_max_grade', 'usa_boulders_numeric']]
                     
user_ascent_grade_method[user_ascent_grade_method["user_id"] == 3][['usa_boulders_numeric', 'max_grade', 'years_exp', 'name_methods']]


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
user_ascent_grade_method = user_ascent_grade_method[user_ascent_grade_method['ascents_max_grade'].notna()]
no_duplicate_users = user_ascent_grade_method.drop_duplicates(subset=['id_user'])

no_duplicate_users.dtypes
plt.scatter(no_duplicate_users['bmi'], no_duplicate_users['max_grade'], alpha=0.5)
plt.scatter(no_duplicate_users['bmi'], no_duplicate_users['years_exp_max_grade'], alpha=0.5)
plt.scatter(no_duplicate_users['years_exp_max_grade'], no_duplicate_users['ascent_age'], alpha=0.5)
plt.scatter(no_duplicate_users['ascent_count'], no_duplicate_users['max_grade'], alpha=0.5)
plt.scatter(no_duplicate_users['ascent_count'], no_duplicate_users['bmi'], alpha=0.5)
plt.scatter(no_duplicate_users['age_max_grade'], no_duplicate_users['bmi'], alpha=0.5)
plt.scatter(no_duplicate_users['age_max_grade'], no_duplicate_users['years_exp_max_grade'], alpha=0.5)

corr_matrix = np.corrcoef(no_duplicate_users['age_max_grade'], no_duplicate_users['years_exp_max_grade'])
corr = corr_matrix[0,1]
R_sq = corr**2


#predict the maximum boulder grade one can complete based on sex, years of experience, bmi, number of logged ascents, and progression time per grade? 
input_df = no_duplicate_users[['id_user', 'sex', 'bmi', 'years_exp_max_grade', 'age_max_grade', 'ascents_max_grade', 'max_grade_combined']]

max_grade_summary = pd.DataFrame(input_df.groupby('max_grade_combined').size())
max_grade_summary.index.name = 'max_grade_combined'
max_grade_summary.reset_index(inplace=True)
max_grade_summary.set_index('max_grade_combined').plot(kind='bar',xlabel='Max Grade', ylabel='Number of climbers', color='skyblue', edgecolor='k' )

x =input_df[['sex', 'bmi', 'years_exp_max_grade', 'ascents_max_grade', 'age_max_grade']]
y= input_df['max_grade_combined']
x
y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = LogisticRegression(solver='lbfgs')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

model.fit(x, y)
# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

#check assumptions

# linearity
#check for each variable combination
#gre = sns.regplot(x= 'gre', y= 'admit', data= df, logistic= True).set_title("GRE Log Odds Linear Plot")

#outliers
#describe()
#gpa_rank_box = sns.boxplot(data= df[['gpa', 'rank']]).set_title("GPA and Rank Box Plot")

#independence

#multicollineraity 
#df.corr() #visualize as heatmap

#instead of using years of exp to calcualte max grade related features, use date of ascent completion
#break ascent count down features by ascent types

#check pca of data before model
#add number of flashes, redpoints, onsights etc?

#max flash - completed first try, but have seen others do it or were told how to do it

#max redpoint per user - lead climb after having practiced

#max onsight per user - flashed without seeing any else do it


