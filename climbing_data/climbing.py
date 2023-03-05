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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb



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

#ascents shape - (4111877, 28)
ascents.shape

#ascnts info
ascents.info


#exploring the user data 
user.head(10)

#user shape - (62593, 23)
user.shape

#user data types
user.dtypes

#exploring the grade data
grades.head(10)

#grade shape - (83, 14)
grades.shape

#grade data types
grades.dtypes

#exploring the grade data
method.head(10)

#grade shape - (83, 14)
method.shape

#grade data types
method.dtypes



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
methods_summary
methods_summary.plot(kind='bar')

#toprope is not supposed to be in the section of boulders, so I will remove it
user_ascent_grade_method = user_ascent_grade_method[user_ascent_grade_method['name_methods'].isin(['Flash', 'Onsight', 'Redpoint'])]
methods_summary = user_ascent_grade_method.groupby('name_methods').size()
methods_summary

#identify max grade completed per user
user_ascent_grade_method['max_grade'] = user_ascent_grade_method['usa_boulders_numeric'].groupby(user_ascent_grade_method['id_user']).transform('max')

#Create column with date of first max grade ascent
user_ascent_grade_method['max_grade_date']=user_ascent_grade_method.sort_values(by=['usa_boulders_numeric', 'year'],ascending=[False, True]).groupby('id_user')['year'].transform('first')

#Remove any ascents that were completed after the max grade date 
user_ascent_grade_method = user_ascent_grade_method[user_ascent_grade_method['year'] <= user_ascent_grade_method['max_grade_date']]


#identify years of exp when climber completed the max grade
#if there is a tie for a user by max grade, select the ascent that was completed first
user_ascent_grade_method['years_exp_max_grade']=user_ascent_grade_method.sort_values(by=['usa_boulders_numeric', 'year'],ascending=[False, True]).groupby('id_user')['years_exp'].transform('first')

##identify age when climber completed the max grade
user_ascent_grade_method['age_max_grade']=user_ascent_grade_method.sort_values(by=['usa_boulders_numeric', 'year'],ascending=[False, True]).groupby('id_user')['ascent_age'].transform('first')

#check number ascents at time max grade was completed
user_ascent_grade_method['ascents_max_grade'] = user_ascent_grade_method.groupby('id_user')['id_user'].transform('count')

#pivot table to find number of redpoints, flashes, and onsights
user_ascent_grade_method['ascent_count'] = 1

ascent_type_sums = user_ascent_grade_method.pivot_table(
    values='ascent_count', index='user_id', columns='name_methods',
    fill_value=0, aggfunc='sum')

user_ascent_grade_method = pd.merge(user_ascent_grade_method, ascent_type_sums, on='user_id')

user_ascent_grade_method[user_ascent_grade_method['user_id'] == 3][['user_id', 'ascents_max_grade', 'Redpoint', 'Onsight', 'Flash']]

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

#combine max greade into categories based on median
np.median(no_duplicate_users['max_grade'])
no_duplicate_users['max_grade_combined'] = np.where(no_duplicate_users['max_grade'].isin([-1,0,1,2,3,4,5,6,7,8]), 'Low', 'High')
max_grade_summary = pd.DataFrame(no_duplicate_users.groupby('max_grade_combined').size())
max_grade_summary.index.name = 'max_grade_combined'
max_grade_summary.reset_index(inplace=True)
max_grade_summary.set_index('max_grade_combined').plot(kind='bar',xlabel='Max Grade', ylabel='Number of climbers', color='skyblue', edgecolor='k' )


no_duplicate_users.dtypes
plt.scatter(no_duplicate_users['bmi'], no_duplicate_users['max_grade'], alpha=0.5)
plt.scatter(no_duplicate_users['bmi'], no_duplicate_users['years_exp_max_grade'], alpha=0.5)
plt.scatter(no_duplicate_users['years_exp_max_grade'], no_duplicate_users['ascent_age'], alpha=0.5)
plt.scatter(no_duplicate_users['age_max_grade'], no_duplicate_users['bmi'], alpha=0.5)
plt.scatter(no_duplicate_users['age_max_grade'], no_duplicate_users['years_exp_max_grade'], alpha=0.5)
plt.scatter(no_duplicate_users['ascents_max_grade'], no_duplicate_users['Redpoint'], alpha=0.5)
plt.scatter(no_duplicate_users['ascents_max_grade'], no_duplicate_users['Onsight'], alpha=0.5)
plt.scatter(no_duplicate_users['ascents_max_grade'], no_duplicate_users['Flash'], alpha=0.5)
plt.scatter(no_duplicate_users['Flash'], no_duplicate_users['Redpoint'], alpha=0.5)
plt.scatter(no_duplicate_users['Onsight'], no_duplicate_users['Redpoint'], alpha=0.5)
plt.scatter(no_duplicate_users['Onsight'], no_duplicate_users['Flash'], alpha=0.5)


corr_matrix = np.corrcoef(no_duplicate_users['ascents_max_grade'], no_duplicate_users['Redpoint'])
corr = corr_matrix[0,1]
R_sq = corr**2


#predict the maximum boulder grade one can complete based on sex, years of experience, bmi, number of logged ascents, and progression time per grade? 
input_df = no_duplicate_users[['id_user', 'sex', 'bmi', 'years_exp_max_grade', 'age_max_grade', 'Redpoint', 'Onsight','Flash', 'max_grade_combined']]


input_df.head(10)

#convert outcome category to indicator variable
#High = 1
#Low =0 
input_df['max_grade_outcome'] = pd.get_dummies(input_df['max_grade_combined'])['High']

x =input_df[['sex', 'bmi', 'years_exp_max_grade', 'age_max_grade', 'Redpoint', 'Onsight','Flash']]
y= input_df['max_grade_outcome']
x
y


#pca
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principal_df = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principal_df['max_grade'] = input_df['max_grade_combined']

sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="max_grade",
    palette=sns.color_palette("hls", 2),
    data=principal_df,
    legend="full",
    alpha=0.85
)


#logistic regression provides explainable outcomes, so I will check if this algorithm is suitable

#check assumptions

#multicollinearity
#linearity
#outliers
#indepdence 


#check features for multicollineraity using pearson's correlation
corr = x.corr()
# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns, annot=True)

#redpoint and flash have a high correlation, so I will drop this redpoint as
#flash could be a more informative feature
x = x.drop('Redpoint', axis=1)

#logistic regression also assumes features are indepdent of one another. In this dataset, the
#features do not come from repeated mesaruremtns of the same feature over time, and they are indeodent of one another 

#Logistic regression assumes the relationship between each ontinuousu feature
#and the logit of the outcome variable
#while most of the features appear to meet this assumption, the number of flashes
#looks a little off. I will continue to test the other fewatures to see if logistic regression
#is appropriatie 

sns.regplot(x= input_df['age_max_grade'], y=input_df['max_grade_outcome'], logistic= True)
sns.regplot(x= input_df['years_exp_max_grade'], y=input_df['max_grade_outcome'], logistic= True)
sns.regplot(x= input_df['bmi'], y=input_df['max_grade_outcome'], logistic= True)
sns.regplot(x= input_df['Onsight'], y=input_df['max_grade_outcome'], logistic= True)
sns.regplot(x= input_df['Flash'], y=input_df['max_grade_outcome'], logistic= True)
sns.regplot(x= input_df['Flash'], y=input_df['max_grade_outcome'], logistic= True)

#outliers
sns.boxplot(data= x['bmi'])
sns.boxplot(data= x[['years_exp_max_grade', 'age_max_grade']])
sns.boxplot(data= x[['Onsight', 'Flash']])
#there are quite a few outliers across all of the categories with many of the features
#skewed toward higher values. Logistic regression is sensitive to outliers and
#is not the best algorithm for this analysis

#instead, I will try svm and random forest since they ar more robust to outliers

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)



#model = svm.SVC()
#model=RandomForestClassifier()

from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV


scaler = StandardScaler()
scaler.fit(x)
scaler.transform(x)


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv)
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

    
#svm hyperparamter tuning
from sklearn.svm import SVC


model = SVC()
kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']
# define grid search
grid = dict(kernel=kernel,C=C,gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring='accuracy',error_score=0)
grid_result_svm = grid_search.fit(scaler.transform(x), y)
# summarize results
print("Best: %f using %s" % (grid_result_svm.best_score_, grid_result_svm.best_params_))
means = grid_result_svm.cv_results_['mean_test_score']
stds = grid_result_svm.cv_results_['std_test_score']
params = grid_result_svm.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#randomforest hyperparamter tuning
model = RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
# define grid search
grid = dict(n_estimators=n_estimators,max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(x, y)
#summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#feature importance
model.fit(x, y)
# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


#flash - completed on first try but have seen others do it or were told how to do it

#redpoint - completed climb after having practiced the route

#onsight - completed first try without seeing any else do it or being told how to do it


