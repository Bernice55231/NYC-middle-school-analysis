#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:37:59 2020

@author: bernice
"""

#%% Final Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, integrate
import seaborn as sns
df = pd.read_csv('middleSchoolData.csv')

#%% 1) What is the correlation between the number of applications 
#      and admissions to HSPHS?
df1 = df[['applications','acceptances']].dropna()
x = df1['applications']
y = df1['acceptances']

# %matplotlib inline
np.random.seed(20180514)

sns.distplot(x)
sns.distplot(y)

plt.scatter(x, y) 
plt.title('A plot to show the correlation between applications and acceptances')
plt.xlabel('applications')
plt.ylabel('acceptances')
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='yellow')
plt.show()

correlation = np.corrcoef(x,y)
print(correlation)


#%% 2)What is a better predictor of admission to HSPHS? 
#     Raw number of applications or application “rate”? 

import statsmodels.api as sm
df2 = df[['applications','acceptances','school_size']].dropna()
rate1 = df2['applications'] / df2['school_size']
rate2 = df2['acceptances'] / df2['school_size'] # admission
y1 = df2['acceptances']
y2 = rate2
x1 = df2['applications']
x2 = rate1

sns.distplot(x1)
sns.distplot(x2)

# Z-score the data:
x1 = stats.zscore(x1)
x2 = stats.zscore(x2)
y1 = stats.zscore(y1)
y2 = stats.zscore(y2)

# visualize ( the same as 1) 
plt.scatter(x1, y1) 
plt.title('A plot to show the correlation between applications and acceptances')
plt.xlabel('applications')
plt.ylabel('acceptances')
plt.plot(np.unique(x1), np.poly1d(np.polyfit(x1, y1, 1))(np.unique(x1)), color='yellow')
plt.show()

# linear regression
x1 = sm.add_constant(x1) # vector of ones
model = sm.OLS(y1,x1) # ordinary least squares from sm
results = model.fit() # fit model
print(results.summary()) # print summary
print(results.params) # print parameters, beta0 beta1

# visualize
plt.scatter(x2, y2) 
plt.title('A plot to show the correlation between applications rate and acceptances')
plt.xlabel('applications rate')
plt.ylabel('acceptances')
plt.plot(np.unique(x2), np.poly1d(np.polyfit(x2, y2, 1))(np.unique(x2)), color='yellow')
plt.show()

# linear regression
x2 = sm.add_constant(x2) # vector of ones
model = sm.OLS(y2,x2) # ordinary least squares from sm
results = model.fit() # fit model
print(results.summary()) # print summary
print(results.params) # print parameters, beta0 beta1

#%% 3) Which school has the best *per student* odds of sending someone to HSPHS?
df3 = df[['school_name','applications','acceptances','school_size']].dropna()
rate = df3['acceptances'] / df3['school_size']
odds = rate / (1 - rate)
df3['odds'] = odds
df3 = df3.sort_values(by=['odds'], ascending=False)

#%% 4) Is there a relationship between how students perceive their school (as reported in columns
#      L-Q) and how the school performs on objective measures of achievement (as noted in
#      columns V-X).
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

df4 = df[['rigorous_instruction','collaborative_teachers','supportive_environment',
                'effective_school_leadership','strong_family_community_ties','trust',
                'student_achievement','reading_scores_exceed','math_scores_exceed']].dropna()
# 1. find the principal components for School Climate
df_school = df4[['rigorous_instruction','collaborative_teachers','supportive_environment',
                'effective_school_leadership','strong_family_community_ties','trust']]
# Compute correlation between each measure across all courses:
r = np.corrcoef(df_school.values,rowvar=False)

# Plot the data:
plt.imshow(r) 
plt.colorbar()

scaled_data = preprocessing.scale(df_school)
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

eig_vals = pca.explained_variance_
labels = ['PC' + str(x) for x in range(1, len(eig_vals)+1)]

plt.bar(x=range(1,len(eig_vals)+1), height=eig_vals, tick_label=labels)
plt.ylabel('Eigenvalue')
plt.xlabel('Principal Component')
plt.title('Screet Plot for School Climate')
plt.plot(eig_vals)
plt.show()
# from the plot, we could choose the most important component based on the Kaiser criterion line

# 2. find the principal components for Objective Achievements
df_objective = df4[['student_achievement','reading_scores_exceed','math_scores_exceed']]

scaled_data2 = preprocessing.scale(df_objective)
pca2 = PCA()
pca2.fit(scaled_data2)
pca_data2 = pca2.transform(scaled_data2)

eig_vals2 = pca2.explained_variance_
labels2 = ['PC' + str(x) for x in range(1, len(eig_vals2)+1)]

plt.bar(x=range(1,len(eig_vals2)+1), height=eig_vals2, tick_label=labels2)
plt.ylabel('Eigenvalue')
plt.xlabel('Principal Component')
plt.title('Screet Plot for Objective Achievement')
plt.plot(eig_vals2)
plt.show()
# from the plot, we could choose the most important component based on the Kaiser criterion line

# find the relationship between them
import statsmodels.api as sm
y = pca_data2[:,0] # objective achievement
x = pca_data[:,0] # school performance

# visualize
plt.scatter(x, y) 
plt.title('A plot to show the correlation between school climaete and objective achievement')
plt.xlabel('school climate')
plt.ylabel('objective achievements')
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='yellow')
plt.show()

# linear regression
x = sm.add_constant(x) # vector of ones
model = sm.OLS(y,x) # ordinary least squares from sm
results = model.fit() # fit model
print(results.summary()) # print summary
print(results.params) # print parameters, beta0 beta1
# They are negatively correlated 


#%% 5) Test a hypothesis of your choice as to which kind of school (e.g. small schools vs. large
#      schools or charter schools vs. not (or any other classification, such as rich vs. poor school)) 
#      performs differently than another kind either on some dependent measure, 
#      e.g. objective measures of achievement or admission to HSPHS (pick one).

# I will classify schools as charter schools or public school to testify whether 
# their admission to HSPHS performs differntly

df5 = df[['dbn','school_name','applications','acceptances','school_size']].dropna()
admission = df5['acceptances'] / df5['school_size']
df5['admission'] = admission
df5 = df5.sort_values(by=['school_size'])
length = len(df5) # the length is an even number, so we can simply classify the schools by select
# first half part and second half from the dataframe
data = df5['admission'].values
small_schools = data[:int(length/2),]
large_schools = data[int(length/2):,]

# I try t-test for the two groups (independent t-test)
t,p = stats.ttest_ind(small_schools, large_schools) # independent t-test

# Considering the sample may not derive from nomarl distribution population
# I try ks test to further confirm:

# the p-value is smaller than 0.05, thus we can conclude that there is a significant difference between two groups

#%% 6) Is there any evidence that the availability of material resources (e.g. per student spending or class size) 
#      impacts objective measures of achievement or admission to HSPHS?
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
df6 = df[['applications','acceptances','per_pupil_spending','school_size']].values
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(df6)
IterativeImputer(random_state=0)
df6 = imp.transform(df6)
dataset = pd.DataFrame({'applications':df6[:,0],'acceptances':df6[:,1],'per_pupil_spending':df6[:,2],'school_size':df6[:,3]})
admission = dataset['acceptances'] / dataset['school_size']
dataset['admission'] = admission
df_new = dataset[['per_pupil_spending','admission']]
data = df_new.values

# 1. linear regression and correlation
import statsmodels.api as sm
y = data[:,1] # admission
x = data[:,0] # student spending

# visualize
plt.scatter(x, y) 
plt.title('A plot to show the correlation between spending and admission')
plt.xlabel('spending')
plt.ylabel('admission')
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='yellow')
plt.show()

x = sm.add_constant(x) # vector of ones
model = sm.OLS(y,x) # ordinary least squares from sm
results = model.fit() # fit model
print(results.summary()) # print summary
print(results.params) # print parameters, beta0 beta1


# 2. anova based on different groups
df_new = df_new.sort_values(by=['per_pupil_spending'],ascending=True)
data_new = df_new.values
length = len(data_new)
index = round(length / 4)
sample1 = data_new[:index+1,1]
sample2 = data_new[index:(2*index+1),1]
sample3 = data_new[(2*index):(3*index+1),1]
sample4 = data_new[(3*index):,1]


sns.distplot(sample4)
sns.distplot(sample1)
sns.distplot(sample2)
sns.distplot(sample3)



f,p1 = stats.f_oneway(sample1,sample2,sample3,sample4) # one-way anova for 4 sample means

# 3. kruskal-wallis on different groups
h,p2 = stats.kruskal(sample1,sample2,sample3,sample4) # 4 sample medians

#%% 7) What proportion of schools accounts for 90% of all students accepted to HSPHS?
df7 = df[['dbn','school_name','applications','acceptances']].dropna()
df7 = df7.sort_values(by=['acceptances'],ascending=False)
data = df7['acceptances'].values
num_acceptances = 0.9 * np.sum(data)
analysis = np.empty(len(data))
analysis[:] = np.NAN
for i in range(len(data)):
    analysis[i] = np.sum(data[:i+1])
index = np.argmin(abs(analysis - num_acceptances))
proportion = index / len(data)

#bar graph
labels = df7['dbn']
data_ = data[:21,]
plt.barh(range(len(data_)), data_, align='center', alpha=0.5)
plt.yticks(range(len(data_)), labels)
plt.xlabel('Acceptances Count')
plt.title('Students accepted to HSPHS')

plt.show()


#%% 8) Build a model of your choice – clustering, classification or prediction – that includes all
#      factors – as to what school characteristics are most important in terms of a) sending
#      students to HSPHS, b) achieving high scores on objective measures of achievement?

# I found that these data for per_pupil_spending are absent in charter school, so I decided to separate 
# charter school and public school groups to make appropriate predictions.

# pip install fancyimpute
from fancyimpute import KNN
from sklearn.decomposition import PCA
from sklearn import preprocessing
# charter school
charter_school = df[df['school_name'].str.contains('CHARTER')]
temp1 = charter_school.drop(['per_pupil_spending','avg_class_size'],axis=1)
temp2 = temp1.drop(['dbn','school_name'],axis=1)
c_filled_knn = KNN(k=3).fit_transform(temp2)

for i in range(20):
    help_column_sc = np.log(c_filled_knn[:,i]+1)
    c_filled_knn[:,i] = preprocessing.normalize([help_column_sc])


# public school
index_ = len(charter_school)
public_school = df.drop(list(range(len(df)-index_,len(df))))
temp3 = public_school.drop(['dbn','school_name'],axis=1)
p_filled_knn = KNN(k=3).fit_transform(temp3)

for i in range(20):
    help_column_p = np.log(p_filled_knn[:,i]+1)
    p_filled_knn[:,i] = preprocessing.normalize([help_column_p])

# Charter School
# Compute correlation between each measure across all variables in CHARTER SCHOOLS:
r = np.corrcoef(c_filled_knn,rowvar=False)

# Plot the data:
plt.imshow(r) 
plt.colorbar()
plt.title('correlation between each measure across all variables in CHARTER SCHOOL')

# As we learned in the lecture and recitation we could firstly analyze which variables 
# we need to reduce. Therefore, we could easily see the middle light square so we nned to 
# do PCA as we already did in 4)
# so we can replace the school climate columns by the most important principal component

charter_school = pd.DataFrame(c_filled_knn,columns=['C','D','G','H','I','J','K','L','M','N','O','P','Q',
                                                    'R','S','T','U','V','W','X'])

# 1. find PC for school climate 
school_climate = charter_school[['L','M','N','O','P','Q']].values
scaled_data = preprocessing.scale(school_climate)
pca = PCA()
pca.fit(school_climate)
pca_data = pca.transform(school_climate)

eig_vals = pca.explained_variance_
labels = ['PC' + str(x) for x in range(1, len(eig_vals)+1)]

plt.bar(x=range(1,len(eig_vals)+1), height=eig_vals, tick_label=labels)
plt.ylabel('Eigenvalue')
plt.xlabel('Principal Component')
plt.title('Screet Plot')
plt.plot(eig_vals)
plt.plot([0,len(eig_vals)],[1,1],color='red',linewidth=1) # Kaiser criterion line
plt.show()

#2. find PC for objective achievements
objective_achievement = charter_school[['V','W','X']]

scaled_data2 = preprocessing.scale(objective_achievement)
pca2 = PCA()
pca2.fit(scaled_data2)
pca_data2 = pca2.transform(scaled_data2)

eig_vals2 = pca2.explained_variance_
labels2 = ['PC' + str(x) for x in range(1, len(eig_vals2)+1)]

plt.bar(x=range(1,len(eig_vals2)+1), height=eig_vals2, tick_label=labels2)
plt.ylabel('Eigenvalue')
plt.xlabel('Principal Component')
plt.title('Screet Plot')
plt.plot(eig_vals)
plt.plot([0,len(eig_vals2)],[1,1],color='red',linewidth=1) # Kaiser criterion line
plt.show()

charter_school = charter_school.drop(['L','M','N','O','P','Q','V','W','X'],axis=1)
temp_sc = pca_data[:,0]
school_climate_new = pca_data[:,0]
charter_school['School Climate'] = school_climate_new
objective_achievement_new = pca_data2[:,0]
charter_school['Objective Achievement'] = objective_achievement_new
admission = charter_school['D'] / charter_school['U']
help_admission = np.log(admission+1)
admission = preprocessing.normalize([help_admission])
charter_school['Admission'] = admission.T

# do multiple linear regression: a) admission
from sklearn import linear_model
X = charter_school[['C','G','H','I','J','K','R','S','T','U','School Climate','Objective Achievement']].values 
Y = charter_school['Admission'].values # admission

regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # fit model
COD = regr.score(X,Y) # r^2
beta = regr.coef_ # beta
intercept = regr.intercept_ # intercept


analysis = pd.DataFrame(beta.T,columns=['Weight'])
IV = ['C','G','H','I','J','K','R','S','T','U','School Climate','Objective Achievement']
analysis['Name'] = IV
analysis.sort_values(by=['Weight'],ascending=False)
analysis['Absolute Weight'] = abs(beta.T)
analysis.sort_values(by=['Absolute Weight'],ascending=False)
#%%
# b) objective achievement
X = charter_school[['C','D','G','H','I','J','K','R','S','T','U','School Climate']].values 
Y = charter_school['Objective Achievement'] # high scores on objective measures of achievement
regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # fit model
COD = regr.score(X,Y) # r^2
beta = regr.coef_ # beta
intercept = regr.intercept_ # intercept


analysis = pd.DataFrame(beta.T,columns=['Weight'])
IV = ['C','D','G','H','I','J','K','R','S','T','U','School Climate']
analysis['Name'] = IV
analysis.sort_values(by=['Weight'],ascending=False)
analysis['Absolute Weight'] = abs(beta.T)
analysis.sort_values(by=['Absolute Weight'],ascending=False)

#%%
# Public School
# Compute correlation between each measure across all variables in PUBLIC SCHOOLS:
r2 = np.corrcoef(p_filled_knn,rowvar=False)

# Plot the data:
plt.imshow(r2) 
plt.colorbar()
plt.title('Public School')


# As we learned in the lecture and recitation we could firstly analyze which variables 
# we need to reduce. Therefore, we could easily see the middle light square so we nned to 
# do PCA as we already did in 4)
# so we can replace the school climate columns by the most important principal component

public_school = pd.DataFrame(p_filled_knn,columns=['C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
                                                    'R','S','T','U','V','W','X'])

# 1. find PC for school climate 
school_climate = public_school[['L','M','N','O','P','Q']].values
scaled_data = preprocessing.scale(school_climate)
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

eig_vals = pca.explained_variance_
labels = ['PC' + str(x) for x in range(1, len(eig_vals)+1)]

plt.bar(x=range(1,len(eig_vals)+1), height=eig_vals, tick_label=labels)
plt.ylabel('Eigenvalue')
plt.xlabel('Principal Component')
plt.title('Screet Plot')
plt.plot(eig_vals)
plt.plot([0,len(eig_vals)],[1,1],color='red',linewidth=1) # Kaiser criterion line
plt.show()

#2. find PC for objective achievements
objective_achievement = public_school[['V','W','X']]

scaled_data2 = preprocessing.scale(objective_achievement)
pca2 = PCA()
pca2.fit(scaled_data2)
pca_data2 = pca2.transform(scaled_data2)

eig_vals2 = pca2.explained_variance_
labels2 = ['PC' + str(x) for x in range(1, len(eig_vals2)+1)]

plt.bar(x=range(1,len(eig_vals2)+1), height=eig_vals2, tick_label=labels2)
plt.ylabel('Eigenvalue')
plt.xlabel('Principal Component')
plt.title('Screet Plot')
plt.plot(eig_vals)
plt.plot([0,len(eig_vals2)],[1,1],color='red',linewidth=1) # Kaiser criterion line
plt.show()

public_school = public_school.drop(['L','M','N','O','P','Q','V','W','X'],axis=1)
min_value = np.sort(pca_data[:,0])
temp_sc = pca_data[:,0] + abs(min_value[0])
school_climate_new = preprocessing.normalize([temp_sc])
public_school['School Climate'] = school_climate_new.T
min_value = np.sort(pca_data2[:,0])
temp_oa = pca_data2[:,0]+abs(min_value[0])
objective_achievement_new = preprocessing.normalize([temp_oa])
public_school['Objective Achievement'] = objective_achievement_new.T
admission = public_school['D'] / public_school['U']
help_admission = np.log(admission+1)
admission = preprocessing.normalize([help_admission])
public_school['Admission'] = admission.T

# do multiple linear regression:
from sklearn import linear_model
X = public_school[['C','E','F','G','H','I','J','K','R','S','T','U','School Climate','Objective Achievement']].values 
Y = public_school[['Admission']].values # admission
regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # fit model
COD = regr.score(X,Y) # r^2
beta = regr.coef_ # beta
intercept = regr.intercept_ # intercept

analysis = pd.DataFrame(beta.T,columns=['Weight'])
IV = ['C','E','F','G','H','I','J','K','R','S','T','U','School Climate','Objective Achievement']
analysis['Name'] = IV
analysis.sort_values(by=['Weight'],ascending=False)

analysis['Absolute Weight'] = abs(beta.T)
analysis.sort_values(by=['Absolute Weight'],ascending=False)

#%%
# b) objective achievement
X = public_school[['C','D','G','H','I','J','K','R','S','T','U','School Climate']].values 
Y = public_school['Objective Achievement'] # high scores on objective measures of achievement
regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # fit model
COD = regr.score(X,Y) # r^2
beta = regr.coef_ # beta
intercept = regr.intercept_ # intercept


analysis = pd.DataFrame(beta.T,columns=['Weight'])
IV = ['C','D','G','H','I','J','K','R','S','T','U','School Climate']
analysis['Name'] = IV
analysis.sort_values(by=['Weight'],ascending=False)
analysis['Absolute Weight'] = abs(beta.T)
analysis.sort_values(by=['Absolute Weight'],ascending=False)





