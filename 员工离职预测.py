
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd

df = pd.read_csv('HR_comma_sep.csv')
print (df.info()) 474241623
df.head()


# In[4]:


# rename some columns
df.rename(columns={'average_montly_hours':'average_monthly_hours', 'sales':'department'}, 
          inplace=True)
df.describe()


# In[5]:


print ('Departments:')
print (df['department'].value_counts())
print ('\nSalary:')
print (df['salary'].value_counts())


# In[6]:


'''
satisfaction_level | Satisfaction level of employee based on survey | Continuous | [0.09, 1]
last_evaluation | Score based on employee's last evaluation | Continuous | [0.36, 1]
number_project | Number of projects | Continuous | [2, 7]
average_monthly_hours | Average monthly hours | Continuous | [96, 310]
time_spend_company | Years at company | Continuous | [2, 10]
Work_accident | Whether employee had a work accident | Categorical | {0, 1}
left | Whether employee had left (Outcome Variable) | Categorical | {0, 1}
promotion_last_5years | Whether employee had a promotion in the last 5 years | Categorical | {0, 1}
department | Department employee worked in | Categorical | 10 departments
salary | Level of employee's salary | Categorical | {low, medium, high}
'''


# In[7]:


df.corr()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


plot = sns.factorplot(x='department', y='left', kind='bar', data=df)
plot.set_xticklabels(rotation=45, horizontalalignment='right');


# In[10]:


# Attrition by salary level
plot = sns.factorplot(x='salary', y='left', kind='bar', data=df);


# In[11]:


df[df['department']=='management']['salary'].value_counts().plot(kind='pie', title='Management salary level distribution');


# In[12]:


df[df['department']=='RandD']['salary'].value_counts().plot(kind='pie', title='R&D dept salary level distribution');


# In[13]:


#satisfaction_level
bins = np.linspace(0.0001, 1.0001, 21)
plt.hist(df[df['left']==1]['satisfaction_level'], bins=bins, alpha=0.7, label='Employees Left')
plt.hist(df[df['left']==0]['satisfaction_level'], bins=bins, alpha=0.5, label='Employees Stayed')
plt.xlabel('satisfaction_level')
plt.xlim((0,1.05))
plt.legend(loc='best');


# In[14]:


# Last evaluation
bins = np.linspace(0.3501, 1.0001, 14)
plt.hist(df[df['left']==1]['last_evaluation'], bins=bins, alpha=1, label='Employees Left')
plt.hist(df[df['left']==0]['last_evaluation'], bins=bins, alpha=0.4, label='Employees Stayed')
plt.xlabel('last_evaluation')
plt.legend(loc='best');


# In[15]:


# Number of projects 
bins = np.linspace(1.5, 7.5, 7)
plt.hist(df[df['left']==1]['number_project'], bins=bins, alpha=1, label='Employees Left')
plt.hist(df[df['left']==0]['number_project'], bins=bins, alpha=0.4, label='Employees Stayed')
plt.xlabel('number_project')
plt.grid(axis='x')
plt.legend(loc='best');


# In[16]:


# Average monthly hours
bins = np.linspace(75, 325, 11)
plt.hist(df[df['left']==1]['average_monthly_hours'], bins=bins, alpha=1, label='Employees Left')
plt.hist(df[df['left']==0]['average_monthly_hours'], bins=bins, alpha=0.4, label='Employees Stayed')
plt.xlabel('average_monthly_hours')
plt.legend(loc='best');


# In[17]:


# Years at company 
bins = np.linspace(1.5, 10.5, 10)
plt.hist(df[df['left']==1]['time_spend_company'], bins=bins, alpha=1, label='Employees Left')
plt.hist(df[df['left']==0]['time_spend_company'], bins=bins, alpha=0.4, label='Employees Stayed')
plt.xlabel('time_spend_company')
plt.xlim((1,11))
plt.grid(axis='x')
plt.xticks(np.arange(2,11))
plt.legend(loc='best');


# In[18]:


# whether employee had work accident
plot = sns.factorplot(x='Work_accident', y='left', kind='bar', data=df);


# In[19]:


#whether employee had promotion in last 5 years
plot = sns.factorplot(x='promotion_last_5years', y='left', kind='bar', data=df);


# In[20]:


X = df.drop('left', axis=1)
y = df['left']
X.drop(['department','salary'], axis=1, inplace=True)

# One-hot encoding
salary_dummy = pd.get_dummies(df['salary'])
department_dummy = pd.get_dummies(df['department'])
X = pd.concat([X, salary_dummy], axis=1)
X = pd.concat([X, department_dummy], axis=1)
X.head()


# In[22]:


# Split Training Set from Testing Set (70/30)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[43]:


from sklearn.preprocessing import StandardScaler
X_example = np.array([[ 10., -2.,  23.],
                      [ 5.,  32.,  211.],
                      [ 10.,  1., -130.]])
X_example = stdsc.fit_transform(X_example)
X_example = pd.DataFrame(X_example)
print (X_example)
X_example.describe()


# In[41]:


from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
# transform our training features
X_train_std = stdsc.fit_transform(X_train)
#print (X_train_std[0])
# transform the testing features in the same way
X_test_std = stdsc.transform(X_test)


# In[23]:


# Cross validation
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=20, test_size=0.3)


# In[25]:


# Model #2: Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rf_model = RandomForestClassifier()

rf_param = {'n_estimators': range(1,11)}
rf_grid = GridSearchCV(rf_model, rf_param, cv=cv)
rf_grid.fit(X_train, y_train)
print('Parameter with best score:')
print(rf_grid.best_params_)
print('Cross validation score:', rf_grid.best_score_)


# In[26]:


best_rf = rf_grid.best_estimator_
print('Test score:', best_rf.score(X_test, y_test))


# In[27]:


# feature importance scores
features = X.columns
feature_importances = best_rf.feature_importances_

features_df = pd.DataFrame({'Features': features, 'Importance Score': feature_importances})
features_df.sort_values('Importance Score', inplace=True, ascending=False)

features_df


# In[28]:


features_df['Importance Score'][:5].sum()


# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('HR_comma_sep.csv')


# In[44]:


plt.figure(figsize = (8,8))
plt.subplot(1,2,1)
plt.plot(data.satisfaction_level[data.left == 1],data.last_evaluation[data.left == 1],'o', alpha = 0.1)
plt.ylabel('Last Evaluation')
plt.title('Employees who left')
plt.xlabel('Satisfaction level')

plt.subplot(1,2,2)
plt.title('Employees who stayed')
plt.plot(data.satisfaction_level[data.left == 0],data.last_evaluation[data.left == 0],'o', alpha = 0.1)
plt.xlim([0.4,1])
plt.ylabel('Last Evaluation')
plt.xlabel('Satisfaction level')


# In[31]:


from sklearn.cluster import KMeans
kmeans_df =  data[data.left == 1].drop([ u'number_project',
       u'average_montly_hours', u'time_spend_company', u'Work_accident',
       u'left', u'promotion_last_5years', u'sales', u'salary'],axis = 1)
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(kmeans_df)
kmeans.cluster_centers_


# In[32]:


left = data[data.left == 1]
left['label'] = kmeans.labels_
plt.figure()
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')
plt.title('3 Clusters of employees who left')
plt.plot(left.satisfaction_level[left.label==0],left.last_evaluation[left.label==0],'o', alpha = 0.2, color = 'r')
plt.plot(left.satisfaction_level[left.label==1],left.last_evaluation[left.label==1],'o', alpha = 0.2, color = 'g')
plt.plot(left.satisfaction_level[left.label==2],left.last_evaluation[left.label==2],'o', alpha = 0.2, color = 'b')
plt.legend(['Winners','Frustrated','Bad Match'], loc = 3, fontsize = 15,frameon=True)

