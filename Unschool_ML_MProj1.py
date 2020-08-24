#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.model_selection import train_test_split as tts


# In[81]:


data=pd.read_csv('E:\\Unschool\\ML\\train_u6lujuX_CVtuZ9i.csv')


# In[82]:


data.head()


# In[83]:


data.isnull().sum()


# In[154]:


dt=data.drop(['Loan_ID'], axis=1)
dt


# In[155]:


data['Gender'].hist(bins=2)


# In[156]:


print('Here, ratio of males who have taken loan is very high than females who have taken loan.')


# In[157]:


data['Married'].hist(bins=2)
plt.show()


# In[158]:


print('Here, married people have take loans are approx twice than the unmarried people taken loan')


# In[159]:


data['Education'].value_counts()


# In[160]:


data['Education'].hist(bins=2)
plt.show()


# In[161]:


data['Loan_Amount_Term'].hist(bins=50)
plt.show()


# In[162]:


print('Here, approx 360+ people have Loan Amount Term more than 500')


# In[163]:


data['Property_Area'].value_counts()


# In[164]:


data['Property_Area'].hist(bins=3)


# In[165]:


print('Here, semi-urbans have higher ratio who have taken loans than other Area type.')


# In[166]:


sns.countplot(data['Property_Area'],hue=data['Loan_Status'])
print(pd.crosstab(data['Property_Area'],data['Loan_Status']))


# In[167]:


data['Self_Employed']=data['Self_Employed'].dropna()


# In[168]:


dt.corr()


# In[169]:


sns.heatmap(dt.corr())


# In[170]:


print('Here, none of feature is correlated with any other.')


# In[171]:


dt['Loan_Status'].replace('Y',1,inplace=True)
dt['Loan_Status'].replace('N',0,inplace=True)


# In[172]:


dt.Gender=dt.Gender.map({'Male':1, 'Female':0})


# In[173]:


dt.Married=dt.Married.map({'Yes':1, 'No':0})


# In[174]:


dt.Education=dt.Education.map({'Graduate': 1, 'Not Graduate':0})


# In[175]:


dt.Self_Employed=dt.Self_Employed.map({'Yes':1, 'No':0})


# In[176]:


dt.Property_Area.value_counts()


# In[177]:


dt.Property_Area=dt.Property_Area.map({'Urban':1, 'Semiurban':2, 'Rural':3})


# In[178]:


dt.Dependents=dt.Dependents.map({'0':0, '1':1, '2':2, '3':3, '3+':3})


# In[179]:


from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan, strategy='mean')
dtt=si.fit_transform(dt)


# In[ ]:





# In[181]:


dt.isnull().sum()


# In[182]:


sns.boxplot(dt.ApplicantIncome)


# In[183]:


sns.boxplot(dt.CoapplicantIncome)


# In[184]:


sns.boxplot(dt.LoanAmount)


# In[185]:


sns.boxplot(dt.Loan_Amount_Term)


# In[186]:


sns.boxplot(dt.Credit_History)


# In[187]:


print('Here, looking by above plots, it looks there are outliers. So, we have to handle outliers')


# In[188]:


dt.describe()


# In[189]:


Q1 = dt.quantile(0.25)
Q3 = dt.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[190]:


print(dt < (Q1 - 1.5 * IQR)) |(dt > (Q3 + 1.5 * IQR))


# In[191]:


dt.boxplot(column='LoanAmount', by='Loan_Status')


# In[192]:


fig, ax=plt.subplots()
ax.scatter(dt['ApplicantIncome'], dt['LoanAmount'])
ax.set_xlabel('Income of applicant')
ax.set_ylabel('Amount of loan')
plt.show()


# Outlier Treatment

# In[193]:


#Quantile-based Flooring and Capping
print(dt['ApplicantIncome'].quantile(0.10))
print(dt['ApplicantIncome'].quantile(0.90))


# In[194]:


#Trimming
dt["ApplicantIncome"] = np.where(dt["ApplicantIncome"] <2960.0, 2960.0,dt['ApplicantIncome'])
dt["ApplicantIncome"] = np.where(dt["ApplicantIncome"] >12681.0, 12681.0,dt['ApplicantIncome'])
print(dt['ApplicantIncome'].skew())


# In[195]:


#checking IQR Score
dt_out = dt[~((dt < (Q1 - 1.5 * IQR)) |(dt > (Q3 + 1.5 * IQR))).any(axis=1)]
print(dt_out.shape)


# In[196]:


#Log Transformation
dt["Log_Loanamt"] = dt["LoanAmount"].map(lambda i: np.log(i) if i > 0 else 0) 
print(dt['LoanAmount'].skew())
print(dt['Log_Loanamt'].skew())


# In[197]:


#replacing outliers
print(dt['LoanAmount'].quantile(0.50)) 
print(dt['LoanAmount'].quantile(0.95)) 
dt['LoanAmount'] = np.where(dt['LoanAmount'] > 325, 140, dt['LoanAmount'])
dt.describe()


# In[198]:


dt.isnull().sum()


# In[200]:


dt['LoanAmount'] = dt['LoanAmount'].fillna(dt['LoanAmount'].mean())


# In[201]:


dt['Credit_History'] = dt['Credit_History'].fillna(dt['Credit_History'].median())


# In[203]:


dt.dropna(inplace=True)


# In[204]:


dt.head()


# In[205]:


dt.isnull().sum()


# In[206]:


x=dt.drop(['Loan_Status'], axis=1)
y=dt['Loan_Status']
x_train, x_test, y_train, y_test=tts(x,y, test_size=0.3, random_state=0)


# In[218]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lr=LogisticRegression()
lr.fit(x_train, y_train)


# In[219]:


y_pred=lr.predict(x_test)


# In[220]:


print('Accuracy Score: ', metrics.accuracy_score(y_test, y_pred))


# In[228]:


features=pd.DataFrame({'Features':x_train.columns, 'Importance':np.round(dt.feature_importances_,3)})
features_dt=features.sort_values('Importance', ascending=False)


# In[232]:


importance = lr.coef_


# In[236]:


importance


# In[239]:


plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[ ]:




