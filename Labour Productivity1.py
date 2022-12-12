#!/usr/bin/env python
# coding: utf-8

# ### Project-82 : To track the performance of the worker in a civil construction site

# ### Business constraints : Minimise = Delay of Work ; Maximise = Performance of the worker 

# ### Procedure : To track the performance of a worker we have created a data set to track the location , motion , work he is doing at the site , over time if he does any , time he spent on doing work . To track the mention parameters we are using RFID  sensors , IOT , sensor and safty we are using gas and IP sensors.

# ### We have considered a csv file and we wil perform EDA on the file and also built up a model according the data analysis done in EDA 

# #### 1. First part in EDA is to Import some of the essential libraries  

# In[9]:


import pandas as pd 
import pandas.io.sql as sqlio
import psycopg2
import numpy as np
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


connection = psycopg2.connect(user = "postgres",
                              password = "Goudar@915",
                              host = "localhost",
                              port = "5433",
                              database = "Project 1")


# In[11]:


connection


# In[12]:


sql = """select * from performance"""


# In[13]:


df = sqlio.read_sql_query(sql,connection)


# In[14]:


df.head()


# In[15]:


df.columns


# In[16]:


df.info()


# ### Performing auto eda using pandas-profiling

# In[17]:


from pandas_profiling import ProfileReport
profile = ProfileReport(df, title = 'Pandas Profiling Report')
profile


# In[18]:


df.shape


# In[19]:


df


# In[20]:


df.info()


# In[21]:


df.describe()


# In[22]:


df['Motion_Productivity'].value_counts()


# In[23]:


df['Motion_Indication'].value_counts()


# In[24]:


df['Noise_Detection'].value_counts()


# In[25]:


df['Infrared_Sensor'].value_counts()


# ### 2.1 Null Values in the dataset

# In[26]:


df.isna().sum()


# #### Duplicate values

# In[27]:


df.duplicated().sum() ## There are no duplicated values


# In[28]:


#### Outliers in the data set 


# In[29]:


import matplotlib.pyplot as plt
fig=plt.figure(figsize=(50,30))
sns.boxplot(data= df)
plt.show()


# In[30]:


import seaborn as sns
# Let's find outliers in Salaries
sns.boxplot(df.Skin_Sensor)


# In[31]:


sns.boxplot(df.RR)


# In[32]:


# Detection of outliers (find limits for skin_sensor based on IQR)
IQR = df['Skin_Sensor'].quantile(0.75) - df['Skin_Sensor'].quantile(0.25)

lower_limit = df['Skin_Sensor'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Skin_Sensor'].quantile(0.75) + (IQR * 1.5)


# In[33]:


from feature_engine.outliers import Winsorizer

# Define the model with IQR method
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                        # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['Skin_Sensor'])

df_s = winsor_iqr.fit_transform(df[['Skin_Sensor']])


# In[34]:


sns.boxplot(df_s.Skin_Sensor)


# In[35]:


IQR = df['RR'].quantile(0.75) - df['RR'].quantile(0.25)

lower_limit = df['RR'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['RR'].quantile(0.75) + (IQR * 1.5)


# In[36]:


winsor_iqr = Winsorizer(capping_method = 'iqr', 
                        # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['RR'])

df_s = winsor_iqr.fit_transform(df[['RR']])


# In[37]:


sns.boxplot(df_s.RR)


# In[38]:


## Data Visualization
#Count Plots


# In[39]:


features_continuous = ['Age','Experience','TotalWorkingHours','Working_Hours','RemainingWorkingHours','ActualWorkingHours','BeatsPerMinute','OverTimeHours','RFID_Tag']
for feature in features_continuous:
  sns.displot(data=df, x=feature, height=4, aspect=2,color='#158685')
  plt.xticks(rotation=65)
  plt.show()


# In[40]:


features_category = ['Designation','Site', 'Motion_Indication','Motion_Productivity','Noise_Detection','Infrared_Sensor','Gas_Sensor','Output']
for feature in features_category:
  sns.countplot(data=df, x=feature, palette='deep')
  plt.xticks(rotation=65)
  plt.show()


# In[41]:


# histogram
df.hist()
plt.show# overall distribution of data


# In[42]:


values = df['Nationality'].value_counts()
labels = df['Nationality'].unique().tolist()

fig = px.pie(df, values = values, labels = labels, title = 'Nationality')

fig.show()
print(values)


# In[49]:


df1= df.drop(["Employee_ID","Name","Gender","Attendance","Nationality","Over_Time","Site","Location","Qualification","Place","RFID_Tag","Time_In","Time_Out","TotalWorkingHours","Infrared_Sensor","Gas_Sensor"],axis=1)
df1


# In[50]:


df1.info()


# In[51]:


##Data Preprocessing
labelencoder = LabelEncoder()


# In[52]:


df1["Designation"]= labelencoder.fit_transform(df1["Designation"])
df1["Motion_Productivity"]= labelencoder.fit_transform(df1["Motion_Productivity"])
df1["Motion_Indication"] = labelencoder.fit_transform(df1["Motion_Indication"])
df1["Noise_Detection"]= labelencoder.fit_transform(df1["Noise_Detection"])
df1


# In[53]:


df1.info()


# #### Pairplot

# In[55]:


sns.pairplot(df1)


# ### Correlation Coefficent Plot

# In[56]:


corelation = df1.corr()
plt.figure(figsize=(20,14))
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)


# ####  Variance, Skewness and Kurtosis

# In[60]:


df1.var() # variance of Numeric Variables


# In[61]:


df1.skew() 


# In[62]:


df1.kurt() 


# ## Model Building

# In[63]:


# separating the data and labels
X = df1.drop(columns = 'Output', axis=1)
Y = df1['Output']


# In[64]:


Y


# In[65]:


# Normalization function

def norm_func(i):
    x = (i - i.min())/(i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df1 = norm_func(X)


# In[66]:


df1


# In[67]:


X = df1
X
Y


# In[68]:


# splitting the data into testing and training data.

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify = Y)


# In[69]:


print(X.shape, X_train.shape, X_test.shape)


# In[70]:


# Support Vector Machine (SVM-Black Box Technique)   
from sklearn import svm
from sklearn.metrics import accuracy_score
classifier = svm.SVC(kernel='linear')


# In[71]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# In[72]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)


# In[73]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)


# In[74]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


# In[75]:


# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)


# In[76]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[77]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[78]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[79]:


print('Accuracy on Test data : ', test_data_accuracy)


# In[80]:


#Desicion tree Model

#Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score,confusion_matrix

# Model building for Decision Tree

dtc = DecisionTreeClassifier(criterion="gini", max_depth=3)

dtc.fit(X_train,Y_train)
Y_train_pred = dtc.predict(X_train)
Y_test_pred = dtc.predict(X_test)

print(f'Train score {accuracy_score(Y_train_pred,Y_train)}') 
print(f'Test score {accuracy_score(Y_test_pred,Y_test)}') 

# confusion matrix for performance metrics
cm = confusion_matrix(Y_test, Y_test_pred)
cm
# or
pd.crosstab(Y_test, Y_test_pred, rownames=['Actual'], colnames=['Predictions'])

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_test_pred))


# In[81]:


# Training the model Random forest model
from sklearn.ensemble import RandomForestClassifier

classifier_rfg=RandomForestClassifier(random_state=33,n_estimators=23)
parameters=[{'min_samples_split':[2,3,4,5],'criterion':['gini','entropy'],'min_samples_leaf':[1,2,3]}]

model_gridrf=GridSearchCV(estimator=classifier_rfg, param_grid=parameters, scoring='accuracy',cv=10)
model_gridrf.fit(X_train,Y_train)

model_gridrf.best_params_

# Predicting the model
Y_predict_rf = model_gridrf.predict(X_test)

# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(Y_test,Y_predict_rf))
print(classification_report(Y_test,Y_predict_rf))


# In[82]:


## Building a Predictive System 
input_data = (40,0,16,32,2504,3,81,131,90.19,6.98,15.0,5,3,1)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model_gridrf.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is non productive')
else:
  print('The person is productive')


# # Saving the trained model

# In[86]:


import pickle
file = 'model.pkl'
pickle.dump(model_gridrf, open(file,'wb'))


# In[87]:


# loading the saved model
loaded_model = pickle.load(open('model.pkl', 'rb'))


# In[88]:


## Building a Predictive System 
input_data = (36,0,10,21,2515,4,11,110,97.07,6.98,15.0,7,2,0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not productive')
else:
  print('The person is productive')

