#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[2]:


wine=pd.read_csv(r"C:\Users\LENOVO\Downloads\winequality-red.csv")


# In[3]:


wine


# In[4]:


import numpy as np # Basic data handling
import pandas as pd

import matplotlib.pyplot as plt     # Data visualization

import seaborn as sns





# In[5]:


wine.head(10) # first 10 rows for sample 


# In[6]:


wine.tail(10)   #last 10 rows 


# In[7]:


wine.isnull().sum()


# In[ ]:





# In[8]:


wine_null = wine.copy()   # Made a copy so your original data stays safe


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


wine.info()   # Info - shows column names, datatype, non-null count




# In[10]:


wine.describe()    # Describe - summary statistics  #Gives important statistical values:mean min max standard deviation


# In[11]:


wine.dtypes   #Shows me  the datatype of each column.



# In[12]:


wine.shape  #Shows number of rows and columns.


# In[13]:


wine.columns  #Shows list of column names.


# ## Drop Target 

# In[14]:


X = wine.drop('quality', axis=1)
y = wine['quality']   # target column


# In[ ]:





# ## Numerical Column

# In[15]:


num_col= X.select_dtypes(include=['int64', 'float64']).columns   # Numerical columns
num_col


# In[16]:


num_col=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']


# ## Skewness
# 

# In[17]:


X[num_col].skew()


# In[18]:


for col in num_col:
    plt.figure(figsize=(6,4))
    sns.kdeplot(X[col], fill=True)
    plt.title(f"KDE Plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.show()



# In[ ]:





# ## OUTLIERS 

# In[19]:


wine.boxplot()


# In[20]:


for col in num_col:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=X[col])
    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)
    plt.show()


# ## 

# In[21]:


columns = list(X.columns)   # Convert directly from X.columns to list
print(columns)
print(type(columns))


# In[22]:


for col in columns:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR
    
    X.loc[X[col] > ub, col] = ub
    X.loc[X[col] < lb, col] = lb


# In[23]:


print(columns)
print(type(columns))


# In[ ]:





# In[24]:


X.boxplot(column=columns, figsize=(14,6))
plt.show()


# In[25]:


X[num_col].skew()  # Re-check skewness after outlier handling


# In[26]:


new_wine=wine.select_dtypes(include="number")


# In[27]:


new_wine.corr().tail(1)


# In[28]:


plt.figure(figsize=(8,1))
sns.heatmap(new_wine.corr().tail(1),annot=True)


# # Scaling

# In[29]:


from sklearn.preprocessing import StandardScaler

scaler_standard = StandardScaler()

                                             
X_standard = scaler_standard.fit_transform(X)         # Fit on the data and transform

print("Standard Scaling completed.")


# In[30]:


import numpy as np

# Encode target values to start from 0
wine["quality_encoded"] = wine["quality"] - wine["quality"].min()

X = wine.drop(["quality","quality_encoded"], axis=1)
y = wine["quality_encoded"]


# # train_test_split

# In[31]:


from sklearn.model_selection import train_test_split

# Used    Standard Scaled data 
X_train, X_test, y_train, y_test = train_test_split(
    X_standard, y, test_size=0.2, random_state=42, stratify=y
)

print("Train-test split done.")


# # Random FOREST CLASSi

# In[32]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[ ]:





# In[33]:


from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("test_Accuracy:", accuracy)


# In[34]:


train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print("Training Accuracy:", train_acc)


# In[ ]:





# # Hyper parameter tuning

# In[35]:


params = {
    "n_estimators": [50, 100, 150, 200, 300, 400, 500],
    "max_depth": [None, 5, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 3, 4],
    "max_features": ["auto", "sqrt", "log2"]
}


# In[36]:


from sklearn.model_selection import RandomizedSearchCV

rand = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=params,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

rand.fit(X_train, y_train)
print(rand.best_params_)


# # Randomized Search CV

# In[ ]:





# In[37]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

rand_search = RandomizedSearchCV(
    estimator = RandomForestClassifier(random_state=42),
    param_distributions = params,
    n_iter = 20,             # tries 20 random combinations
    cv = 5,                  # 5-fold cross validation
    scoring = 'accuracy',
    n_jobs = -1,             # use all cores
    random_state = 42
)

rand_search.fit(X_train, y_train)


# In[38]:


print("Best Parameters:", rand_search.best_params_)



# In[39]:


best_model = rand_search.best_estimator_
best_model.fit(X_train, y_train)


# In[40]:


from sklearn.metrics import accuracy_score, classification_report

y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)






# In[42]:


from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("test_Accuracy:", accuracy)


# In[43]:


train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print("Training Accuracy:", train_acc)


# # GradientBoostingClassifier

# In[44]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # LogisticRegression

# In[45]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

log = LogisticRegression(max_iter=500, multi_class='multinomial')
log.fit(X_train, y_train)

y_pred = log.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, log.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[46]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[47]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train_scaled, y_train)


# In[ ]:





# In[ ]:





# # svm

# In[ ]:





# In[48]:


from sklearn.svm import SVC
svm = SVC(kernel='rbf')
svm.fit(X_train_scaled, y_train)


y_pred = svm.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, svm.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # xgboost

# In[49]:


pip install xgboost


# In[50]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Encode target
wine["quality_encoded"] = wine["quality"] - wine["quality"].min()

X = wine.drop(["quality", "quality_encoded"], axis=1)
y = wine["quality_encoded"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42n 
)

# Model
xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=len(y.unique()),
    eval_metric='mlogloss',
    random_state=42
)

# Train
xgb.fit(X_train, y_train)

# Predict
y_pred = xgb.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



# In[ ]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=len(y.unique()),
    eval_metric='mlogloss',
    random_state=42
)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, xgb.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # STREAM 

# In[ ]:


import pickle

pickle.dump(svm, open("svm_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train SVM
svm = SVC()
svm.fit(X_train_scaled, y_train)

# save model + scaler
pickle.dump(svm, open("svm_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))


# In[ ]:


pip install streamlit


# In[ ]:


get_ipython().system('pip install streamlit')


# In[ ]:


import streamlit as st
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# load model and scaler
svm = pickle.load(open("svm_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("SVM Prediction App")

st.write("Enter the input values:")

# Example: assuming your model takes 4 inputs
# Change names & count according to your dataset
val1 = st.number_input("Feature 1")
val2 = st.number_input("Feature 2")
val3 = st.number_input("Feature 3")
val4 = st.number_input("Feature 4")

if st.button("Predict"):
    # put values in array
    features = np.array([[val1, val2, val3, val4]])

    # scale features
    features_scaled = scaler.transform(features)

    # prediction
    prediction = svm.predict(features_scaled)[0]

    st.success(f"Prediction: {prediction}")


# In[ ]:


pickle.dump(svm, open("svm_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))


# In[ ]:


# import os
# os.listdir()


# In[51]:


print(wine['quality'].value_counts())


# In[52]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# load
wine = pd.read_csv(r"C:\Users\LENOVO\Downloads\winequality-red.csv")

# features & target
X = wine.drop("quality", axis=1)
y = wine["quality"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# train SVM
svm = SVC(kernel="rbf")
svm.fit(X_train_scaled, y_train)

# save correct model
pickle.dump(svm, open("svm_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("MODEL SAVED SUCCESSFULLY")


# In[ ]:




