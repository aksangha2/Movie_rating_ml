#!/usr/bin/env python
# coding: utf-8

# In[106]:


import pandas as pd
import numpy as np
data=pd.read_csv("titanic.csv")
#data.head()


# In[107]:


columns_not_used=["name","ticket","cabin","embarked","home.dest","boat","parch"]
data=data.drop(columns=columns_not_used,axis=1)
data["age"]=data["age"].fillna(data["age"].mean())
data["fare"]=data["fare"].fillna(data["fare"].mean())
data['body']=data["body"].fillna(data['body'].mean())


# In[108]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['sex']=le.fit_transform(data['sex'])

#data['boat']=le.fit_transform(data['boat'])


# In[109]:


data['sex']=data['sex'].fillna(data['sex'].mean())
#data['boat']=data["boat"].fillna(data['boat'].mean())


# In[110]:


data=data.values
data[:,[0,1]]=data[:,[1,0]]
x_data=data[:,1:]
y_data=data[:,0]


# In[124]:


from sklearn import tree
dt=tree.DecisionTreeClassifier(max_depth=10)
dt=dt.fit(x_data,y_data)


# In[125]:


test=pd.read_csv("titanictest.csv")
test=test.drop(columns=columns_not_used,axis=1)
test['age']=test['age'].fillna(test['age'].mean())
test['fare']=test['fare'].fillna(test['fare'].mean())
test['sex']=le.fit_transform(test['sex'])
test['sex']=test['sex'].fillna(test['sex'].mean())
test['body']=test["body"].fillna(test['body'].mean())
#data['boat']=le.fit_transform(data['boat'])
#test['boat']=test["boat"].fillna(test['boat'].mean())
test=test.values


# In[126]:


pred=dt.predict(test)


# In[127]:


pd.DataFrame(pred).to_csv("titanicsolnscikit.csv")


# In[115]:





# In[ ]:




