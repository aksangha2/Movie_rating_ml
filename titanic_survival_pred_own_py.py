#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv("titanic.csv")


# In[4]:


df.head(n=10)


# In[441]:


df.info()


# In[442]:


columns_not_used=["name","ticket","cabin","embarked","home.dest","body","boat"]


# In[443]:


data=df.drop(columns=columns_not_used,axis=1)


# In[444]:


data.info()


# In[445]:


data["age"]=data["age"].fillna(data["age"].mean())


# In[446]:


data.info()


# In[447]:


data["fare"]=data["fare"].fillna(data["fare"].mean())


# In[448]:


data.info()


# In[449]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['sex']=le.fit_transform(data['sex'])


# In[450]:


data.head()


# In[451]:


data['sex']=data['sex'].fillna(data['sex'].mean())


# In[452]:


data.info()


# In[453]:


def entropy(x_data,col):
    num=np.unique(x_data['survived'],return_counts=True)
    ent=0.0
    for i in num[1]:
        p=i/col.shape[0]
        ent+=p*(np.log2(p))
        #print(ent)
    return ent


def info_gain(x_data,fkey,fval):
    x_left=x_data.loc[x_data[fkey]<fval]
    x_right=x_data.loc[x_data[fkey]>=fval]
    
    #print(x_left.shape)
    #print(x_right.shape)
    if x_left.shape[0]==0 or x_right.shape[0]==0:
        return -1000000
    hs=entropy(x_data,x_data[fkey])
    x_left_ent=entropy(x_left,x_left[fkey])
    x_right_ent=entropy(x_right,x_right[fkey])
    
    ig=hs-(((x_left.shape[0])/x_data.shape[0])*x_left_ent)-(((x_right.shape[0])/x_data.shape[0])*x_right_ent)
    return ig
    
    
    
        
        
        
    
    


# In[454]:


#p=np.unique(data['survived'],return_counts=True)
#print(p)
#len(p)
#p[1][0]


# In[455]:


i=info_gain(data,'age',data['age'].mean())
print(i)


# In[456]:


class decisiontree:
    def __init__(self,depth,max_depth=7):
        self.left=None
        self.right=None
        self.fkey=None
        self.fval=None
        self.target=None
        self.depth=depth
        self.max_depth=max_depth
    def build(self,x_data):
        features=["pclass","sex","age","fare","sibsp","parch"]
        ig=[]
        for i in features:
            p=info_gain(x_data,i,x_data[i].mean())
            ig.append(p)
        index=np.argmax(ig)
        self.fkey=features[index]
        self.fval=x_data[self.fkey].mean()
        
        print(self.fkey)
        x_left=x_data.loc[x_data[self.fkey]<self.fval]
        x_right=x_data.loc[x_data[self.fkey]>=self.fval]
        
        #print(x_left.shape,x_right.shape)
        
        if x_left.empty==True or x_right.empty==True:
            if(((np.sum(x_data['survived']))/x_data.shape[0])>=0.5):
                self.target=1
            else:
                self.target=0
            return
        
        
        if(self.depth>self.max_depth):
            if(((np.sum(x_data['survived']))/x_data.shape[0])>=0.5):
                self.target=1
            else:
                self.target=0
            return
        
        self.left=decisiontree(self.depth+1)
        self.right=decisiontree(self.depth+1)
        
        x_left=x_left.reset_index(drop=True)
        x_right=x_right.reset_index(drop=True)
       
        self.left.build(x_left)
        self.right.build(x_right)  
        
        if(((np.sum(x_data['survived']))/x_data.shape[0])>=0.5): 
            self.target=1
        else:
            self.target=0
        return
       
    def predict(self,test):
        if self.left==None or self.right==None:
            return self.target
        if test[self.fkey]<self.fval:
            return(self.left.predict(test))
        else:
            return(self.right.predict(test))
        
            
        
        


# In[457]:


data.shape


# In[458]:


dt=decisiontree(0)
dt.build(data)


# In[459]:


dt.predict(data.loc[3])


# In[460]:


pred=[]
for i in range(data.shape[0]):
    pred.append(dt.predict(data.loc[i]))
y_actual=data["survived"]
a=(np.sum(np.array(y_actual)==np.array(pred)))/y_actual.shape[0]
    
print(a)


# In[461]:


test=pd.read_csv("titanictest.csv")


# In[462]:


test.head()


# In[463]:


test.info()


# In[464]:


test=test.drop(columns=columns_not_used,axis=1)


# In[467]:


test.info()


# In[466]:


test["age"]=test["age"].fillna(test["age"].mean())


# In[468]:


test['sex']=le.fit_transform(test['sex'])
pred=[]
for i in range(test.shape[0]):
    pred.append(dt.predict(test.loc[i]))
pred=np.array(pred)
pd.DataFrame(pred).to_csv("titanicsolution.csv")


# In[ ]:




