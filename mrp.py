import numpy as np
import pandas as pd
df=pd.read_csv("Train.csv")
data=df.values
x_train=data[:,0]
y_train=data[:,1]
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
tokenizer = RegexpTokenizer('[a-zA-Z]+')
stopword = set(stopwords.words('english'))
#ps = PorterStemmer()
Wordnet_Lemmatizer=WordNetLemmatizer()
def clean_review(review):
    review=str(review)
    review = review.lower()
    review = review.replace("<br /><br />"," ")
    
    tokens = tokenizer.tokenize(review)
    new_tokens = [Wordnet_Lemmatizer.lemmatize(token) for token in tokens if token not in stopword]
    
    return ' '.join(new_tokens)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(tokenizer=clean_review,ngram_range=(1,1),norm='l2')
vector=vectorizer.fit_transform(x_train)
df2=pd.read_csv("Test.csv")
x_test=df2.values
x_test=list(x_test)
x_test = [clean_review(sent) for sent in x_test]
x_test=vectorizer.transform(x_test)
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(vector,y_train)
p=mnb.predict(x_test)
y_test=np.array(p)
pd.DataFrame(y_test).to_csv("testing6.csv")