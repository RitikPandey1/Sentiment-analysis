from nltk.corpus.reader import ipipan
from nltk.probability import KneserNeyProbDist
import numpy as np
import pandas as pd


dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',  quoting=3)

import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

clean_reviews = []

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    stop_words = stopwords.words('english')
    stop_words.remove('not')
    stop_words.remove('no')
    review = [ps.stem(word) for word in review if not word in set(stop_words)]
    review = ' '.join(review)
    clean_reviews.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(clean_reviews).toarray()
y = dataset['Liked']


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

def accuCheck(y_pred):
    print("confusion matrix :-")
    print(confusion_matrix(y_test,y_pred))
    print('Accuracy score : ',accuracy_score(y_test,y_pred)*100,'%')
    

from sklearn.naive_bayes import GaussianNB
gn = GaussianNB()
gn.fit(x_train,y_train)
y_pred = gn.predict(x_test)
print("-- naive bayes --")
accuCheck(y_pred)

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(random_state=0)
lg.fit(x_train,y_train)
y_pred1 = lg.predict(x_test)
print('-- logistic regression --')
accuCheck(y_pred1)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(x_train,y_train)
y_pred2 = knn.predict(x_test)
print("-- KNN --")
accuCheck(y_pred2)