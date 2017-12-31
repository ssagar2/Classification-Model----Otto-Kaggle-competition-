
# coding: utf-8

# In[112]:


import pandas as pd
import numpy

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE

from sklearn.cross_validation import train_test_split

from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier as xgb
from sklearn.ensemble import AdaBoostClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


data = pd.read_csv("C:\\Users\\swapnil sagar\\Documents\\UIC\\AAI\\final project\\new data set\\train.csv");
data.head(5)


# In[26]:


array = data.values
X = array[:,1:94]
Y = array[:,94]


# In[279]:


X[:2,:]


# In[280]:


Y


# In[339]:


test = SelectKBest(score_func=chi2, k=78)
f = test.fit(X, Y)


# In[340]:


numpy.set_printoptions(precision=3)
print(f.scores_)
features = f.transform(X)


# In[299]:


print(features[0:2,:])


# In[17]:


model = LogisticRegression()
rfe = RFE(model, 70)
fit = rfe.fit(X, Y)
#print("Num Features: %d") % fit.n_features_
#print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_


# In[18]:


fea =fit.transform(X)
fea[:2,:]


# In[27]:


trainData, testTrainData , y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state = 40)


# In[128]:


knn = KNeighborsClassifier().fit(X,Y)


# In[103]:


classifier_sgd = MultinomialNB()
classifier_sgd.fit(trainData,y_train)


# In[136]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)


# In[141]:


SVM = linear_model.SGDClassifier()
SVM.fit(X,Y)


# In[118]:


clf1 = LogisticRegression(random_state=1)
clf1 = clf1.fit(X,Y)


# In[81]:


RFC = RandomForestClassifier(n_estimators=10).fit(X,Y)


# In[114]:


GBC = GradientBoostingClassifier(n_estimators=10).fit(trainData,y_train)


# In[100]:


bdt = AdaBoostClassifier(n_estimators=200).fit(trainData,y_train)


# In[113]:


params = {"objective": "multi:softprob", "num_class": 9}

T_train_xgb = xgb.DMatrix(X_train, Y_train)
X_test_xgb  = xgb.DMatrix(X_test)

gbm = xgb.train(params, T_train_xgb, 20)
Y_pred = gbm.predict(X_test_xgb)


# In[135]:


predicted_labels = clf.predict(testTrainData)
predicted_labelsList = predicted_labels.tolist();
clf.score(testTrainData, y_test)


# In[161]:


target_classifiers = ['1', '2', '3','9']
print(classification_report(y_test, predicted_labelsList, target_names=target_classifiers))


# In[182]:


predicted_labels


# In[84]:


test_df = pd.read_csv("C:\\Users\\swapnil sagar\\Documents\\UIC\\AAI\\final project\\new data set\\test.csv");
#array = data.values
X_test = test_df.values[:,1:]


# In[85]:


X_test


# In[143]:


prepredicted_labels_test = clf1.predict_proba(X_test)
prepredicted_labels_test[:2,:]


# In[87]:


submission = pd.DataFrame(columns=['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
#submission.loc[0] = [1, 2, 3]
submission


# In[144]:


i = 0
#range_of_classes = range(1, 10)
# Create column name based on target values(see sample_submission.csv)
for num in prepredicted_labels_test:
    #col_name = str("Class_{}".format(num))
    submission.loc[i+1] = [i+1,prepredicted_labels_test[i][0],prepredicted_labels_test[i][1],prepredicted_labels_test[i][2],
                           prepredicted_labels_test[i][3],prepredicted_labels_test[i][4]
                          ,prepredicted_labels_test[i][5],prepredicted_labels_test[i][6],prepredicted_labels_test[i][7],
                           prepredicted_labels_test[i][8]]

    i=i+1


# In[140]:



submission.to_csv('C:\\Users\\swapnil sagar\\Documents\\UIC\\AAI\\final project\\new data set\\finalLR.csv', index=False)


# In[145]:


submission.head(5)

