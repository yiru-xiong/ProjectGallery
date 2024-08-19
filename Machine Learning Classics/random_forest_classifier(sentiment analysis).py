# -*- coding: utf-8 -*-
"""Random Forest Classifier

@author: Yiru Xiong

This file contains the implementation of the random forest classifier.
The beginning part of the program aims to import relevant python packages, retrieve and preprocess data. Data could be downloaded from the original source website and directly loaded into the workspace for further manipulation.The latter part of the program is to train and fit random forest model with different feature representations and boosting approaches.
The optimal setup of the random forest model is 800 decision trees (n_estimators = 800) with 100 maximum depths (max_depth=100) and has minimum8 datapoints placed in a node before the split (min_sample_split=8).
System requirement: tensorflow, sklearn, nltk and other common packages
"""

import tensorflow as tf
import os
import re
import pandas as pd
import numpy as np
import tensorflow_hub as hub
import re
import nltk
import nltk.data
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve,auc
from sklearn import datasets, linear_model
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2

nltk.download("stopwords")
nltk.download('wordnet')

# read data and convert into dataframe
def directory_data(directory):
  data = {}
  data["Reviews"] = []
  for file_path in os.listdir(directory):
    with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["Reviews"].append(f.read())
  return pd.DataFrame.from_dict(data)

def load_dataset(directory):
  pos_df = directory_data(os.path.join(directory, "pos"))
  neg_df = directory_data(os.path.join(directory, "neg"))
  pos_df["label"] = 1
  neg_df["label"] = 0
  all_df = pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)
  return all_df

# retrieve data from the source website
def df_datasets(force_download=False):
  dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz", 
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
      extract=True)
  train_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                       "aclImdb", "train"))
  test_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                      "aclImdb", "test"))
  return train_df, test_df

train_df, test_df = df_datasets()

train_text = train_df['Reviews'].tolist()
train_text = [' '.join(t.split()) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = train_df['label'].tolist()

test_text = test_df['Reviews'].tolist()
test_text = [' '.join(t.split()) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_df['label'].tolist()

# concat train and test data to a complete 50K dataset
train_all = pd.concat([train_df, test_df], axis=0)
np.shape(train_all)
train_all.head()

# stemming and lemmatization
stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()
stemmer = nltk.stem.SnowballStemmer('english')

def data_preprocess(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = re.sub(r'[?|!|\'|"|#]',r'',text)
    text = re.sub(r'[.|,|)|(|\|/]',r' ',text)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    #text = [lemmatizer.lemmatize(token, "v") for token in text]
    #text = [lemmatizer.lemmatize(token, "a") for token in text]
    text = [word for word in text if not word in stop_words]
    #text = " ".join(text)
    text = [stemmer.stem(word) for word in text]
    text = " ".join(text)
    return text

train_all['Processed_Reviews'] = train_all.Reviews.apply(lambda x: data_preprocess(x))

train_all.head()
train_all.label.value_counts()

X_train, X_test, y_train, y_test = train_test_split(train_all['Processed_Reviews'], train_all['label'], test_size=0.1)
print ('Shape of train dataset:', X_train.shape, y_train.shape)
print ('Shape of test dataset:', X_test.shape, y_test.shape)

# bag of words representation 
cv = CountVectorizer(ngram_range =(1,2),min_df=5, max_df=0.8, stop_words=stopwords.words('english'),analyzer='word')
train_cv=cv.fit_transform(X_train)
test_cv =cv.transform(X_test)

# feature selections using chi square, keep 6000
np.seterr(divide='ignore', invalid='ignore')
chi2_features = SelectKBest(chi2, k = 6000) 
selector = chi2_features.fit(train_cv, y_train)
train_selected = selector.transform(train_cv)
test_selected = selector.transform(test_cv)

# hyperparameter tuning - grid search with cross validation
param_grid = {
    'max_depth': [50, 100, 200],
    'max_features': ['auto', 2, 5],
    #'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [2, 8, 12],
    #'n_estimators': [100, 200, 500, 800]
    'n_estimators': [800, 1200, 2000]
}

rf_classifier = RandomForestClassifier(criterion = 'entropy')
#grid_search = GridSearchCV(estimator = rf_classifier, param_grid = param_grid, cv = 3, n_jobs = -1)

# Warning-- takes long to run 
#grid_search.fit(train_selected,y_train)

grid_search.best_params_
#{'max_depth': 100, 'min_samples_split': 8, 'n_estimators': 800}

best_grid = grid_search.best_estimator_
"""
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=100, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=8,
                       min_weight_fraction_leaf=0.0, n_estimators=800,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
"""

# try random search 
param_range = {
    'max_depth': [50, 200],
    'max_features': [2, 8],
    #'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [2,12],
    #'n_estimators': [100, 200, 500, 800]
    'n_estimators': [300,1000]
}
random_search = RandomizedSearchCV(rf_classifier, param_distributions=param_range,n_iter=16)

#random_search.fit(train_selected, y_train)

#plot
import matplotlib.pyplot as plt
#tree_num_grid = {'n_estimators': [int(x) for x in np.linspace(100, 1000, 200)]}
#tree_grid_search = GridSearchCV(estimator = rf_classifier, param_grid=tree_num_grid, verbose = 2, n_jobs=-1,cv=1)

# Model Fitting 
rf_classifier = RandomForestClassifier(criterion = 'gini', n_estimators=500,max_depth=100,min_samples_split=8,n_jobs=-1)
rf_classifier.fit(train_selected,y_train)

score=rf_classifier.score(train_selected, y_train)
print("*======Random Forest Output=======*","\nTraining Data Size:45000",
      "\nTraining accuracy:", rf_classifier.score(train_selected, y_train), "\nValidation accuracy:",rf_classifier.score(test_selected, y_test))
"""
=======Random Forest Output======== 
Training Data Size:45000 
Training accuracy: 0.9964666666666666 
Validation accuracy: 0.857
*======Random Forest Output=======* 
Training Data Size:45000 
Training accuracy: 0.9966444444444444 
Validation accuracy: 0.8678
"""

# confusion matrix 
# f1-score :0.87
rfc_predict=rf_classifier.predict(test_selected)
cm=confusion_matrix(y_test,rfc_predict)
cr=classification_report(y_test,rfc_predict)
print('Classification report is:\n',cr)
# AUC score
# AUC score: 0.868
fpr_rf,tpr_rf,threshold_rf=roc_curve(y_test,rfc_predict)
auc_rf=auc(fpr_rf,tpr_rf)
print('AUC score for Current Random Forest classifier:',np.round(auc_rf,3))

# add TfIdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# TF-IDF transformation 
tfidf_transform = TfidfTransformer().fit(train_cv)
train_tfidf = tfidf_transform.transform(train_cv)
test_tfidf = tfidf_transform.transform(test_cv)
np.seterr(divide='ignore', invalid='ignore')
chi2_features = SelectKBest(chi2, k = 6000) 
selector = chi2_features.fit(train_tfidf, y_train)
train_selected = selector.transform(train_tfidf)
test_selected = selector.transform(test_tfidf)

rf_classifier = RandomForestClassifier(criterion = 'gini', n_estimators=800, max_depth=100,min_samples_split=8,n_jobs=-1)
rf_classifier.fit(train_tfidf,y_train)

score=rf_classifier.score(train_tfidf, y_train)
print("*======Random Forest Output=======*","\nBow with TfIDF transformation\n","\nTraining Data Size:45000",
      "\nTraining accuracy:", rf_classifier.score(train_tfidf, y_train), "\nValidation accuracy:",rf_classifier.score(test_tfidf, y_test))
"""
=======Random Forest Output======== 
Bow with TfIDF transformation
Training Data Size:45000 
Training accuracy: 0.9980666666666667 
Validation accuracy: 0.879
*======Random Forest Output=======* 
Bow with TfIDF transformation
Training Data Size:45000 
Training accuracy: 0.9976888888888888 
Validation accuracy: 0.8746
"""

# confusion matrix 
# f1-score :0.87
rfc_predict=rf_classifier.predict(test_tfidf)
cm=confusion_matrix(y_test,rfc_predict)
cr=classification_report(y_test,rfc_predict)
print('Classification report is:\n',cr)
# AUC score
# AUC score: 0.868
fpr_rf,tpr_rf,threshold_rf=roc_curve(y_test,rfc_predict)
auc_rf=auc(fpr_rf,tpr_rf)
print('AUC score for Current Random Forest classifier (with TF-IDF Representation):',np.round(auc_rf,3))

from sklearn.ensemble import AdaBoostClassifier

# adaboost
boost_reg = AdaBoostClassifier(base_estimator = rf_classifier, n_estimators = 800, learning_rate = 0.9)

boost_reg.fit(train_tfidf, y_train)

print("*======Random Forest Output=======*","\nTraining accuracy with Adaptive Boosting:", boost_reg.score(train_tfidf, y_train), "\nValidation accuracy with Adaptive Boosting:",
      boost_reg.score(test_tfidf, y_test))
"""
*======Random Forest Output=======* 
Training accuracy with Adaptive Boosting: 0.9997555555555555 
Validation accuracy with Adaptive Boosting: 0.8898
"""

# confusion matrix 
# f1-score :0.89
rfc_predict=boost_reg.predict(test_tfidf)
cm=confusion_matrix(y_test,rfc_predict)
cr=classification_report(y_test,rfc_predict)
print('Classification report is:\n',cr)
# AUC score
# AUC score: 0.89
fpr_rf,tpr_rf,threshold_rf=roc_curve(y_test,rfc_predict)
auc_rf=auc(fpr_rf,tpr_rf)
print('AUC score for Current Random Forest classifier (with TF-IDF Representation):',np.round(auc_rf,3))
