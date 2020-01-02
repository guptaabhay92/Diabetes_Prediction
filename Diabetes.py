# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 18:52:31 2020

@author: Abhay
"""

import pandas as pd
import numpy as np

df=pd.read_csv('pima-data.csv')
df.shape

df.isnull().values.any()

df.corr()

df.head(5)

diabetes_map={True:1, False:0}
df['diabetes']=df['diabetes'].map(diabetes_map)
df.head()

diabetes_true_count=len(df.loc[df['diabetes']==True])
diabetes_false_count=len(df.loc[df['diabetes']==False])

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=.20, random_state=0)

#how to check missing(zero) values

print("total number of rows {}".format(len(df)))
print("total number of missing glucose_conc : {}".format(len(df.loc[df['glucose_conc']==0])))
print("total number of missing diastolic_bp : {}".format(len(df.loc[df['diastolic_bp']==0])))
print("total number of missing insulin : {}".format(len(df.loc[df['insulin']==0])))
print("total number of mising bmi : {}".format(len(df.loc[df['bmi']==0])))
print("total number of missing diab_pred : {}".format(len(df.loc[df['diab_pred']==0])))
print("total number of missing age : {}".format(len(df.loc[df['age']==0])))
print("total number of mising skin : {}".format(len(df.loc[df['skin']==0])))

from sklearn.preprocessing import Imputer

fill_values=Imputer(missing_values=0, strategy='mean', axis=0)
X_train=fill_values.fit_transform(X_train)
X_test=fill_values.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)


from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)

# HyperParameter Optimization

para={"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
      "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
      "min_child_weight" : [ 1, 3, 5, 7 ],
      "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
      "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }


from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
classifier= XGBClassifier()

rscv=RandomizedSearchCV(classifier,param_distributions=para,n_iter=5,scoring='roc_auc',
                        n_jobs=-1,cv=5,verbose=3)

rscv.fit(X,y.ravel())
rscv.best_estimator_

classifier=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.0,
              learning_rate=0.2, max_delta_step=0, max_depth=8,
              min_child_weight=5, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

from sklearn.model_selection import cross_val_score
cvs=cross_val_score(classifier, X, y.ravel(), cv=10)

cvs.mean()
