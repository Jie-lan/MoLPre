import re
import pandas as pd
import numpy as np
import matplotlib.pylab as plt,savefig
from pylab import mpl
from sklearn import set_config
set_config(display = 'diagram')
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold,train_test_split
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from functools import reduce
warnings.filterwarnings('ignore')

###load data and remove samples with NA
dat = pd.read_csv("pulmonary_data.csv")
dat = dat.dropna()
X = dat.drop(labels=['Class', 'Sample', 'Stage'], axis=1)
y = dat["Class"]

selected_features = ['Age', 'Sex', 'Nodules_numbers', 'Nodule_diameter', 'Nodule_location', 'CEA', 'NSE', 'CYFRA21-1',
                     'SCC']
x = X[selected_features]

###split data into training and validation
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=25, train_size=0.7)

search_grid1 = [ {'model': [RandomForestClassifier(random_state=123)],
     'model__criterion': ['entropy', 'gini'],
     'model__max_depth': list(range(1, 20)) + [None],
     'model__min_samples_leaf': list(range(1, 10)),
     'model__n_estimators': [20, 40, 60, 80, 100, 120, 140, 200]},
]

pipeline1 = Pipeline([('model', RandomForestClassifier())])
clf_RF = GridSearchCV(pipeline1, search_grid1, cv=StratifiedKFold(n_splits=10),scoring='accuracy',
                   n_jobs=-1, verbose=1)

clf_RF = clf_RF.fit(x_train, y_train)
print("Best parameters of RandomForest: ",clf_RF.best_params_)
print("Best accuracy of RandomForest: ",clf_RF.best_score_)

search_grid2 = [{'model':[AdaBoostClassifier(random_state=123)],
    'model__n_estimators':[20,40,60,80,100,120,140,200],
    'model__learning_rate':[0.01,0.05,0.1,0.5,0.6,0.7,0.8,0.9,1],
    'model__algorithm':['SAMME','SAMME.R']}]

pipeline2 = Pipeline([('model',AdaBoostClassifier())])
clf_Ada = GridSearchCV(pipeline2, search_grid2, cv=StratifiedKFold(n_splits=10),scoring='accuracy',
                   n_jobs=-1, verbose=1)

clf_Ada = clf_Ada.fit(x_train, y_train)
print("Best parameters of AdaBoost: ",clf_Ada.best_params_)
print("Best accuracy of AdaBoost: ",clf_Ada.best_score_)

search_grid3 = [{'model':[GradientBoostingClassifier(random_state=123)],
     'model__n_estimators':[80,100,120,140,200],
     'model__learning_rate':[0.01,0.05,0.1,0.5,0.6,0.7,0.8,0.9,1],
     'model__loss':['deviance','exponential'],
     'model__subsample':[0.7,0.8,0.9,1]}]

pipeline3 = Pipeline([('model',GradientBoostingClassifier())])
clf_GB = GridSearchCV(pipeline3, search_grid3,cv=StratifiedKFold(n_splits=10),scoring='accuracy',
                   n_jobs=-1,verbose=1)

clf_GB = clf_GB.fit(x_train, y_train)
print("Best parameters of GradientBoosting: ",clf_GB.best_params_)
print("Best accuracy of GradientBoosting: ",clf_GB.best_score_)
