import pandas as pd
import shap
import numpy as np
import matplotlib.pylab as plt,savefig
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

dat = pd.read_csv("pulmonary_data.csv")
dat = dat.dropna()

X=dat.drop(['Class'],axis=1)
print(X.columns.tolist())
y=dat["Class"]
y = [1 if each == "metastasis" else 0 for each in y]

###split files
validation_size = 0.3
seed = 25 

import pandas as pd
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=25)

######
selected_features=['Age', 'Sex', 'Smoking', 'Environmental_exposure', 'Prior_cancer',
                   'Family_cancer_history', 'Nodules_numbers',
                   'Nodule_diameter', 'Nodule_location', 'CEA', 'Pro-GRP', 'NSE',
                   'CYFRA21-1', 'SCC', 'Histology', 'Micropapillary']

model_rf = RandomForestClassifier(random_state=123)
X_train=X_train[selected_features]
X_test=X_test[selected_features]
model_rf.fit(X_train, y_train) 
print(model_rf.feature_importances_)
print(X_train.columns.tolist())

###SHAP for RF
explainer = shap.Explainer(model_rf)
shap_values = explainer(X_train)



