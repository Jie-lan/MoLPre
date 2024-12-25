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


#######validation of contructed models
new_X=dat[['Age','Sex','Nodules_numbers','Nodule_diameter','Nodule_location','CEA','NSE','CYFRA21-1','SCC']]
new_y=dat['Class']
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(new_X, new_y, test_size=validation_size, random_state=25)

import pickle
filename="../preconstructed/pulmonary_nodule_best_RF.sav"
#loading the model using pickle
loaded_classifier=pickle.load(open(filename,'rb'))
print(X_test_new)
y_pred2_train=loaded_classifier.predict(X_train_new)
y_pred2_test=loaded_classifier.predict(X_test_new)

#evaluating the algorithm on test set
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("\n train-set score:{:.5f} is:",accuracy_score(y_train_new,y_pred2_train))
print(confusion_matrix(y_test_new,y_pred2_test))
print(classification_report(y_test_new,y_pred2_test))
print(accuracy_score(y_test_new,y_pred2_test))

y_train_new=np.where(y_train_new=="metastasis", 0,1)
y_pred2_train=np.where(y_pred2_train=="metastasis", 0,1)
y_test_new=np.where(y_test_new=="metastasis", 0,1)
y_pred2_test=np.where(y_pred2_test=="metastasis", 0,1)

print('Validation of RF')
auc = metrics.roc_auc_score(y_test_new, y_pred2_test)
accuracy = metrics.accuracy_score(y_test_new, y_pred2_test)
Recall = metrics.recall_score(y_test_new, y_pred2_test)
Prec = metrics.precision_score(y_test_new, y_pred2_test)
f1 = metrics.f1_score(y_test_new, y_pred2_test)

cfmetric = confusion_matrix(y_test_new, y_pred2_test, labels=[0,1])
tn, fp, fn, tp = cfmetric.ravel()
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
RF_results={"Accuracy":round(accuracy*100.0,2),"Precision":round(Prec*100.0,2),
            "Spec":round(specificity*100.0,2),"F1-score":round(f1*100.0,2),"Sens":round(sensitivity*100.0,2)}

#######AdaBoosting
model_ada = AdaBoostClassifier(random_state=123)
model_ada.fit(X_train_new,y_train_new)

print('Validation of Ada')
y_pred2_test=model_ada.predict(X_test_new)
auc = metrics.roc_auc_score(y_test_new, y_pred2_test)
accuracy = metrics.accuracy_score(y_test_new, y_pred2_test)
Recall = metrics.recall_score(y_test_new, y_pred2_test)
Prec = metrics.precision_score(y_test_new, y_pred2_test)
f1 = metrics.f1_score(y_test_new, y_pred2_test)

cfmetric = confusion_matrix(y_test_new, y_pred2_test, labels=[0, 1])
tn, fp, fn, tp = cfmetric.ravel()
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
Ada_results={"Accuracy":round(accuracy*100.0,2),"Precision":round(Prec*100.0,2),
            "Spec":round(specificity*100.0,2),"F1-score":round(f1*100.0,2),"Sens":round(sensitivity*100.0,2)}

#######GradientBoosting
model_gb = GradientBoostingClassifier(learning_rate=0.01, loss='exponential',
                           n_estimators=80,subsample=0.9,random_state=123)
model_gb.fit(X_train_new,y_train_new)

y_pred2_test=model_gb.predict(X_test_new)

print('Validation of Gradient')
auc = metrics.roc_auc_score(y_test_new, y_pred2_test)
accuracy = metrics.accuracy_score(y_test_new, y_pred2_test)
Recall = metrics.recall_score(y_test_new, y_pred2_test)
Prec = metrics.precision_score(y_test_new, y_pred2_test)
f1 = metrics.f1_score(y_test_new, y_pred2_test)

cfmetric = confusion_matrix(y_test_new, y_pred2_test, labels=[1, 0])
tn, fp, fn, tp = cfmetric.ravel()
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
GB_results={"Accuracy":round(accuracy*100.0,2),"Precision":round(Prec*100.0,2),
            "Spec":round(specificity*100.0,2),"F1-score":round(f1*100.0,2),"Sens":round(sensitivity*100.0,2)}

results=[RF_results,GB_results,Ada_results]
print(results)
#########SHAP for 3 classifier
###SHAP for RF
explainer = shap.Explainer(loaded_classifier)
shap_values = explainer(X_test_new)

###SHAP for Ada
explainer = shap.Explainer(model_ada)
shap_values = explainer.shap_values(X_test_new)

###SHAP for GB
explainer = shap.Explainer(model_gb)
shap_values = explainer(X_test_new)
