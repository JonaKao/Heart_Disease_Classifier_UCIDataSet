import pandas as pd
import numpy as np
import sklearn as sklearn
from sklearn.model_selection import train_test_split #this allows me to split data into training and testing sets. otherwise i would have horrible generalization
from sklearn.metrics import accuracy_score, classification_report #this allows me to evaluate the model after training it
from xgboost import XGBClassifier
from ucimlrepo import fetch_ucirepo 
from hyperopt import hp, fmin, tpe, Trials

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(params):
    model = XGBClassifier(
        n_estimators=int(params['n_estimators']),
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        subsample=params['subsample']
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return -accuracy  # Since Hyperopt minimizes, we negate the accuracy
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'subsample': hp.uniform('subsample', 0.7, 1.0)
}

trials = Trials()

best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,  # Number of evaluations
    trials=trials
)

print("Best hyperparameters:", best)