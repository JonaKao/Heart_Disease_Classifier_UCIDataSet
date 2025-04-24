import pandas as pd
import numpy as np
import sklearn as sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split #this allows me to split data into training and testing sets. otherwise i would have horrible generalization
from sklearn.metrics import accuracy_score, classification_report #this allows me to evaluate the model after training it
from xgboost import XGBClassifier
from ucimlrepo import fetch_ucirepo 

XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.047332838087974255,
    subsample=0.9981446094951785,
    use_label_encoder=False,
    eval_metric='logloss'
)

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
xgb_model = XGBClassifier()

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))