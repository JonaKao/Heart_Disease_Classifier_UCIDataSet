import pandas as pd
import numpy as np
import sklearn as sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split #this allows me to split data into training and testing sets. otherwise i would have horrible generalization
from sklearn.metrics import accuracy_score, classification_report #this allows me to evaluate the model after training it
from xgboost import XGBClassifier
from ucimlrepo import fetch_ucirepo 
from scipy.stats import uniform, randint

XGBClassifier(
    base_score=0.5, 
    booster='gbtree', 
    colsample_bylevel=1,
    colsample_bynode=1, 
    colsample_bytree=1, 
    enable_categorical=False,
    gamma=0, gpu_id=-1, 
    importance_type=None,
    interaction_constraints='', 
    learning_rate=0.300000012,
    max_delta_step=0, 
    ax_depth=6, 
    min_child_weight=1, 
    missing=np.nan,
    monotone_constraints='()', 
    n_estimators=100, 
    n_jobs=4,
    num_parallel_tree=1, 
    predictor='auto', 
    random_state=0,
    reg_alpha=0, reg_lambda=1, 
    scale_pos_weight=1, 
    subsample=1,
    tree_method='exact',
    use_label_encoder=False,
    validate_parameters=1,
    verbosity=None)

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
xgb_model = XGBClassifier()

# # Define the distribution for hyperparameters
# param_dist = {
#     'n_estimators': randint(50, 200),
#     'learning_rate': uniform(0.01, 0.3),
#     'max_depth': randint(3, 10),
#     'subsample': uniform(0.7, 0.3),
# }

# # Set up RandomizedSearchCV with 5-fold cross-validation
# random_search = RandomizedSearchCV(xgb_model, param_dist, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# # Fit the model with random search
# random_search.fit(X_train, y_train)

# # Best hyperparameters
# print("Best Hyperparameters:", random_search.best_params_)

# # Best score
# print("Best Score:", random_search.best_score_)

xgb_model.fit(X_train, y_train) #fit the model to the training data
predictions = xgb_model.predict(X_test) #make predictions on the test data

print("Accuracy:", accuracy_score(y_test, predictions)) #evaluate the model using accuracy score
print(classification_report(y_test, predictions)) #evaluate the model using classification report
