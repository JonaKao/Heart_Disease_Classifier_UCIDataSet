import pandas as pd
import sklearn as sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split #this allows me to split data into training and testing sets. otherwise i would have horrible generalization
from sklearn.metrics import accuracy_score, classification_report #this allows me to evaluate the model after training it
from sklearn.model_selection import GridSearchCV #this allows me to do hyperparameter tuning (finding the best parameters for the model). It is a very basic version but I realized that adjusting my estimate and learning rate gave me 3% of accuracy. So let's brute force it
from xgboost import XGBClassifier
from ucimlrepo import fetch_ucirepo 

xgb_model = XGBClassifier(
    n_estimators=50, #number of revamps of decision trees
    learning_rate=0.1, #approach value, this is how much we learn from each tree. 0.1 seems good for most cases
    max_depth=5, #can experiment with this value, this is just how deep into the decision trees we go (raising it led to greater precision for the more common classes -> potential overfit?), raising it even higher 4->8, completely eradicated class 1
    random_state=42, #random seed for later reproduction
    use_label_encoder=False, #this is a warning that we can ignore for now
    eval_metric='logloss', #this is the metric we will use to evaluate our model, logloss is a good default for (binary) classification problems
    subsample=1.0 #this is the fraction of samples to be used for each tree, 0.8 is a good default value, but according to the optimization of the grid we are using 1
)

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set the hyperparameters to search over
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
}
# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
# Fit the model with grid search
grid_search.fit(X_train, y_train)
# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)
# Best score
print("Best Score:", grid_search.best_score_)


# xgb_model.fit(X_train, y_train) #fit the model to the training data
# predictions = xgb_model.predict(X_test) #make predictions on the test data

# print("Accuracy:", accuracy_score(y_test, predictions)) #evaluate the model using accuracy score
# print(classification_report(y_test, predictions)) #evaluate the model using classification report
