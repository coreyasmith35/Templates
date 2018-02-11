# XGBoost Classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def train():
    
    # Importing the dataset
    dataset = pd.read_csv('../DataSets/Churn_Modelling.csv')
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values
    
    # Encoding categorical data
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
    onehotencoder = OneHotEncoder(categorical_features = [1])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:, 1:]
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    # Configuring the input paramaters for the grid search
    cv_params = {'learning_rate': [0.001, 0.01], 'subsample': [0.8,0.85,0.9]}
    ind_params = {'n_estimators': 1000, 'seed':0, 'colsample_bytree': 0.8, 
                 'objective': 'binary:logistic', 'max_depth': 35, 'min_child_weight': 1}
    
    
    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                                cv_params, 
                                 scoring = 'accuracy', cv = 5, 
                                 n_jobs = -1,
                                 verbose=2)
    
    # Fitting XGBoost optimized clasifier to the data
    optimized_GBM.fit(X_train, y_train)
    
	# Print the results of the grid search
    print(optimized_GBM.grid_scores_)

    # Using the optimized model
    classifier = xgb.XGBClassifier(learning_rate=.001, subsample=.85)
    
    # fitting the data
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(cm)
    
    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print('k-Fold Cross Validation')
    print('Mean accuracy:',accuracies.mean())
    print('Standard diviation:',accuracies.std())
  

if __name__ == "__main__":
    
    train()