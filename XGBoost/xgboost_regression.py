# Linear Regression using sklearn

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.grid_search import GridSearchCV


def train():
    
    # Importing the dataset
    dataset = pd.read_csv('dataSets/Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    
    sc_Y = StandardScaler()
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    y_train = sc_Y.fit_transform(y_train)
    y_test = sc_Y.transform(y_test)
    
    
    # Configuring the input paramaters for the grid search
    cv_params = {'learning_rate': [0.001, 0.01,0.1], 'gamma': [0,1,2], 'subsample': [0.75,0.85]}
    ind_params = {'n_estimators': 1000, 'seed':0, 'colsample_bytree': 1, 
                 'objective': 'reg:linear', 'max_depth': 30, 'min_child_weight': 1}
    
    
    optimized_GBM = GridSearchCV(xgb.XGBRegressor(**ind_params), 
                                cv_params, 
                                 scoring = 'mean_squared_error', cv = 10, 
                                 n_jobs = -1,
                                 verbose=2)
    
    # Fitting XGBoost optimized regressor to the data
    optimized_GBM.fit(X_train, y_train)
    
	# Print the results of the grid search
    print(optimized_GBM.grid_scores_)
    
    # Using the optimized model
    regressor = xgb.XGBRegressor(learning_rate=0.01, subsample=0.85, gamma= 0)
    
    # fitting the data
    regressor.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    
    # looking at the mse
    from sklearn.metrics import mean_squared_error
    print('MSE:', mean_squared_error(y_test,y_pred))
    
    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = regressor, 
                                 X = X_train, 
                                 y = y_train, 
                                 cv = 5, 
                                 scoring='neg_mean_absolute_error')
    
    print('k-Fold Cross Validation')
    print('Mean accuracy:',abs(accuracies.mean()))
    print('Standard diviation:',accuracies.std())

    
if __name__ == "__main__":
    
    train()