from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import csv
import numpy as np
import joblib

def SVMTrain():
    df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
    X = df.drop(['Diabetes_binary','Education','Income'] , axis = 1)
    y=np.array(df['Diabetes_binary'])
    print(df.shape)
    
    X , y = make_classification(n_samples=10000, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    svm = SVC(C=1.7, kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
    
    svm.fit(X_train, y_train)
    joblib.dump(svm, 'svm_model.pkl')
    
    y_pred = svm.predict(X_test)
    
    return y_pred, y_test