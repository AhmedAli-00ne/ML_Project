from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import joblib


def RFTrain():
    df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
    X = df.drop(['Diabetes_binary','Education','Income'] , axis = 1)
    y = np.array(df['Diabetes_binary'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'RF_model.pkl')
    
    y_pred = clf.predict(X_test)
    
    return y_pred, y_test