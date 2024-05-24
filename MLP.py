import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
import joblib
import pandas as pd

def MLPTrain():
    df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

    X = df.drop(['Diabetes_binary','Education','Income'] , axis = 1)

    y = np.array(df['Diabetes_binary'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(100, ),        # Number of hidden layers and units per layer
                    activation='relu',                  # Activation function ('identity', 'logistic', 'tanh', 'relu')
                    solver='adam',                     # Solver for weight optimization ('lbfgs', 'sgd', 'adam')
                    alpha=0.0001,                      # L2 penalty (regularization term) parameter
                    batch_size='auto',                 # Size of minibatches for stochastic optimizers
                    learning_rate='constant',          # Learning rate schedule ('constant', 'invscaling', 'adaptive')
                    learning_rate_init=0.001,          # The initial learning rate
                    max_iter=200,                      # Maximum number of iterations
                    shuffle=True,                      # Whether to shuffle samples in each iteration
                    random_state=None,                 # Seed for the random number generator
                    tol=0.001,                          # Tolerance for the optimization
                    verbose=False,                     # Whether to print progress messages
                    beta_1=0.9,                        # Exponential decay rate for estimates of first moment vector in adam
                    beta_2=0.999,                      # Exponential decay rate for estimates of second moment vector in adam
                    epsilon=1e-8,                      # Value for numerical stability in adam
                    n_iter_no_change=20,               # Maximum number of epochs without any improvement in the loss
                    max_fun=15000)                     # Maximum number of function calls for the solver
    mlp.fit(X_train, y_train)
    joblib.dump(mlp, 'mlp_model.pkl')
    y_pred = mlp.predict(X_test)
    return y_pred, y_test