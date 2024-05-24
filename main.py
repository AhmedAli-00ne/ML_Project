import tkinter as tk
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn import metrics
from MLP import MLPTrain
from RandomForest import RFTrain
from SVM import SVMTrain
from KNN import KNNTrain
import matplotlib.pyplot as plt

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Machine Learning Project")
        self.geometry("800x400")
        self.resizable(False, False)
        self.StartScreen()
    def ConfusionMatrix(self,cm):
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["not diabetes", "diabetes"])
        cm_display.plot()
        plt.show()
    def KNN(self, num):
        if num == 0:
            self.records = self.RecordNumber.get()
            self.clear_wndow()
            self.records = self.records.split(",")
            self.records = list(map(float, self.records))
            df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
            df = df.drop(['Diabetes_binary','Education','Income'] , axis = 1)
            records_data = df.iloc[self.records]
            self.model = joblib.load('knn_model.pkl')
            if len(self.records) == 1:
                records_data = records_data.values.reshape(1, -1)
            self.prediction = self.model.predict(records_data)
            self.Header = tk.Label(self, text="Prediction Result", font=("Arial", 20))
            self.Header.pack()
            self.result = tk.Label(self, text=self.prediction, font=("Arial", 20))
            self.result.pack()
            self.BackButton = tk.Button(self, text="Back", command=lambda: self.DataInput(4),bg = 'black', fg = 'white', width=110)
            self.BackButton.pack()
        else:
            self.y_pred, self.y_test = KNNTrain()
            self.clear_wndow()
            self.Header = tk.Label(self, text="Training Result", font=("Arial", 20))
            self.Header.pack()
            self.accuracy = tk.Label(self, text="Accuracy: " + str(accuracy_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.accuracy.pack()
            self.recall = tk.Label(self, text="Recall: " + str(recall_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.recall.pack()
            self.precision = tk.Label(self, text="Precision: " + str(precision_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.precision.pack()
            self.f1 = tk.Label(self, text="F1: " + str(f1_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.f1.pack()
            self.confusion_matrixButton = tk.Button(self, text="Confusion Matrix", command=lambda: self.ConfusionMatrix(confusion_matrix(self.y_test, self.y_pred)),bg = 'black', fg = 'white', width=110)
            self.confusion_matrixButton.pack()
            self.BackButton = tk.Button(self, text="Back", command=lambda: self.DataInput(4),bg = 'black', fg = 'white', width=110)
            self.BackButton.pack()
    def SVM(self, num):
        if num == 0:
            self.records = self.RecordNumber.get()
            self.clear_wndow()
            self.records = self.records.split(",")
            self.records = list(map(float, self.records))
            df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
            df = df.drop(['Diabetes_binary','Education','Income'] , axis = 1)
            records_data = df.iloc[self.records]
            self.model = joblib.load('svm_model.pkl')
            if len(self.records) == 1:
                records_data = records_data.values.reshape(1, -1)
            self.prediction = self.model.predict(records_data)
            self.Header = tk.Label(self, text="Prediction Result", font=("Arial", 20))
            self.Header.pack()
            self.result = tk.Label(self, text=self.prediction, font=("Arial", 20))
            self.result.pack()
            self.BackButton = tk.Button(self, text="Back", command=lambda: self.DataInput(1),bg = 'black', fg = 'white', width=110)
            self.BackButton.pack()
        else:
            self.y_pred, self.y_test = SVMTrain()
            self.clear_wndow()
            self.Header = tk.Label(self, text="Training Result", font=("Arial", 20))
            self.Header.pack()
            self.accuracy = tk.Label(self, text="Accuracy: " + str(accuracy_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.accuracy.pack()
            self.recall = tk.Label(self, text="Recall: " + str(recall_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.recall.pack()
            self.precision = tk.Label(self, text="Precision: " + str(precision_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.precision.pack()
            self.f1 = tk.Label(self, text="F1: " + str(f1_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.f1.pack()
            self.confusion_matrixButton = tk.Button(self, text="Confusion Matrix", command=lambda: self.ConfusionMatrix(confusion_matrix(self.y_test, self.y_pred)),bg = 'black', fg = 'white', width=110)
            self.confusion_matrixButton.pack()
            self.BackButton = tk.Button(self, text="Back", command=lambda: self.DataInput(1),bg = 'black', fg = 'white', width=110)
            self.BackButton.pack()

    def RF(self, num):
        if num ==0:
            self.records = self.RecordNumber.get()
            self.clear_wndow()
            self.records = self.records.split(",")
            self.records = list(map(float, self.records))
            df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
            df = df.drop(['Diabetes_binary','Education','Income'] , axis = 1)
            records_data = df.iloc[self.records]
            self.model = joblib.load('RF_model.pkl')
            if len(self.records) == 1:
                records_data = records_data.values.reshape(1, -1)
            self.prediction = self.model.predict(records_data)
            self.Header = tk.Label(self, text="Prediction Result", font=("Arial", 20))
            self.Header.pack()
            self.result = tk.Label(self, text=self.prediction, font=("Arial", 20))
            self.result.pack()
            self.BackButton = tk.Button(self, text="Back", command=lambda: self.DataInput(2),bg = 'black', fg = 'white', width=110)
            self.BackButton.pack()
        else:
            self.y_pred, self.y_test = RFTrain()
            self.clear_wndow()
            self.Header = tk.Label(self, text="Training Result", font=("Arial", 20))
            self.Header.pack()
            self.accuracy = tk.Label(self, text="Accuracy: " + str(accuracy_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.accuracy.pack()
            self.recall = tk.Label(self, text="Recall: " + str(recall_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.recall.pack()
            self.precision = tk.Label(self, text="Precision: " + str(precision_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.precision.pack()
            self.f1 = tk.Label(self, text="F1: " + str(f1_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.f1.pack()
            self.confusion_matrixButton = tk.Button(self, text="Confusion Matrix", command=lambda: self.ConfusionMatrix(confusion_matrix(self.y_test, self.y_pred)),bg = 'black', fg = 'white', width=110)
            self.confusion_matrixButton.pack()
            self.BackButton = tk.Button(self, text="Back", command=lambda: self.DataInput(2),bg = 'black', fg = 'white', width=110)
            self.BackButton.pack()
    def MLP(self, num):
        if num == 0:
            self.records = self.RecordNumber.get()
            self.clear_wndow()
            self.records = self.records.split(",")
            self.records = list(map(float, self.records))
            df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
            df = df.drop(['Diabetes_binary','Education','Income'] , axis = 1)
            records_data = df.iloc[self.records]
            self.model = joblib.load('MLP_model.pkl')
            if len(self.records) == 1:
                records_data = records_data.values.reshape(1, -1)
            self.prediction = self.model.predict(records_data)
            self.Header = tk.Label(self, text="Prediction Result", font=("Arial", 20))
            self.Header.pack()
            self.result = tk.Label(self, text=self.prediction, font=("Arial", 20))
            self.result.pack()
            self.BackButton = tk.Button(self, text="Back", command=lambda: self.DataInput(0),bg = 'black', fg = 'white', width=110)
            self.BackButton.pack()
        else:
            self.y_pred, self.y_test = MLPTrain()
            self.clear_wndow()
            self.Header = tk.Label(self, text="Training Result", font=("Arial", 20))
            self.Header.pack()
            self.accuracy = tk.Label(self, text="Accuracy: " + str(accuracy_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.accuracy.pack()
            self.recall = tk.Label(self, text="Recall: " + str(recall_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.recall.pack()
            self.precision = tk.Label(self, text="Precision: " + str(precision_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.precision.pack()
            self.f1 = tk.Label(self, text="F1: " + str(f1_score(self.y_test, self.y_pred)), font=("Arial", 20))
            self.f1.pack()
            self.confusion_matrixButton = tk.Button(self, text="Confusion Matrix", command=lambda: self.ConfusionMatrix(confusion_matrix(self.y_test, self.y_pred)),bg = 'black', fg = 'white', width=110)
            self.confusion_matrixButton.pack()
            self.BackButton = tk.Button(self, text="Back", command=lambda: self.DataInput(0),bg = 'black', fg = 'white', width=110)
            self.BackButton.pack()
    def DataInput(self,num):
        self.clear_wndow()
        self.RecordNumber = tk.Entry(self)
        if num == 0 or num == 1 or num == 2 or num == 4:
            self.Header = tk.Label(self, text="Choose a record to test", font=("Arial", 20))
            self.Header.pack()
            self.RecordNumber.pack()
        else:
            self.Header = tk.Label(self, text="Choose an image to test", font=("Arial", 20))
            self.Header.pack()
        if num == 0:
            self.PredectionButton = tk.Button(self, text="Predict", command=lambda: self.MLP(0),bg = 'black', fg = 'white', width=110)
            self.TrainButton = tk.Button(self, text="Train", command=lambda: self.MLP(1),bg = 'black', fg = 'white', width=110)
        elif num == 1:
            self.PredectionButton = tk.Button(self, text="Predict", command=lambda: self.SVM(0),bg = 'black', fg = 'white', width=110)
            self.TrainButton = tk.Button(self, text="Train", command=lambda: self.SVM(1),bg = 'black', fg = 'white', width=110)
        elif num == 2:
            self.PredectionButton = tk.Button(self, text="Predict", command=lambda: self.RF(0),bg = 'black', fg = 'white', width=110)
            self.TrainButton = tk.Button(self, text="Train", command=lambda: self.RF(1),bg = 'black', fg = 'white', width=110)
        elif num == 3:
            self.PredectionButton = tk.Button(self, text="Predict", command=lambda: self.VGG(0),bg = 'black', fg = 'white', width=110)
            self.TrainButton = tk.Button(self, text="Train", command=lambda: self.VGG(1),bg = 'black', fg = 'white', width=110)
        elif num == 4:
            self.PredectionButton = tk.Button(self, text="Predict", command=lambda: self.KNN(0),bg = 'black', fg = 'white', width=110)
            self.TrainButton = tk.Button(self, text="Train", command=lambda: self.KNN(1),bg = 'black', fg = 'white', width=110)
        self.PredectionButton.place(x=10, y=70)
        self.TrainButton.place(x=10, y=110)
        self.BackButton = tk.Button(self, text="Back", command=lambda: self.StartScreen(),bg = 'black', fg = 'white', width=110)
        self.BackButton.place(x=10, y=150)
    def StartScreen(self):
        self.clear_wndow()
        self.Header = tk.Label(self, text="Machine Learning Project", font=("Arial", 20))
        self.Header.pack()
        self.VGGButton = tk.Button(self, text="VGG16", command=lambda: self.DataInput(3),bg = 'black', fg = 'white', width=110)
        self.VGGButton.place(x=10, y=40)
        self.SVMButton = tk.Button(self, text="Support Vector Machine", command=lambda: self.DataInput(1),bg = 'black', fg = 'white', width=110)
        self.SVMButton.place(x=10, y=80)
        self.MLPButton = tk.Button(self, text="Multi Layer Perceptron", command=lambda: self.DataInput(0),bg = 'black', fg = 'white', width=110)
        self.MLPButton.place(x=10, y=120)
        self.RFButton = tk.Button(self, text="Random Forest", command=lambda: self.DataInput(2),bg = 'black', fg = 'white', width=110)
        self.RFButton.place(x=10, y=160)
        self.KNNButton = tk.Button(self, text="K Nearest Neighbors", command=lambda: self.DataInput(4),bg = 'black', fg = 'white', width=110)
        self.KNNButton.place(x=10, y=200)
    def clear_wndow(self):
        for widget in self.winfo_children():
            widget.destroy()

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()