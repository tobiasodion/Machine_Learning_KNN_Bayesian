"""
A Nearest neighbor
Question 4

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris ()
X = iris.data
Y = iris.target

correctPrediction = 0
wrongPrediction = 0
status = 'correct'
i = 0

loo = LeaveOneOut()
loo.get_n_splits(X)

for train_index, test_index in loo.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    #print('Data : ' , X_test)

    #scaler = StandardScaler()
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)

    #print(X_train, X_test, Y_train, Y_test)

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)

    print('Prediction : ', y_pred, ':', 'Actual : ' ,Y[test_index])

    if (Y[test_index] == y_pred):
        correctPrediction = correctPrediction + 1
        status = 'correct'
    else:
        wrongPrediction = wrongPrediction + 1
        status = 'wrong'
    
    i = i + 1

    print('Test No',i,' SUMMARY')
    print('Test Data: ',X_test, ' Actual Label: ',Y[test_index], ' Predicted Label: ',y_pred, 'Status: ',status)


percentageError = (wrongPrediction/(wrongPrediction + correctPrediction)) * 100

print('MODEL SUMMARY')
print('Percentage Badly Predicted: ', round(percentageError,2), '%')


