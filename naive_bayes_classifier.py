"""
B. Naive Bayesian classifier
Question 1, 2 & 4

"""

import numpy as np
from sklearn import datasets

def fit(num_classes, X):
    classes_mean = {}
    classes_variance = {}
    classes_prior = {}

    for c in range(num_classes):
        X_c = X[y == c]
        
        classes_mean[str(c)] = np.mean(X_c, axis=0)
        classes_variance[str(c)] = np.var(X_c, axis=0)
        classes_prior[str(c)] = X_c.shape[0] / X.shape[0]
    
    return [classes_mean , classes_variance, classes_prior]

def predict(model, stats, X):
    classes_mean = model[0]
    classes_variance = model[1]
    classes_prior = model[2]

    num_examples = stats[0]
    num_features = stats[1]
    num_classes = stats[2]

    probs = np.zeros((num_examples, num_classes))

    for c in range(num_classes):
        prior = classes_prior[str(c)]
        probs_c = density_function(
            X, num_features, classes_mean[str(c)], classes_variance[str(c)]
        )
        probs[:, c] = probs_c + np.log(prior)

    return np.argmax(probs, 1)

def density_function(x, num_features, mean, sigma):
    eps = 1e-6
    # Calculate probability from Gaussian density function
    const = -num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(
        np.log(sigma + eps)
    )
    probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + eps), 1)
    return const - probs

def CBN(X, y):
    num_examples,num_features = X.shape
    num_classes = len(np.unique(y))

    model = fit(num_classes, X)

    stats = [num_examples, num_features, num_classes]
    y_pred = predict(model, stats, X)

    i = 0
    correctPrediction = 0
    wrongPrediction = 0

    for pred in y_pred:
        test = y[i]
        if (test == pred):
            correctPrediction = correctPrediction + 1
            status = 'correct'
        else:
            wrongPrediction = wrongPrediction + 1
            status = 'wrong'

        print('Test No',i,' SUMMARY')
        print('Actual : ' ,test, 'Prediction : ', pred, 'Status: ',status)

        i = i + 1

    #percentageError = (1 - (sum(y_pred==y)/X.shape[0])) * 100

    percentageError = (wrongPrediction/(wrongPrediction + correctPrediction)) * 100

    print("Percentage: ", round(percentageError, 2))


iris = datasets.load_iris ()
X = iris.data
y = iris.target

CBN(X, y)