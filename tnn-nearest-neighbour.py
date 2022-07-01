"""
A Nearest neighbor
Question 1, 2, 3 & 5

"""
import numpy as np
from sklearn import datasets, metrics

def leaveOneOutSplit(X, Y):
    #split given data into testData, testLabel and traingData, trainingLabel
    i = 0
    splitDataSet = []
    
    for data in X:
        testData = X[i]
        testLabel = Y[i]

        trainingData = np.delete(X, i, axis=0)
        trainingLabel = np.delete(Y, i)

        #print(trainingData)
        #print(trainingLabel)
        #trainingData = X[0:i] + X[i + 1 :len(X)+1]
        #trainingLabel = Y[0:i] + Y[i + 1 :len(Y)+1]

        i = i+1
        splitDataSet.append([testData, testLabel, trainingData, trainingLabel])
        #print([testData, testLabel, trainingData, trainingLabel])

    return splitDataSet

def nearestNeighbourModel(testData, trainingData, trainingLabel):
    i=0
    analysisSet = []
    analysisSetItem = []

    #build a model for the KNN analysis using the distance between test data and a set of training data
    for data in trainingData:
        
        #find the euclidean distance of the testdata from each element in the training set
        eucliDistance = metrics.pairwise.euclidean_distances([testData], [data])
        #print(trainingLabel[i])
        #print(eucliDistance[0][0])

        analysisSetItem = [eucliDistance[0][0],trainingLabel[i]]
        #print(analysisSetItem)
        analysisSet.append(analysisSetItem)
        i = i+1
    
    return analysisSet

def predictX(nNLabels):
    print(nNLabels) 
    prediction = np.bincount(nNLabels).argmax()

    #print(prediction)
    return prediction 
                
def nearestNeighborLabels(neighborsByDistance, k):
    #getting the kth nearest neighbours
    nNearestNeighbors = neighborsByDistance[0:k]
    #print(nNearestNeighbors)

    nNLabels = []
    #get the labels of the nearest neighbours
    for nn in nNearestNeighbors:
            distance, label = nn
            nNLabels.append(label)

    return nNLabels  


def TNN(X, Y):
    correctPrediction = 0
    wrongPrediction = 0
    status = 'correct'
    i = 0

    splitDataSet = leaveOneOutSplit(X,Y)
    for dataSet in splitDataSet:
        testData = dataSet[0]
        testLabel = dataSet[1]
        trainingData = dataSet[2]
        trainingLabel = dataSet[3]

        analysisModel = nearestNeighbourModel(testData, trainingData, trainingLabel)

        #sort the test data in increasing distance with respect to the training set
        neighborsByDistance = sorted(analysisModel, key=lambda x:x[0])
        #nearestNeighbor = np.argsort(analysisSet, axis=0)
        #print(neighborsByDistance)

        #select the K nearest neighbour to the test data
        nNLabel = nearestNeighborLabels(neighborsByDistance, 1)
        prediction = predictX(nNLabel)

        #check if prediction is right or wrong
        if (testLabel == prediction):
            correctPrediction = correctPrediction + 1
            status = 'correct'
        else:
            wrongPrediction = wrongPrediction + 1
            status = 'wrong'
        i = i + 1
        
        print('Test No',i,' SUMMARY')
        print('Test Data: ',testData, ' Actual Label: ',testLabel, ' Predicted Label: ',prediction, 'Status: ',status)


    percentageError = (wrongPrediction/(wrongPrediction + correctPrediction)) * 100

    print('MODEL SUMMARY')
    print('Percentage Badly Predicted: ', round(percentageError,2), '%')

iris = datasets.load_iris ()
X = iris.data
Y = iris.target


TNN(X,Y)