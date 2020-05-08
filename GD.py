# read data from file

import matplotlib as plt


import csv
import os

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import myGDBatch
import myGDBatchBi
import myNormalisation
from normalisationSS import normalisation
from plot import plot3Ddata


def loadData(fileName, inputVariabName1, inputVariabName2, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable1 = dataNames.index(inputVariabName1)
    selectedVariable2 = dataNames.index(inputVariabName2)
    inputs = [[float(data[i][selectedVariable1]), float(data[i][selectedVariable2])] for i in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]

    return inputs, outputs


def main():
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'world-happiness-report-2017.csv')

    inputs, outputs = loadData(filePath, 'Economy..GDP.per.Capita.', 'Freedom', 'Happiness.Score')
    print('in:  ', inputs[:5])
    print('out: ', outputs[:5])

    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]
    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]
    feature1 = [ex[0] for ex in inputs]
    feature2 = [ex[1] for ex in inputs]
    # data normalisation for train and test data
    # using my normalisation
    feature1train = [ex[0] for ex in trainInputs]
    feature2train = [ex[1] for ex in trainInputs]
    feature1test = [ex[0] for ex in testInputs]
    feature2test = [ex[1] for ex in testInputs]
    ## trainInputs, testInputs = normalisation(trainInputs, testInputs)
    ## trainOutputs, testOutputs = normalisation(trainOutputs, testOutputs)
    meanVal1, stdVal1 = myNormalisation.statisticalNormalisation(feature1train)
    meanVal2, stdVal2 = myNormalisation.statisticalNormalisation(feature2train)
    feature1normalisedTrain = myNormalisation.normalisation(feature1train, meanVal1, stdVal1)
    feature2normalisedTrain = myNormalisation.normalisation(feature2train, meanVal2, stdVal2)
    feature1normalisedTest = myNormalisation.normalisation(feature1test, meanVal1, stdVal1)
    feature2normalisedTest = myNormalisation.normalisation(feature2test, meanVal2, stdVal2)
    trainInputs = [[feature1normalisedTrain[i], feature2normalisedTrain[i]] for i in
                   range(len(feature1normalisedTrain))]
    testInputs = [[feature1normalisedTest[i], feature2normalisedTest[i]] for i in range(len(feature1normalisedTest))]

    # ######GD BATCH REGRESSION WITH SKLEARN######

    xx = [el for el in trainInputs]
    regressor = linear_model.SGDRegressor()
    # make SGDregressor into GD Batch regressor

    # regressor.fit(xx, trainOutputs)
    # w0, w1, w2 = regressor.intercept_[0], regressor.coef_[0], regressor.coef_[1]
    # print('the learnt model with sklearn norm: f(x) = ', w0, ' + ', w1, ' * x', ' +', w2, ' * x^2')

    # computedTestOutputs = regressor.predict([x for x in testInputs])

    # error = mean_squared_error(testOutputs, computedTestOutputs)
    # print("prediction error (tool): ", error)
    nr_iter = 10000
    for i in range(0, nr_iter):
        regressor.partial_fit(xx, trainOutputs)

    w0, w1, w2 = regressor.intercept_[0], regressor.coef_[0], regressor.coef_[1]
    print('the learnt model with sklearn : f(x) = ', w0, ' + ', w1, ' * x', ' +', w2, ' * x^2')
    computedTestOutputs = regressor.predict([x for x in testInputs])
    error = mean_squared_error(testOutputs, computedTestOutputs)
    print("prediction error (tool): ", error)

    # ######GD BATCH REGRESSION USING MY OWN REGRESSION ######
    myBatchRegressor = myGDBatch.MyGDregressor()
    myBatchRegressor.fit(xx, trainOutputs)
    w0, w1, w2 = myBatchRegressor.intercept_, myBatchRegressor.coef_[0], myBatchRegressor.coef_[1]
    print('the learnt model with my regression : f(x) = ', w0, ' + ', w1, ' * x', ' +', w2, ' * x^2')
    computedTestOutputs = myBatchRegressor.predict([x for x in testInputs])

    # my prediction error
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        error += (t1 - t2) ** 2
    error = error / len(testOutputs)
    print("prediction error (manual): ", error)

    # numerical representation of the regressor model
    noOfPoints = 50
    xref1 = []
    val = min(feature1)
    step1 = (max(feature1) - min(feature1)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref1.append(val)
        val += step1

    xref2 = []
    val = min(feature2)
    step2 = (max(feature2) - min(feature2)) / noOfPoints
    for _ in range(1, noOfPoints):
        aux = val
        for _ in range(1, noOfPoints):
            xref2.append(aux)
            aux += step2
    yref = [w0 + w1 * el1 + w2 * el2 for el1, el2 in zip(xref1, xref2)]
    plot3Ddata(feature1train, feature2train, trainOutputs, xref1, xref2, yref, [], [], [],
               'train data and the learnt model')
    plot3Ddata([], [], [], feature1test, feature2test, computedTestOutputs, feature1test, feature2test, testOutputs,
               'predictions vs real test data')
main()
