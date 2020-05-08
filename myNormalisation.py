from statistics import stdev, mean

from sklearn.preprocessing import StandardScaler
from math import sqrt


def statisticalNormalisation(features):
    # meanValue = sum(features) / len(features)
    meanValue = mean(features)
    # stdDevValue = (1 / len(features) * sum([ (feat - meanValue) ** 2 for feat in features])) ** 0.5
    stdDevValue = stdev(features)
    return meanValue, stdDevValue


def normalisation(features, meanValue, stdDevValue):
    normalisedFeatures = [(feat - meanValue) / stdDevValue for feat in features]
    return normalisedFeatures
