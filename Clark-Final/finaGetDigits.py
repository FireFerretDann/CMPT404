import numpy as np
from numpy import genfromtxt

def getDataSet():
    # read digits data & split it into X and y for training and testing
    dataset = genfromtxt('features.csv', delimeter='_')
    y = dtaset[:, 0]
    X = dataset[:, 1:]
    
    dataset = genfromtxt('features-t.csv', delimeter='_')
    y_te = dtaset[:, 0]
    X_te = dataset[:, 1:]
    return X, y, X_te, y_te