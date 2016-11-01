import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsRegressor

def genDataSet(N):
    x = np.random.normal(0, 1, N)
    yTrue = (np.cos(x) + 2)/(np.cos(x * 1.4) + 2)
    noise = np.random.normal(0, 0.2, N)
    y = yTrue + noise
    return x, y, yTrue


N = 1000
FOLDS = 10

x, y, yTrue = genDataSet(N)
plt.plot(x, y, '.')
plt.plot(x, yTrue, 'rx')
plt.show()


errors = []

# For each odd number of neighbors
for numNeighbors in range(1, ((FOLDS-1)*N)//FOLDS + 1, 2):
    totErr = 0
    # Split
    #print "numNeighbors: " + str(numNeighbors)
    kf = KFold(N, n_folds=FOLDS)
    for train, test in kf:
        # Train
        knr = KNeighborsRegressor(n_neighbors = numNeighbors)
        knr.fit(x[train].reshape(len(x[train]), 1), y[train])
        
        # Test
        err = sum(abs(y[test] - knr.predict(x[test].reshape(len(x[test]), 1))))  /  len(test)
        totErr += err
        #print "Error: " + str(err)
        
    avgErr = totErr/FOLDS
    #print avgErr
    errors.append((avgErr, numNeighbors))
#print errors
#print '\n'

errors.sort()
print errors[:3]