import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from numpy import genfromtxt



def plot(data, results, title, figNumber = 0, save = False, vec1 = None, vec2 = None):
    #plot data
    plt.figure(figNumber)
    c0 = plt.scatter(data[results==-1,0], data[results==-1,1], s = 20, color = 'r', marker = 'x')
    c1 = plt.scatter(data[results==1,0], data[results==1,1], s = 20, color = 'b', marker = 'o')
    
    #make legend
    plt.legend((c0, c1), ('Class -1', 'Class +1'), loc='upper right', scatterpoints = 1, fontsize = 11)
    
    #label and title plot
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(title)
    
    l = np.linspace(-9,9)
    if vec1 != None:
        aa, bb = -vec1[1]/vec1[2], -vec1[0]/vec1[2]
        plt.plot(l, aa*l+bb, 'g-', lw=1)
    if vec2 != None:
        aa, bb = -vec2[1]/vec2[2], -vec2[0]/vec2[2]
        plt.plot(l, aa*l+bb, 'r-', lw=1)
    
    if save:
        plt.show
        plt.savefig(title + '.pdf', bbox_inches = 'tight')

def classification_error(data, results, vec):
    N = len(data)
    n_mispts = 0
    #for x, s in data, results:
    for i in range(0, N):
        x = np.array([1, data[i][0], data[i][1]])
        s = results[i]
        if int(np.sign(vec.T.dot(x))) != s:
            n_mispts += 1
    error = n_mispts / float(N)
    return error


def choose_miscl_point(data, results, vec):
    N = len(data)
    # Choose a random point among the misclassified
    mispts = []
    for i in range(0, N):
        x = np.array([1, data[i][0], data[i][1]])
        s = results[i]
        if int(np.sign(vec.T.dot(x))) != s:
            mispts.append((x, s))
    return mispts[random.randrange(0,len(mispts))]


def pocket(data, results, save=False, w = np.zeros(3), cutoff = 50):
    if(np.array_equal(w, np.zeros(3))):
        title = "pocket Iteration"
        linearBased = 1
    else:
        title = "Linear-Pocket Iteration"
        linearBased = -1
    N = len(data)
    it = 0
    bestIt = 0
    currentErr = classification_error(data, results, w)
    bestErr = currentErr
    bestW = w.copy()
    # Iterate until all points are correctly classified
    while bestErr != 0  and  it - bestIt < cutoff:
        it += 1
        # Pick random misclassified point
        x, s = choose_miscl_point(data, results, w)
        # Update weights
        w += s*x
        
        currentErr = classification_error(data, results, w)
        if(currentErr < bestErr):
            bestErr = currentErr
            bestW = w.copy()
            bestIt = it
        
        if save:
            plot(data = data, results = results, title = title + str(it), figNumber = linearBased*it, vec1=w, vec2 = bestW, save = False)
            
    print ("This took " + str(it) + " iterations to converge.")
    return bestW, it
    
    
def linear_regression(data, results, save = False):
    #Create X and Y from the data
    N = len(data)
    xTemp = []
    yTemp = []
    w = np.zeros(3)
    for i in range(0, N):
        x = np.array([1, data[i][0], data[i][1]])
        s = results[i]
        xTemp.append(x)
        yTemp.append(s)
    
    X = np.matrix(xTemp)
    Y = np.matrix(yTemp)

    #Derive intermediary matrices
    Xcross = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
    w += (np.dot(Xcross, Y.T).T).A1
    
    #Save and plot w
    if(save):
        plot(data = data, results = results, figNumber = 0, vec1=w, title = "Linear")
    return w




#cutoff for pocket algorithms
cutoff = 100
#target number to identify
target = 1

#Read data
dataset = genfromtxt('features.csv', delimiter = ' ')
y = dataset[:, 0]
X = dataset[:, 1:]
#Breaks data into not the target and the target
y[y <> target] = -1
y[y == target] = 1



wLinear = linear_regression(X, y, save = False)
print "Linear weights:" + str(wLinear)

print

wLP, it = pocket(X, y, save = False, w = wLinear, cutoff = cutoff)
print "Linear-Pocket weights:" + str(wLP)
itTotLP = it

print

wPocket, it = pocket(X, y, save = False, cutoff = cutoff)
print "Pocket weights:" + str(wPocket)
itTotPocket = it

print


#GETTING EIN:

errTotLinear = classification_error(data = X, results = y, vec = wLinear)
print "Linear Ein: " + str(errTotLinear)

errTotLP = classification_error(data = X, results = y, vec = wLP)
print "Linear-Pocket Ein:" + str(errTotLP)

errTotPocket = classification_error(data = X, results = y, vec = wPocket)
print "Pocket Ein: " + str(errTotPocket)


print "Pocket Iterations: " + str(itTotPocket)
print "Linear-Pocket Iterations: " + str(itTotLP)