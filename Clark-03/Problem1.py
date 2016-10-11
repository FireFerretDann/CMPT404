import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs



def plot(data, results, title, figNumber = 0, save = False, vec1 = None, vec2 = None):
    #plot data
    plt.figure(figNumber)
    plt.xlim(-9, 9)
    plt.ylim(-9, 9)
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


#Variables to track cumulative error and iterations
errTotLinear = 0
errTotPocket = 0
errTotLP = 0
itTotPocket = 0
itTotLP = 0

#number of trials used for each 
reps = 200
#cutoff for pocket algorithms
cutoff = 100
for i in range(0, reps):
    print '--------------------\n\nTrial number ' + str(i)
    #random centers
    ctrs = 3 * np.random.normal(0, 1, (2, 2))
    
    #makes data blobs
    X, y = make_blobs(centers = ctrs, random_state = 2*i)
    
    #make any 0 points be -1
    y[y==0] = -1
    
    
    wLinear = linear_regression(X, y, save = False)
    print "Linear weights:"
    print wLinear
    
    print
    
    wPocket, it = pocket(X, y, save = False, cutoff = cutoff)
    print "Pocket weights:"
    print wPocket
    itTotPocket += it
    
    print
    
    wLP, it = pocket(X, y, save = False, w = wLinear, cutoff = cutoff)
    print "Linear-Pocket weights:"
    print wLP
    itTotLP += it
    
    print
    
    
    #GETTING EOUT:
    
    #makes different data blobs with the same centers for testing puroposes
    XTest, yTest = make_blobs(centers = ctrs, random_state = 2*i + 1)
    #make any 0 points be -1
    yTest[yTest==0] = -1
    
    
    
    tempErr = classification_error(data = XTest, results = yTest, vec = wLinear)
    errTotLinear += tempErr
    print "Linear Eout: " + str(tempErr)
    
    tempErr = classification_error(data = XTest, results = yTest, vec = wPocket)
    errTotPocket += tempErr
    print "Pocket Eout: " + str(tempErr)
    
    tempErr = classification_error(data = XTest, results = yTest, vec = wLP)
    errTotLP += tempErr
    print "Linear-Pocket Eout:" + str(tempErr)
    
    #plot(X, y, "Training data " + str(i), save = False, figNumber = 2*i)
    #plot(XTest, yTest, "Testing Data " + str(i), save = False, figNumber = 2*i + 1)


print "\n\n"

print "Linear Average Error: " + str(errTotLinear/float(reps))
print "Pocket Average Error: " + str(errTotPocket/float(reps))
print "Linear-Pocket Average Error: " + str(errTotLP/float(reps))

print "\n"

print "Pocket Average Iterations: " + str(itTotPocket/float(reps))
print "Linear-Pocket Average Iterations: " + str(itTotLP/float(reps))