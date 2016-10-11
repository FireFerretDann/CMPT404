import numpy as np
import random
import os, subprocess
import matplotlib.pyplot as plt
import makeSemiCircles
 
class Perceptron:
    def __init__(self, N):
        self.X = self.generate_points(N)
 
    def generate_points(self, N):
        X = []
        points, sign = makeSemiCircles.make_semi_circles(n_samples = N, thk=5, rad=10, sep=5, plot=False)
        for i in range(N):
            x = np.array([1, points[i,0], points[i,1]])
            s = sign[i]
            X.append((x, s))
        return X
 
    def plot(self, mispts=None, vec=None, save=False):
        fig = plt.figure(figsize=(5,5))
        plt.xlim(-1.5,2.5)
        plt.ylim(-2,2)
        l = np.linspace(-1.5,2.5)
        for x,s in self.X:
            if(s == -1):
                plt.plot(x[1], x[2], "r.")
            else:
                plt.plot(x[1], x[2], "b.")
        if mispts:
            for x,s in mispts:
                plt.plot(x[1], x[2], cols[s]+'.')
        if vec != None:
            aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
            plt.plot(l, aa*l+bb, 'g-', lw=2)
        if save:
            if not mispts:
                plt.title('N = %s' % (str(len(self.X))))
            else:
                plt.title('N = %s with %s test points' \
                          % (str(len(self.X)),str(len(mispts))))
            plt.savefig('ap_N%s' % (str(len(self.X))), \
                        dpi=200, bbox_inches='tight')
 
    def classification_error(self, vec, pts=None):
        # Error defined as fraction of misclassified points
        if not pts:
            pts = self.X
        M = len(pts)
        n_mispts = 0
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                n_mispts += 1
        error = n_mispts / float(M)
        return error

    def choose_miscl_point(self, vec):
        # Choose a random point among the misclassified
        pts = self.X
        mispts = []
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                mispts.append((x, s))
        return mispts[random.randrange(0,len(mispts))]
 
    def pla(self, save=False):
        # Initialize the weigths to zeros
        w = np.zeros(3)
        X, N = self.X, len(self.X)
        it = 0
        # Iterate until all points are correctly classified
        while self.classification_error(w) != 0:
            it += 1
            # Pick random misclassified point
            x, s = self.choose_miscl_point(w)
            # Update weights
            w += s*x
            if save:
                self.plot(vec=w)
                plt.title('N = %s, Iteration %s\n' \
                          % (str(N),str(it)))
                plt.savefig('ap_N%s_it%s' % (str(N),str(it)), \
                            dpi=200, bbox_inches='tight')
        self.w = w
        print ("This took " + str(it) + " iterations to converge.")
 
    def check_error(self, M, vec):
        check_pts = self.generate_points(M)
        return self.classification_error(vec, pts=check_pts)
        
    def linear_regression(self):
        #Create X and Y from the data
        xTemp = []
        yTemp = []
        w = np.zeros(3)
        for x, s in self.X:
            xTemp.append(x)
            yTemp.append(s)
        
        X = np.matrix(xTemp)
        Y = np.matrix(yTemp)

        #Derive intermediary matrices
        Xcross = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        w += (np.dot(Xcross, Y.T).T).A1
        
        #Save and plot w
        self.w = w
        self.plot(vec = w)
        
        


p = Perceptron(2000)
p.pla(save = True)

p.linear_regression()