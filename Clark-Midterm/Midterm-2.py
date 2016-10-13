#   This program uses iteration to find the minimum required
# number of samples for a given dvc, confidence interval,
# and generalization error.
import numpy as np
import matplotlib.pyplot as plt


def numberOfSamples(dvc, confidenceDec, error, guess = 1, shouldPlot = False):
    # Prepare data
    N = guess
    delta = 1 - confidenceDec
    if shouldPlot:
        partialNs = [N]
    
    # Perform initial calculation
    newN = calculate(dvc, delta, error, N)
    
    #Iteratively improves guess until it does not noticably change
    while newN <> N:
        if shouldPlot:
            partialNs.append(newN)
        N = newN
        newN = calculate(dvc, delta, error, N)
        #print N
    
    if shouldPlot:
        plt.figure()
        plt.title('Aproximations')
        plt.xlabel('Iteration')
        plt.ylabel('N')
        plt.plot(partialNs, color='blue', lw=1)
        plt.show
        plt.savefig('N over Iterations.pdf', bbox_inches = 'tight')
        
    return np.ceil(newN)
    
def calculate(dvc, delta, err, N):
    return (8/err**2)*np.log(4*((2*N)**dvc + 1)/delta)



print numberOfSamples(dvc = 10, confidenceDec = .95, error = .05, shouldPlot = True)