from context import solvers
import solvers.activeSetSVM as ASVM
from context import utils
import utils.generateProblem as GP
import utils.kernels as KN
import matplotlib.pyplot as plt
import time
import numpy as np
import scipy.linalg as la

N = 500
Dim = 2
X,y,Xt,yt = GP.largerRegressionProblem(N,Dim,5)


# kernels = ['linear','rbf','tanh','poly']
# params = ['',10,[-1,-1],[3,1,1]]

kernels = ['rbf']
params = [10]
C = 1000
epsilon = 0.1

for kerIter in range(len(kernels)):

    ################ compute kernels
    if (kernels[kerIter]=='linear'):
        K = KN.linearKernel(X)
        Kt = KN.linearKernel(X,Xt=Xt)
    if (kernels[kerIter]=='rbf'):
        K = KN.rbfKernel(X,params[kerIter])
        Kt = KN.rbfKernel(X,params[kerIter],Xt=Xt)
    if (kernels[kerIter]=='tanh'):
        K = KN.tanhKernel(X,params[kerIter])
        Kt = KN.tanhKernel(X,params[kerIter],Xt=Xt)
    if (kernels[kerIter]=='poly'):
        K = KN.polyKernel(X,params[kerIter])
        Kt = KN.polyKernel(X,params[kerIter],Xt=Xt)
    ################ train SVR
    tic = time.time()
    if (kernels[kerIter]=='tanh'):
        [D,U,Ktilde] = ASVM.preKSVM(K) # to be checked for regression
        alpha,b = ASVM.solveSVR(Ktilde,y,C,epsilon,verbose=1)
        temp = alpha.copy()
        alpha = ASVM.postKSVM(alpha,y.T,D,U)
    else:
        alpha,b = ASVM.solveSVR(K,y,C,epsilon,verbose=1)
    toc = time.time()-tic
    print('Training time: ' ,toc, 'sec')
    if (Dim==2):
        if (kernels[kerIter]=='tanh'):
            GP.plot2dResultSVR(X,y,alpha,b,kernels[kerIter],C,epsilon,param = params[kerIter],num=kerIter,origAlpha=temp)
        else:
            GP.plot2dResultSVR(X,y,alpha,b,kernels[kerIter],C,epsilon,param = params[kerIter],num=kerIter)

    perf1 = ASVM.evaluateSVR(yt,alpha=alpha,b=b,Kt=Kt)
    print("Perf 1 : ", 100*perf1, "(MSE)")


################ show All
if (Dim==2):
    plt.show()
