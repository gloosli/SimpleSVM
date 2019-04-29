from context import solvers
import solvers.activeSetSVM as ASVM
from context import utils
import utils.generateProblem as GP
import utils.kernels as KN
import matplotlib.pyplot as plt
import time
import numpy as np
import scipy.linalg as la


N = 300
Dim = 2
X,y,Xt,yt = GP.largerBinaryProblem(np.int(N/2),Dim,5)
#X,y,Xt,yt = GP.smallBinaryProblem(N,prop=5)

kernels = ['linear','rbf','tanh']
params = ['',0.2,[-0.1,0.1]]

C = 1

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
    ################ train binary CSVM
    tic = time.time()
    if (kernels[kerIter]=='tanh'):
        [D,U,Ktilde] = ASVM.preKSVM(K)
        alpha,b = ASVM.solve(Ktilde,y,C,verbose=1)
        temp = alpha.copy()
        alpha = ASVM.postKSVM(alpha,y.T,D,U)
    else:
        alpha,b = ASVM.solve(K,y,C,verbose=1)
    toc = time.time()-tic
    print('Training time: ' ,toc, 'sec')
    if (Dim==2):
        if (kernels[kerIter]=='tanh'):
            GP.plot2dResult(X,y,alpha,b,kernels[kerIter],C,param = params[kerIter],num=kerIter,origAlpha=temp)
        else:
            GP.plot2dResult(X,y,alpha,b,kernels[kerIter],C,param = params[kerIter],num=kerIter)

    perf1 = ASVM.evaluate(yt,alpha=alpha,b=b,y=y,Kt=Kt)
    print("Perf 1 : ", 100*perf1, "%")


################ show All
if (Dim==2):
    plt.show()
