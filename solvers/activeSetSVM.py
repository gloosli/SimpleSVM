

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import ipdb

def lunopiv(A,ptol):

    m,n = np.shape(A)
    for i in np.arange(0,n):
        pivot = A[i,i]
        if abs(pivot) < ptol:
           print('zero pivot encountered:',pivot)
           break
        for k in np.arange(i+1,n):
            A[k,i] = A[k,i]/pivot
            A[k,i+1:n] = A[k,i+1:n] - A[k,i]*A[i,i+1:n]
    L = np.eye(n)+np.tril(A,-1)
    D = np.diag(A)
    return L,D


def preKSVM(K):
    [D,U] = np.linalg.eigh(np.array(K))
    Ktilde = np.dot(np.dot(U,np.abs(np.diag(D))),U.T)
    return D,U,Ktilde

def postKSVM(alpha,y,D,U):
    alphat = np.dot(np.dot(U,np.dot(np.diag(np.sign(D)),U.T)),(alpha*y.reshape((-1,1))))
    return alphat*(y.reshape((-1,1)))


def preKSVM2(K):
    L,D = lunopiv(np.array(K),0.000001)
    Ktilde = L.dot(np.diag(np.abs(D)).dot(L.T))
    return L,D,Ktilde

def postKSVM2(alpha,y,L,D):
    alphat = np.dot(np.linalg.solve(L.T,L.dot(np.diag(np.sign(D))).T),(alpha*y.reshape((-1,1))))
    return alphat*(y.reshape((-1,1)))

def predict(alpha,b,y,Kt):
    return np.dot(y*alpha.T,Kt)+b

def predictSVR(alpha,b,Kt):
    return np.dot(alpha.T,Kt)+b

def evaluate(yt, ypred= None, alpha=None,b=None,y=None,Kt=None ):
    if (ypred is None):
        ypred = predict(alpha,b,y,Kt)
    return np.count_nonzero(np.where(np.sign(ypred)==np.sign(yt)))/np.size(yt)


def evaluateSVR(yt, ypred= None, alpha=None,b=None,Kt=None ):
    if (ypred is None):
        ypred = predictSVR(alpha,b,Kt)
    return ((ypred-yt)**2).mean(axis=None)

def solveSVR(K,y,C,epsilon,x0 = None, verbose = 1,lb=None,ub=None,ce=None):
    eqcons = np.concatenate((np.ones(y.size),-np.ones(y.size)))
    if (ce is None):
        ce = np.concatenate((y-epsilon,-y-epsilon),axis=0)
    G = np.tile(K,(2,2))
    beta,b = solve(G,eqcons,C,x0 = x0, verbose = verbose,lb=lb,ub=ub,ce=ce,type='regression')
    alpha = beta[0:y.size]-beta[y.size:]
    return alpha,b

def initialize(I,smin,ub,lb,G,y,ce,be):
    n = I.size
    xnew = -np.ones((smin))
    w = [0,1]
    ness=0
    while ((np.where(xnew<lb[w])[0].size>0 or (np.where(xnew>ub[w])[0].size>0)) and ness<100):
        w = np.random.choice(n, smin, replace=False)
        Z = np.linalg.solve(G[np.ix_(w,w)],y[w].T)
        W = np.linalg.solve(G[np.ix_(w,w)],ce[w])
        b = (be+np.dot(W.T,y[w]))/np.dot(y[w],Z)
        xnew = (-b*Z+W.T).T
        ness += 1
    if ness==100:
        point1 = 0
        I[0]+=1.
    else:
        I[w]+=1
    print(ness)
    return I

def solve(K,y,C,x0 = None, verbose = 1,lb=None,ub=None,ce=None,type='binary'):
    n = np.shape(y)[0]
    prec = 0.0001
    epochs = 10

    G = K*np.outer(y,y)+(prec*prec)*np.eye(n)

    be = 0
    if (lb is None):
        lb = np.zeros((n,1))
    else:
        lb = lb.reshape((-1,1))
    if (ub is None):
        ub = C*np.ones((n,1))
    else:
        ub = ub.reshape((-1,1))

    if (ce is None):
        ce = np.ones((n,1))
    else:
        ce = ce.reshape((-1,1))

    margin=ce.copy()

    probSelection = np.ones(n)/n
    iter_max = n*epochs
    MAX_SELECT = np.max((np.min((200,n-10)),2))

    I = -np.ones(n)
    x = lb.copy()
    if x0 is not None:
        for i in range(n):
            if (x0[i]>lb[i]):
                I[i]+=1.
                ce -= lb[i]*G[:,i]
                be += lb[i]*y[:,i]
                x[i] = x0[i]
            if (x[i]>=ub[i]):
                I[i]+=1.
                ce -= ub[i]*G[:,i]
                be += ub[i]*y[:,i]
                x[i] = ub[i]
    else:
        sMin = 2
        I = initialize(I,sMin,ub,lb,G,y,ce,be)

    optimal = False
    iter = 0
    if (verbose>=1):
        print("itinial iteration ",iter, ". nb Iw : ", (np.where(I==0)[0].size),". nb Iu : ", (np.where(I==1)[0].size),". nb Il : ", (np.where(I==-1)[0].size))

    count_ss=0
    while (not(optimal) or (np.any(x<lb) or np.any(x>ub))) and iter<iter_max and sMin>0:
        iter+=1

        if (verbose>1 and iter%100==0):
            print("iteration ",iter, ". nb Iw : ", (np.where(I==0)[0].size),". nb Iu : ", (np.where(I==1)[0].size),". nb Il : ", (np.where(I==-1)[0].size))

        xold = x.copy()
        w = np.where(I==0)[0]
        Z = np.linalg.solve(G[np.ix_(w,w)],y[w].T)
        W = np.linalg.solve(G[np.ix_(w,w)],ce[w])

        b = (be+np.dot(Z.T,ce[w]))/np.dot(y[w],Z)
        x[w] = (-b*Z+W.T).T
        if (np.any(x<lb) or np.any(x>ub)) and np.size(w)>sMin:
            di = x[w]-xold[w]
            indl = np.where(x[w]<lb[w])[0]
            stepl = np.min((lb[w]-xold[w])/di)
            indu = np.where(x[w]>ub[w])[0]
            stepu = np.min((xold[w]-ub[w])/di)
            if indl.size==0:
                stepl = stepu+1
            if indu.size==0:
                stepu = stepl+1
            if stepl<stepu:
                cand = np.argmin((lb[w[indl]]-xold[w[indl]])/di[indl])
                I[w[indl[cand]]]-=1.
                ce -= np.reshape(lb[w[indl[cand]]]*G[:,w[indl[cand]]],(n,1))
                be += lb[w[indl[cand]]]*y[w[indl[cand]]]
                x[w] += stepl*di
                x[w[indl[cand]]] = lb[w[indl[cand]]]
                probSelection[w[indl[cand]]] = 1/n
                probSelection /= np.sum(probSelection)
            else:
                cand = np.argmin((xold[w[indu]]-ub[w[indu]])/di[indu])
                I[w[indu[cand]]]+=1.
                ce -= np.reshape(ub[w[indu[cand]]]*G[:,w[indu[cand]]],(n,1))
                be += ub[w[indu[cand]]]*y[w[indu[cand]]]
                x[w] += stepu*di
                x[w[indu[cand]]] = ub[w[indu[cand]]]
                probSelection[w[indu[cand]]] = 1/n
                probSelection /= np.sum(probSelection)


        else:
            w = np.where(I>=0)[0]
            if (np.size(w)==np.size(y)):
                optimal = True
            else:
                sizeSelection = min(MAX_SELECT,np.size(np.where(probSelection>0)[0]))
                selection =  np.sort(np.random.choice(n, sizeSelection, replace=False, p=probSelection))

                nu = ((np.dot(G[np.ix_(selection,w)],x[w]).T+b*y.T[selection]-margin[selection].T)*-I[selection])[0]
                cand = np.argmin(nu)
                if (nu[cand]<0):
                    if I[selection[cand]]==-1:
                        I[selection[cand]]+=1.
                        ce += np.reshape(lb[selection[cand]]*G[:,selection[cand]],(n,1))
                        be -= lb[selection[cand]]*y[selection[cand]]
                        x[selection[cand]] = lb[selection[cand]]
                    else:
                        I[selection[cand]]-=1.
                        ce += np.reshape(ub[selection[cand]]*G[:,selection[cand]],(n,1))
                        be -= ub[selection[cand]]*y[selection[cand]]
                        x[selection[cand]] = ub[selection[cand]]
                    probSelection[selection[cand]] = 0

                    probSelection[selection[np.where(nu>0)]] /= 1.5
                    probSelection[selection[np.where(nu<0)]] *= 1.5

                else:
                    selection =  np.where(I!=0)[0]
                    if (selection.size>0):
                        nu = ((np.dot(G[np.ix_(selection,w)],x[w]).T+b*y.T[selection]-margin[selection].T)*-I[selection])[0]
                        cand = np.argmin(nu)
                        if (nu[cand]>=0):
                            optimal = True
                        else:
                            if I[selection[cand]]==-1:
                                I[selection[cand]]+=1.
                                ce += np.reshape(lb[selection[cand]]*G[:,selection[cand]],(n,1))
                                be -= lb[selection[cand]]*y[selection[cand]]
                                x[selection[cand]] = lb[selection[cand]]
                            else:
                                I[selection[cand]]-=1.
                                ce += np.reshape(ub[selection[cand]]*G[:,selection[cand]],(n,1))
                                be -= ub[selection[cand]]*y[selection[cand]]
                                x[selection[cand]] = ub[selection[cand]]

                            probSelection[selection] = - nu + np.max(nu)
                            probSelection[selection[cand]] = 0
                    else:
                        optimal = True

                probSelection /= np.sum(probSelection)

    if iter==iter_max:
        print("not converged")
    else:
        print("converged ok")


    if (verbose>=0):
        print("final iteration ",iter, ". nb Iw : ", (np.where(I==0)[0].size),". nb Iu : ", (np.where(I==1)[0].size),". nb Il : ", (np.where(I==-1)[0].size))
    return x,b
