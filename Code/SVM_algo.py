import cvxopt
import numpy as np
import pandas as pd
import kernel_functions
from sklearn.metrics import accuracy_score

def qp(P, q, A, b, C=100, l=1e-8, verbose=True):
    
    # Gram matrix
    n = P.shape[0]
    P = cvxopt.matrix(P)
    #A = cvxopt.matrix(A, (1, n))
    q = (q).astype(float)
    #b = cvxopt.matrix(b)
    A=None
    b=None

    G = cvxopt.matrix(np.concatenate([np.diag(np.ones(n) * q), np.diag(np.ones(n))*-q], axis=0))
    h = cvxopt.matrix(np.concatenate([np.zeros(n), C * np.ones(n)]))
    q = cvxopt.matrix(q)


    # Solve QP problem
    cvxopt.solvers.options['show_progress'] = verbose
    solution = cvxopt.solvers.qp(P,q,G,h,A,b)
 
    # Lagrange multipliers
    alpha = np.ravel(solution['x'])

    return alpha


def svm_solver(K, X, y, C=1.):

    alpha = qp(P = K, q = -y, A = np.ones(X.shape[0]), b = 0., C = C, l=1e-8, verbose=False)
    
    idx_support = np.where(np.abs(alpha) > 1e-5)[0]
    
    alpha_support = alpha[idx_support]

    return alpha_support, idx_support
    
#alpha, idx = svm_solver(kernel_functions.kernel_test(X_train,X_train),X_train,y_train)

def compute_b(Kernel, y, alpha_support, idx_support):
    # DONE
    y_support = y[idx_support]
    Kernel_support = Kernel[idx_support][:, idx_support]
    b = np.zeros(y_support.shape[0])
    b = y_support - np.dot(alpha_support, np.dot(np.diag(y_support), Kernel_support))
    b = b.tolist()
    b = np.mean(b)
    return b


class SVM():

    def __init__(self,lbd,Ktrain,ker,c, classe1):
        self.Ktrain=Ktrain
        self.kernel = ker
        self.C = c
        self.lbda = lbd
        self.classe = classe1

    def fit(self, X_tr, y_tr2):
        y_tr=y_tr2.copy()
        for i in range(len(y_tr)):
            if y_tr[i]==self.classe:
                y_tr[i]=1
            else:
                y_tr[i]=-1
        #print(y_tr[y_tr==-1].shape)
        #print(y_tr[y_tr==1].shape)

        #self.K = self.kernel(X_tr, X_tr)
    
        self.a_support, self.idx_support = svm_solver(self.Ktrain, X_tr, y_tr)
        self.b_model = compute_b(self.Ktrain, y_tr, self.a_support, self.idx_support)
        self.X_support = X_tr[self.idx_support]
        self.y_support = y_tr[self.idx_support]

    def predict(self, X_te):
        G = self.kernel(X_te, self.X_support)
        decision = G.dot(self.a_support) #+ self.b_model
        self.y_pred = decision# Calcul du label predit
        return self.y_pred
    
    def score(self, y_te2):
        y_te=y_te2.copy()
        for i in range(len(y_te)):
            y_te[y_te!=self.classe]=-1
            y_te[y_te==self.classe]=1

        return accuracy_score(np.sign(self.y_pred), y_te)
        
    
#################################################################

    




