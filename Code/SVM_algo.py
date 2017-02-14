import cvxopt
import numpy as np
import pandas as pd

def qp(P, q, A, b, C=100, l=1e-8, verbose=True):
    
    # Gram matrix
    n = P.shape[0]
    P = cvxopt.matrix(P)
    A = cvxopt.matrix(A)
    q = cvxopt.matrix(-q)
    b = cvxopt.matrix(b)

    G = cvxopt.matrix(np.concatenate([np.diag(np.ones(n) * -1), np.diag(np.ones(n))], axis=0))
    h = cvxopt.matrix(np.concatenate([np.zeros(n), C * np.ones(n)]))

    # Solve QP problem
    cvxopt.solvers.options['show_progress'] = verbose
    solution = cvxopt.solvers.qp(P,q,G,h,A,b)
 
    # Lagrange multipliers
    alpha = np.ravel(solution['x'])
    return alpha


def svm_solver(Kernel, X, y, C=100):
    
    alpha = qp(P = Kernel(X), q = y, A = np.ones(X.shape[0]), b = 0, C = C, l=1e-8, verbose=False)
    
    idx_support = np.where(np.abs(mu) > 1e-5)[0]
    
    alpha_support = alpha[idx_support]
    
    return alpha_support, idx_support
    

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

    def __init__(self,lbd,ker,c):
        self.kernel = ker
        self.C = c
        self.lbda = lbd

    def fit(self, X_tr, y_tr):
        K=self.kernel(X)
        a_support, idx_support = svm_solver(K, X_tr, y_tr)
        b_model = compute_b(K, y_tr, mu_support, idx_support)
        self.alpha = a_support
        self.idx = idx_support 
        self.b = b_model
        self.X_support = X_tr[idx_support]

    def predict(self, X_te):
        G = self.kernel(X_te, self.X_support)
        decision = G.dot(self.alpha * y[self.idx]) + self.b
        # Calcul du label prédit
        y_pred = np.sign(decision)
        return y_pred



