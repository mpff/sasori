import time

import numpy

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_random_state
from sklearn.metrics import make_scorer



class MatrixFactorization(BaseEstimator):
    '''Perform Biased Sparse Matrix Factorization with Alternating Least
    Squares

    The objective function is minimized with an alternating minimization of U
    and V. Each minimization is done by solving a least squares problem.

    Parameters
    ----------
    n_features : integer, default: 40
        Number of features.

    reg : double, default: 1.
        Constant that multiplies the regularization terms. Set it to zero to
        have no regularization.

    max_iter : integer, default: 200
        Maximum number of iterations before timing out.

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance to define the state of the
        random permutations generator.

    verbose : bool, default=False
        Whether to be verbose.


    Attributes
    ----------
    error_ : number
        Reconstruction error.

    n_iter_ : int
        Actual number of iterations.
    '''

    
    def __init__(self, n_features=40, reg=1., tol=1e-4, max_iter=200,
                 random_state=0, verbose=False):
        self.n_features = n_features
        self.reg = reg
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        
    def fit(self, X, y=None):
        '''Learn a MF model for the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_users, n_items)
            Data matrix to be decomposed

        y : Ignored

        Returns
        ------
        self
        '''
        tstart = time.time()

        X = check_array(X, accept_sparse='csc', dtype=float)

        self.random_state_ = check_random_state(self.random_state)
        
        U = self.random_state_.randn(X.shape[0],self.n_features)
        self.V_ = self.random_state_.randn(X.shape[1],self.n_features)
        bu = numpy.zeros(X.shape[0])
        self.bv_ = numpy.zeros(X.shape[1])
        self.mu_ = X.data.mean()
        
        self.error_ = [self.loss_internal(X,U,bu)]

        if self.verbose:
            print(f"Init          :  Training Error = {self.error_[-1]:3.4f}  ({time.time()-tstart:.2f}s)")
        
        for n_iter in range(1,self.max_iter+1):
            tstart = time.time()
            
            X = X.tocsr()
            # Update user matrix U.
            for i in range(U.shape[0]):
                jvec = X.getrow(i).nonzero()[1]
                matrix_ = self.V_[jvec].T.dot(self.V_[jvec]) + numpy.eye(self.n_features) * self.reg
                vector_ = (X[i,jvec] - (bu[i] + self.bv_[jvec] + self.mu_)).dot(self.V_[jvec]).T
                bu[i] = (X[i,jvec] - self.V_[jvec].dot(U[i]) - self.bv_[jvec] - self.mu_).sum()
                bu[i] = bu[i] / ( len(jvec) + self.reg)
                U[i] = numpy.squeeze(numpy.linalg.solve(matrix_,vector_))
            
            # Update item matrix V.
            X = X.tocsc()
            for j in range(self.V_.shape[0]):
                ivec = X.getcol(j).nonzero()[0]
                matrix_ = U[ivec].T.dot(U[ivec]) + numpy.eye(self.n_features) * self.reg
                vector_ = (X[ivec,j].T - bu[ivec] - self.bv_[j] - self.mu_).dot(U[ivec]).T
                self.bv_[j] = (X[ivec,j].T - (U[ivec].dot(self.V_[j]) + bu[ivec] + self.mu_)).sum()
                self.bv_[j] = self.bv_[j] / ( len(ivec) + self.reg)
                self.V_[j] = numpy.squeeze(numpy.linalg.solve(matrix_,vector_))

            # Calculate Training Error.
            self.error_.append(self.loss_internal(X,U,bu))
            if self.verbose:
                print(f"Iteration {n_iter:3d} :  Training Error = {self.error_[-1]:3.4f}  ({time.time()-tstart:.2f}s)")
            if (self.error_[-2] - self.error_[-1]) / self.error_[0] < self.tol:
                if self.verbose:
                    print("Converged at iteration", n_iter)
                self.n_iter_ = n_iter
                break

        return self
            
        
    def predict(self, X):
        X = check_array(X, accept_sparse='csr', dtype=float)
        Xhat = numpy.zeros(X.shape)
        V = numpy.c_[ numpy.ones(self.V_.shape[0]), self.V_ ]

        # Predict (bias, features) user matrix.
        X.data = X.data - self.mu_
        for i in range(X.shape[0]):
            jvec = X.getrow(i).nonzero()[1]
            vector = numpy.dot(X[i,jvec] - self.bv_[jvec], V[jvec]).T
            matrix = V[jvec].T.dot(V[jvec]) + numpy.eye(self.n_features+1) * self.reg
            xhat = numpy.linalg.solve(matrix, vector)
            Xhat[i,:] = xhat[1:].T.dot(V[:,1:].T) + xhat[0] + self.bv_ + self.mu_

        return Xhat

           
            
    def loss(self, X):
        X = check_array(X, accept_sparse='csr', dtype=float)
        N = 0.
        E = 0.
        for i in range(X.shape[0]):
            jvec = X.getrow(i).nonzero()[1]
            xhat = self.predict(X.getrow(i))
            resd = X[i,jvec] - xhat[:,jvec]
            E += resd.dot(resd.T)
            N += len(jvec)
        return E[0,0] / N


    def score(self, X):
        score = -1.0 * self.loss(X)
        return score


    def loss_internal(self, X, U, bu):
        N = 0.
        E = 0.
        for j in range(self.V_.shape[0]):
            ivec = X.getcol(j).nonzero()[0]
            xhat = U[ivec].dot(self.V_[j].T) + bu[ivec] + self.bv_[j] + self.mu_
            resd = X[ivec,j].T - xhat
            E += resd.dot(resd.T) 
            N += len(ivec)
        return E[0,0] / N

