import time

import numpy

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_random_state



class MatrixFactorization(BaseEstimator, RegressorMixin):
    '''Perform Biased Sparse Matrix Factorization with Alternating Least
    Squares

    The objective function is minimized with an alternating minimization of U
    and V. Each minimization is done by solving a least squares problem.

    Parameters
    ----------
    n_features : integer, default: 40
        Number of features.

    reg : double, default: 0.
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

    
    def __init__(self, n_features=40, reg=0., max_iter=200, random_state=0, verbose=False):
        self.n_features = n_features
        self.reg = reg
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

        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float)
        self.random_state_ = check_random_state(self.random_state)
        
        self.U_ = self.random_state_.randn(X.shape[0],self.n_features)
        self.V_ = self.random_state_.randn(X.shape[1],self.n_features)
        self.bu_ = numpy.zeros(X.shape[0])
        self.bv_ = numpy.zeros(X.shape[1])
        self.mu_ = X.data.mean()
        
        tstart = time.time()
        self.error_ = [self.loss(X)]
        print(f"Iteration  0 :  Training Error = {self.error_[-1]:3.4f}  Time = {time.time()-tstart:.2f}s")
        
        for t in range(1,max_iter+1):
            tstart = time.time()
            
            # Update user matrix U.
            X = X.tocsr()
            for i in range(self.U_.shape[0]):
                jvec = X.getrow(i).nonzero()[1]
                matrix_ = self.V_[jvec].T.dot(self.V_[jvec]) + numpy.eye(self.n_features) * self.reg
                vector_ = (X[i,jvec] - (self.bu_[i] + self.bv_[jvec] + self.mu_)).dot(self.V_[jvec]).T
                self.bu_[i] = (X[i,jvec] - self.V_[jvec].dot(self.U_[i]) - self.bv_[jvec] - self.mu_).sum()
                self.bu_[i] = self.bu_[i] / ( len(jvec) + self.reg)
                self.U_[i] = numpy.squeeze(numpy.linalg.solve(matrix_,vector_))
            
            # Update item matrix V.
            X = X.tocsc()
            for j in range(self.V_.shape[0]):
                ivec = X.getcol(j).nonzero()[0]
                matrix_ = self.U_[ivec].T.dot(self.U_[ivec]) + numpy.eye(self.n_features) * self.reg
                vector_ = (X[ivec,j].T - self.bu_[ivec] - self.bv_[j] - self.mu_).dot(self.U_[ivec]).T
                self.bv_[j] = (X[ivec,j].T - (self.U_[ivec].dot(self.V_[j]) + self.bu_[ivec] + self.mu_)).sum()
                self.bv_[j] = self.bv_[j] / ( len(ivec) + self.reg)
                self.V_[j] = numpy.squeeze(numpy.linalg.solve(matrix_,vector_))

            # Calculate Training Error.
            self.error_.append(self.loss(X))
            if self.verbose == True:
                print(f"Iteration {t:2d} :  Training Error = {self.error_[-1]:3.4f}  Time = {time.time()-tstart:.2f}s")

        return self
            
        
    def predict(self, r):
        rb = (r - self.mu_ - self.bv_)
        w = ~numpy.isnan(r)
        
        # Create (1,features) item vector.
        V = numpy.c_[ numpy.ones(self.V_.shape[0]), self.V_ ]
        
        # Predict (bias, features) user vector.
        vector = numpy.dot(V[w,:].T, rb[w])
        matrix = V[w,:].T.dot(V[w,:]) + self.reg * numpy.eye(self.n_features+1)
        xhat = numpy.linalg.solve(matrix, vector)
                
        # Predict scores.
        rhat = xhat[1:].dot(V[:,1:].T) + xhat[0] + self.bv_ + self.mu_
        
        return rhat
           
            
    def loss(self, X):
        N = 0.
        E = 0.
        for j in range(self.V_.shape[0]):
            ivec = X.getcol(j).nonzero()[0]
            xtru = X[ivec,j].todense().T
            xhat = self.U_[ivec].dot(self.V_[j].T) + self.bu_[ivec] + self.bv_[j] + self.mu_
            resd = xtru - xhat
            E += resd.dot(resd.T)
            N += len(ivec)
        return E[0,0] / N
