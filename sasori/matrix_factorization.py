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

    reg : double, default: 0
        Constant that multiplies the regularization terms. Set it to zero to
        have no regularization.

    tol : double, default: 1e-4
        Stopping criterion. Stop if the training error changes by less than
        [ TODO: Insert criterion here ].

    max_iter : integer, default: 200
        Maximum number of iterations before timing out.

    n_iter : integer, default: None
        Set this parameter if instead of using the tolerance stopping criteron
        you want to perform n_iter ALS iterations.

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance to define the state of the
        random permutations generator.

    verbose : bool, default=False
        Toggle verbose output.


    Attributes
    ----------
    mu_ : float
        Mean Rating in training data `X`.

    V_ : array, (n_items, n_features)
        Trained item features.

    reconstruction_error_ : float
        Difference between training data `X` and reconstructed data.

    item_ids_ : dict
        Matches item ids from the training data to inner ids.

    n_items_ : int
        Number of items, the model was fit for. 

    n_iter_ : int
        Actual number of iterations.

    '''

    
    def __init__(self, n_features=40, reg=0., n_iter=40, random_state=0, verbose=False):
        self.n_features = n_features
        self.reg = reg
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose

        
    def fit(self, X, y=None):
        '''Learn a MF model for the data X.

        Parameters
        ----------
        X : , shape (n_users, n_items)
            Data matrix to be decomposed

        y : Ignored

        Returns
        ------
        self
        '''

        tstart_ = time.time()
        self.random_state_ = check_random_state(self.random_state)

        if self.verbose:
            print(f"Starting MatrixFactorization.")

        if self.n_iter:
            n_iter = self.n_iter
        else:
            n_iter = self.max_iter

        self.V_ = self.random_state_.randn(X.shape[1],self.n_features)
        U = self.random_state_.randn(X.shape[0],self.n_features)

        
        self.bv_ = numpy.zeros(X.shape[1])
        bu = numpy.zeros(X.shape[0])

        self.mu_ = X.data.mean()
        
        
        for current_iter in range(n_iter):
            if self.verbose:
                print(f"Starting iteration {current_iter} of {n_iter}.")
            # Do ALS step


        if self.verbose:
            print(f"Converged at iteration {current_iter}")

        return self
            
        
    def predict(self, X):
        return None

            
    def loss(self, X):
        return None


    def score(self, X):
        score = -1.0 * self.loss(X)
        return score
