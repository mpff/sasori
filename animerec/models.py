import time
import numpy
import matplotlib.pyplot as plt


class MatrixFactorization():
    
    def __init__(self, k, reg):
        self.k = k
        self.reg = reg

        
    def fit(self, X, steps):
        '''
        Performs Sparse Matrix Factorization with bias terms.

        Input:

            X :  scipy.sparse.csc_matrix
            steps :  int
        '''
        
        self.U = numpy.random.randn(X.shape[0],self.k)
        self.V = numpy.random.randn(X.shape[1],self.k)
        self.bu = numpy.zeros(X.shape[0])
        self.bv = numpy.zeros(X.shape[1])
        self.mu = X.data.mean()
        
        tstart = time.time()
        self.error = [self.loss(X)]
        print(f"Iteration  0 :  Training Error = {self.error[-1]:3.4f}  Time = {time.time()-tstart:.2f}s")
        
        for t in range(1,steps+1):
            tstart = time.time()
            
            # Update user matrix U.
            X = X.tocsr()
            for i in range(self.U.shape[0]):
                jvec = X.getrow(i).nonzero()[1]
                matrix_ = self.V[jvec].T.dot(self.V[jvec]) + numpy.eye(self.k) * self.reg
                vector_ = (X[i,jvec] - (self.bu[i] + self.bv[jvec] + self.mu)).dot(self.V[jvec]).T
                self.bu[i] = (X[i,jvec] - self.V[jvec].dot(self.U[i]) - self.bv[jvec] - self.mu).sum()
                self.bu[i] = self.bu[i] / ( len(jvec) + self.reg)
                self.U[i] = numpy.squeeze(numpy.linalg.solve(matrix_,vector_))
            
            # Update item matrix V.
            X = X.tocsc()
            for j in range(self.V.shape[0]):
                ivec = X.getcol(j).nonzero()[0]
                matrix_ = self.U[ivec].T.dot(self.U[ivec]) + numpy.eye(self.k) * self.reg
                vector_ = (X[ivec,j].T - self.bu[ivec] - self.bv[j] - self.mu).dot(self.U[ivec]).T
                self.bv[j] = (X[ivec,j].T - (self.U[ivec].dot(self.V[j]) + self.bu[ivec] + self.mu)).sum()
                self.bv[j] = self.bv[j] / ( len(ivec) + self.reg)
                self.V[j] = numpy.squeeze(numpy.linalg.solve(matrix_,vector_))

            # Calculate Training Error.
            self.error.append(self.loss(X))
            print(f"Iteration {t:2d} :  Training Error = {self.error[-1]:3.4f}  Time = {time.time()-tstart:.2f}s")
            
        self.plot_loss()
        
        
    def predict(self, r):
        rb = (r - self.mu - self.bv)
        w = ~numpy.isnan(r)
        
        # Create (1,features) item vector.
        V = numpy.c_[ numpy.ones(self.V.shape[0]), self.V ]
        
        # Predict (bias, features) user vector.
        vector = numpy.dot(V[w,:].T, rb[w])
        matrix = V[w,:].T.dot(V[w,:]) + self.reg * numpy.eye(self.k+1)
        xhat = numpy.linalg.solve(matrix, vector)
                
        # Predict scores.
        rhat = xhat[1:].dot(V[:,1:].T) + xhat[0] + self.bv + self.mu
        
        return rhat
           
            
    def loss(self, X):
        N = 0.
        E = 0.
        for j in range(self.V.shape[0]):
            ivec = X.getcol(j).nonzero()[0]
            xtru = X[ivec,j].todense().T
            xhat = self.U[ivec].dot(self.V[j].T) + self.bu[ivec] + self.bv[j] + self.mu
            resd = xtru - xhat
            E += resd.dot(resd.T)
            N += len(ivec)
        return E[0,0] / N
    
    
    def plot_loss(self):
        plt.plot(self.error,color="C0", label="Training Error")
        plt.legend()
        plt.show()
