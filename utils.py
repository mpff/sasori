import time
import requests
import numpy as np
import matplotlib.pyplot as plt


class MatrixFactorization():
    
    def __init__(self, k, reg):
        self.k = k
        self.reg = reg

        
    def fit(self, X, steps):
        self.U = np.random.randn(X.shape[0],self.k)
        self.V = np.random.randn(X.shape[1],self.k)
        self.bu = np.zeros(X.shape[0])
        self.bv = np.zeros(X.shape[1])
        self.mu = X.data.mean()
        
        tstart = time.time()
        self.error = [self.loss(X)]
        print(f"Iteration  0 :  Training Error = {self.error[-1]:3.4f}  Time = {time.time()-tstart:.2f}s")
        
        for t in range(1,steps+1):
            tstart = time.time()
            
            # Update U
            X = X.tocsr()
            for i in range(self.U.shape[0]):
                jvec = X.getrow(i).nonzero()[1]
                matrix_ = self.V[jvec].T.dot(self.V[jvec]) + np.eye(self.k) * self.reg
                vector_ = (X[i,jvec] - (self.bu[i] + self.bv[jvec] + self.mu)).dot(self.V[jvec]).T
                self.bu[i] = (X[i,jvec] - self.V[jvec].dot(self.U[i]) - self.bv[jvec] - self.mu).sum()
                self.bu[i] = self.bu[i] / ( len(jvec) + self.reg)
                self.U[i] = np.squeeze(np.linalg.solve(matrix_,vector_))
            
            # Update V
            X = X.tocsc()
            for j in range(self.V.shape[0]):
                ivec = X.getcol(j).nonzero()[0]
                matrix_ = self.U[ivec].T.dot(self.U[ivec]) + np.eye(self.k) * self.reg
                vector_ = (X[ivec,j].T - self.bu[ivec] - self.bv[j] - self.mu).dot(self.U[ivec]).T
                self.bv[j] = (X[ivec,j].T - (self.U[ivec].dot(self.V[j]) + self.bu[ivec] + self.mu)).sum()
                self.bv[j] = self.bv[j] / ( len(ivec) + self.reg)
                self.V[j] = np.squeeze(np.linalg.solve(matrix_,vector_))

 
                
            self.error.append(self.loss(X))
            
            print(f"Iteration {t:2d} :  Training Error = {self.error[-1]:3.4f}  Time = {time.time()-tstart:.2f}s")
            
        self.plot_loss()
           
            
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


    
def get_user_anime_list(user_id):
    '''
    Based on QasimK/mal-scraper
    Returns animelist of user 'user_id' as list of dictionaries. Keys:
        'name':     anime title
        'id_ref':   anime id
        'status':   consumption status (1:=watching, 2:=completed, 3:= ...)
        'score':    anime score from user
    '''

    anime = []

    has_more_anime = True
    while has_more_anime:
        url = f'https://myanimelist.net/animelist/{user_id}/load.json?offset={len(anime)}&status=7'
        response = requests.get(url)

        time.sleep(1)  # TODO: smarter wait

        if not response.ok:  # Raise an exception
            # TODO: build an actual exception
            print("Response not ok!")
            return None

        additional_anime = get_user_anime_list_from_json(response.json())
        if additional_anime:
            anime.extend(additional_anime)
            print(f"Scraped {len(additional_anime)} additional anime from {url}")
        else:
            has_more_anime = False

    return anime



def get_user_anime_list_from_json(json):
    anime = []
    for mal_anime in json:

        tags = set(
            filter(
                bool,  # Ignore empty tags
                map(
                    str.strip,  # Splitting by ',' leaves whitespaces
                    str(mal_anime['tags']).split(','),  # Produce a list
                    # Sometimes the tag is an integer itself
                )
            )
        )

        anime.append({
            'name': mal_anime['anime_title'],
            'id_ref': int(mal_anime['anime_id']),
            'status': mal_anime['status'],
            'score': int(mal_anime['score'])
        })

    return anime


def user_anime_list_to_score_vector(user_anime_list):
    return None
