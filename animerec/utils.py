import time
import requests
import pandas
import numpy as np
import matplotlib.pyplot as plt


class MatrixFactorization():
    
    def __init__(self, k, reg):
        self.k = k
        self.reg = reg

        
    def fit(self, X, steps):
        ''' Performs Sparse Matrix Factorization with bias terms. '''
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
            
            # Update user matrix U.
            X = X.tocsr()
            for i in range(self.U.shape[0]):
                jvec = X.getrow(i).nonzero()[1]
                matrix_ = self.V[jvec].T.dot(self.V[jvec]) + np.eye(self.k) * self.reg
                vector_ = (X[i,jvec] - (self.bu[i] + self.bv[jvec] + self.mu)).dot(self.V[jvec]).T
                self.bu[i] = (X[i,jvec] - self.V[jvec].dot(self.U[i]) - self.bv[jvec] - self.mu).sum()
                self.bu[i] = self.bu[i] / ( len(jvec) + self.reg)
                self.U[i] = np.squeeze(np.linalg.solve(matrix_,vector_))
            
            # Update item matrix V.
            X = X.tocsc()
            for j in range(self.V.shape[0]):
                ivec = X.getcol(j).nonzero()[0]
                matrix_ = self.U[ivec].T.dot(self.U[ivec]) + np.eye(self.k) * self.reg
                vector_ = (X[ivec,j].T - self.bu[ivec] - self.bv[j] - self.mu).dot(self.U[ivec]).T
                self.bv[j] = (X[ivec,j].T - (self.U[ivec].dot(self.V[j]) + self.bu[ivec] + self.mu)).sum()
                self.bv[j] = self.bv[j] / ( len(ivec) + self.reg)
                self.V[j] = np.squeeze(np.linalg.solve(matrix_,vector_))

            # Calculate Training Error.
            self.error.append(self.loss(X))
            print(f"Iteration {t:2d} :  Training Error = {self.error[-1]:3.4f}  Time = {time.time()-tstart:.2f}s")
            
        self.plot_loss()
        
        
    def predict(self, x):
        u = (x - self.mu - self.bv)
        
        w = ~np.isnan(x)
        n = w.sum()
        
        bu = u[w].mean()
        u = u - bu
        
        # Calculate user feacture vector and anime feature covariance matrix.
        vu_, vcov_ = (0,0)
        for i,e in enumerate(x):
            if not np.isnan(e):
                vu_ += self.V[i,:]*u[i]
                vcov_ += np.outer(self.V[i,:], self.V[i,:])
        vcov_ = np.linalg.inv(vcov_)
        
        
        vu_ = np.dot(self.V[w,:].T, u[w])
        vcov_ = np.outer(self.V[w,:], self.V[w,:])
        print(vcov_)
        vcov_ = np.linalg.inv(vcov_)
    
        # Predict scores.
        uhat = vcov_.dot(vu_)
        xhat = uhat.dot(self.V.T) + bu + self.bv + self.mu
        
        return xhat
           
            
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

    

def print_features(model, data, nfeatures=0) :
    ''' 
    Prints top and bottom scoring anime for nfeatures
    features of model. 
    '''

    if model.k <= nfeatures:
        print(f"Only {model.k} features in model.\
                Printin first {model.k} features.")
        nfeatures = model.k
    elif nfeatures == 0:
        print(f"Printing all {model.k} features.")
        nfeatures = model.k
    else:
        print(f"Printing {nfeatures} of {model.k} features.")

    for k in range(nfeatures):
        jmax = model.V[:,k].argmax()
        jmin = model.V[:,k].argmin()
        idmax = data.cindex[jmax]
        idmin = data.cindex[jmin]
        amax = data.get_anime_by_id(idmax)
        amin = data.get_anime_by_id(idmin)
        str_ = (
            f'Feature {k+1}:\n'
            f'\tmax : ({model.V[jmax,k]:+2.1f}) {amax.title}\n'
            f'\tmin : ({model.V[jmin,k]:+2.1f}) {amin.title}'
        )
        print(str_)

    return None

    
    
def get_user_anime_list(user_id):
    '''
    Based on QasimK/mal-scraper.
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

        time.sleep(1.5)  # TODO: smarter wait

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



def get_score_vector_from_user_anime_list(user_anime_list, cindex):
    
    vec = pandas.Series(index=cindex)
    
    for anime in user_anime_list:
        if anime['status'] not in [2,4]:  # Only completed and dropped animes.
            continue
        vec[anime['id_ref']] = anime['score']
        
    vec = vec[cindex]
    
    return np.array(vec)



def prediction_to_dataframe(xhat, user_anime_list, cindex, keep_all=False):
    
    prediction = pandas.Series(xhat, index=cindex)
    
    if not keep_all:
        watched = [a['id_ref'] for a in user_anime_list if a['id_ref'] in prediction.index]
        prediction = prediction.drop(list(watched))  # Note: columns == anime_ids's
    
    return prediction
    
