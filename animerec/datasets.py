import os
import pandas
import numpy
import pickle
from scipy.sparse import csc_matrix, vstack
from tqdm import tqdm


class MyAnimeList:

    def __init__(self,extension="",debug=False, chunksize=1e6):
        if extension != "":
            extension = "_"+extension
        self.animes = pandas.read_csv(f'data/AnimeList{extension}.csv')
        self.users  = pandas.read_csv(f'data/UserList{extension}.csv')
        self.X,self.cindex,self.rindex = self.read_useranimelist(f'data/UserAnimeList{extension}.csv',debug,chunksize)
    
    
    def split(self, k=5, seed=1234):
        """
        Partition dataset into k parts. K training and one test set.
        
        Returns:
            Xtest: list (if k>1) of k-1 sparse matrices or (if k=1) sparse matrix
            Xtrain: sparse matrix
            R: list of k permutation vectors
            
        """
        n = self.X.shape[0]

        rstate = numpy.random.mtrand.RandomState(seed)

        R = numpy.array_split(rstate.permutation(n),k+1)
        
        if k == 1:
            Xtrain = self.X[R[0]]
        else:
            Xtrain = [self.X[r] for r in R[:-1]]  # k-fold Training data

        Xtest  = self.X[R[-1]] # Test data

        return Xtrain,Xtest,R
    
    
    def get_anime_by_id(self, id_):
        return self.animes[self.animes.anime_id == id_].iloc[0]

    
    def read_useranimelist(self, path, debug=False, chunksize=1e6):
        '''
        Reads the MyAnimeList dataset in chunks. If debug=True: only loads
        a single chunk.

        Returns:
            X : sparse matrix of dimension #users times #animes
            cindex : list of column indices of X (anime id)
            rindex : list of row indices of X (users)

        '''
        if not debug:
            if os.path.exists("data/MyAnimeList.pickle"):
                pickle_in = open("data/MyAnimeList.pickle","rb")
                data = pickle.load(pickle_in)
                pickle_in.close()
                return data.X, data.cindex, data.rindex
            

        reader = pandas.read_csv(path,chunksize=chunksize)

        chunks = []
        cindex = sorted(self.animes.anime_id.tolist())
        rindex = []
        
        # Each chunk is in dataframe format
        for chunk in tqdm(reader):
            chunk = chunk[['username', 'anime_id', 'my_score']]
            chunk = chunk[chunk.my_score != 0]

            # Transform DataFrame to (N=#users x K=#animes) matrix
            chunk = chunk.pivot(index="username", columns="anime_id", values="my_score")

            # populate dataframe with all anime columns            
            chunk = chunk.reindex(columns = cindex)
            
            # add current chunk user positions to row index list
            rindex = rindex + chunk.index.tolist()
            
            # convert to scipy sparse matrix
            chunk = csc_matrix(chunk.fillna(0))

            chunks.append(chunk)
            
            if debug: break  # Load only one chunk.


        X = vstack(chunks, format="csc")

        return X, cindex, rindex
