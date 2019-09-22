import pandas
from scipy.sparse import csr_matrix, vstack


def load(chunksize=1e6):
    '''
    Loads the MyAnimeList dataset in chunks.

    Returns:
        X : sparse matrix of dimension #users times #animes
        cindex : list of column indices of X (anime id)
        rindex : list of row indices of X (users)

    '''

    reader = pandas.read_csv('data/UserAnimeList.csv', chunksize = chunksize)

    chunks = []
    cindex = []
    rindex = []

    # Each chunk is in dataframe format
    for chunk in reader:  
        chunk = chunk[['username', 'anime_id', 'my_score']]
        chunk = chunk[chunk.my_score != 0]
        
        # Transform DataFrame to (N=#users x K=#animes) matrix
        chunk = chunk.pivot(index="username", columns="anime_id", values="my_score")
        
        rindex = rindex + chunk.index.tolist()
        cindex = cindex + chunk.columns.tolist()
        
        chunks.append(chunk)

    for chunk in chunks:
        chunk = chunk.reindex(columns = cindex)
        chunk = csr_matrix(chunk.fillna(0))

    X = vstack(chunks, format="csr")

    return X, cindex, rindex


def download():
    return None
