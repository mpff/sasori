import os
import pandas
import numpy
import pickle
from scipy.sparse import csc_matrix, vstack
from tqdm import tqdm

# Use this to link anime id to anime metadata.
animes = pandas.read_csv(f'data/AnimeList.csv')


# Dataset needs to be loaded in chunks as it is too big for RAM.
reader = pandas.read_csv(f'data/UserAnimeList.csv', chunksize=1e5)

# Lists to save user and item index, as sorting on whole dataset is not possible.
items = sorted(animes.anime_id.tolist())
users = []

chunks = []
# Each chunk is in dataframe format.
for chunk in tqdm(reader):
    chunk = chunk[['username', 'anime_id', 'my_score']]
    chunk = chunk[chunk.my_score != 0]

    # Transform DataFrame to (N=#users x K=#animes) matrix.
    chunk = chunk.pivot(index="username", columns="anime_id", values="my_score")

    # populate dataframe with all anime columns. 
    chunk = chunk.reindex(columns = items)
    
    # add current chunk user positions to row index list.
    users = users + chunk.index.tolist()
    
    # convert to scipy sparse matrix.
    chunk = csc_matrix(chunk.fillna(0))

    chunks.append(chunk)


scores = vstack(chunks, format="csc")


# Save to .pickle
for obj in ["animes", "items", "users", "scores"]:
    pickle_out = open(f"data/{obj}.pickle", "wb")
    pickle.dump(globals()[obj],pickle_out)
    pickle_out.close()

