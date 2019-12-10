import time
import requests
import pandas
import numpy


def split(X, k=5, seed=1234):
    """
    Partition dataset into k parts. K training and one test set.
    
    Returns:
        Xtest: list (if k>1) of k-1 sparse matrices or (if k=1) sparse matrix
        Xtrain: sparse matrix
        R: list of k permutation vectors
        
    """
    n = X.shape[0]

    rstate = numpy.random.mtrand.RandomState(seed)

    R = numpy.array_split(rstate.permutation(n),k+1)
    
    if k == 1:
        Xtrain = X[R[0]]
    else:
        Xtrain = [X[r] for r in R[:-1]]  # k-fold Training data

    Xtest  = X[R[-1]]  # Test data

    return Xtrain,Xtest,R



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
#        if anime['status'] not in [2,4]:  # Only completed and dropped animes.
#            continue
        if anime['score'] == 0:
            continue
        vec[anime['id_ref']] = anime['score']
    vec = vec[cindex]
    return numpy.array(vec)



def prediction_to_dataframe(xhat, user_anime_list, cindex, keep_all=False):
    prediction = pandas.Series(xhat, index=cindex)
    if not keep_all:
        user_anime_list = [anime for anime in user_anime_list if anime['status'] == 2]
        completed = [a['id_ref'] for a in user_anime_list if a['id_ref'] in prediction.index]
        prediction = prediction.drop(list(completed))  # Note: columns == anime_ids's
    return prediction
