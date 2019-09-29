'''
Based on QasimK/mal-scraper
'''
import time
import requests

def get_user_anime_list(user_id):
    '''
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

        time.sleep(2)  # TODO: smarter wait

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
