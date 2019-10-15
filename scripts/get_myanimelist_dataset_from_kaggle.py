import kaggle
import sys
import os

path = "data/raw/"

for file_ in ['AnimeList.csv', 'UserList.csv', 'UserAnimeList.csv']:
    os.system(f'kaggle datasets download --file {file_} --path {path} --unzip azathoth42/myanimelist')
