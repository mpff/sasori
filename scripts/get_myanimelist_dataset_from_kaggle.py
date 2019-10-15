import kaggle
import sys
import os

path = "data/raw/"

for file_ in ['AnimeList.csv', 'UserList.csv', 'UserAnimeList.csv']:
    os.system(f'kaggle datasets download\
            --file {file_}\
            --path {path}\
            azathoth42/myanimelist')
    os.system(f'unzip {path+file_} -d {path}')
    os.system(f'rm {path+file_}.zip')
