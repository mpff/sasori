TARGETS := raw data train

.PHONY: all raw data train help


.DEFAULT: help
help:
	@echo "raw"
	@echo "  Downloads raw dataset"
	@echo "data"
	@echo "  Pickles datasets as scipy sparse matrix."
	@echo "train"
	@echo "  Trains MatrixFactorization model and pickles it."


all: $(TARGETS)


train: data/scores.pickle
	python scripts/train_model_and_save_to_pickle.py


data: data/animes.pickle data/users.pickle data/items.pickle data/scores.pickle

data/animes.pickle data/users.pickle data/items.pickle: data/scores.pickle

data/scores.pickle: data/raw/AnimeList.csv data/raw/UserList.csv data/raw/UserAnimeList.csv scripts/load_dataset_in_chunks_and_save_to_pickle.py
	python scripts/load_dataset_in_chunks_and_save_to_pickle.py


raw: data/raw/AnimeList.csv data/raw/UserList.csv data/raw/UserAnimeList.csv
	
data/raw/AnimeList.csv data/raw/UserList.csv: data/raw/UserAnimeList.csv

data/raw/UserAnimeList.csv: scripts/get_myanimelist_dataset_from_kaggle.py
	python tests/test_can_access_kaggle_api.py
	python scripts/get_myanimelist_dataset_from_kaggle.py
