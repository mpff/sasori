TARGETS := data train

.PHONY: help data train all

.DEFAULT: help
help:
	@echo "data"
	@echo "  Downloads and pickles datasets as sparse matrix."
	@echo "train"
	@echo "  Trains a model and saves pickled weights."

all: $(TARGETS)

train: data

data: raw

raw: data/raw/AnimeList.csv data/raw/UserList.csv data/raw/UserAnimeList.csv

data/raw/AnimeList.csv data/raw/UserList.csv data/raw/UserAnimeList.csv:
	python tests/test_can_access_kaggle_api.py
	python scripts/get_myanimelist_dataset_from_kaggle.py
