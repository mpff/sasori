Hey there! 
In this file I want to give a 'high'-level overview of this projects directory structure and internal dependencies.

### Make

#### build
1. `test_can_access_kaggle.py`
2. `get_myanimelist_dataset_from_kaggle.py` --> `data/raw/*`
3. `pickle_myanimelist_dataset.py` --> `data/*.pickle`

#### train
4. `train_model_and_save_to_pickle.py` --> `model/model.pickle`, `model/anime-ids.pickle`


### Directory Structure

#### `/animerec`
Contains the core package.
  * `/animerec/models.py` : Module containing the matrix factorization and deep learning models.
  * `/animerec/datasets.py` : Module containing functions for loading the dataset (in chunks).
  * `/animerec/utils.py` : Module containing various utility functions, e.g. for scraping current animelists and converting them to numpy arrays for prediction.

#### `/data`
Contains the raw and pickled datasets.
  * `/data/raw/*` : Contains the raw MAL data downloaded from Kaggle by `get_myanimelist_dataset_from_kaggle.py`.
  * `/data/*.pickle` : Contains pickled list of all animes in the dataset created by `pickle_myanimelist_dataset.py`
  
#### `/model`
Contains the latest trained model and an anime identifier file that allows mapping columns in the model to anime ids.

#### `/scripts`
Contains the scripts used by make to download and pickle datasets and train models.

#### `/tests`
Contains tests run by make.

#### `/docs`
Contains project overviews, links to resources and mathematical explainations of the algorithms implemented.
