Sasori
======

**At the moment this README is more of a design documentation. Nothing actually works yet ;)**

Sasori is a Python scikit 
for building building recommender systems 
that are serveable and scaleable.

**Sasori is designed with the following purposes in mind**:

- Allow users to make out-of-sample and cold-start predictions.
- Reduce model size by only saving item related weights.
- Integrate easily into existing scikit-learn pipelines.

Algorithms are implemented by adhering to the standards of scientific rigor.
To this end, references to relevant papers and derivations of the algorithms
will be provided.

The name **Sasori** stems from the initial letters of 
**s**erveable, **s**erviceable and **r**ecommender system. 
*Sasori* (japanese for scorpion) is also the name of 
the main character of the 1972 Japanese film 
*Female Convict 701: Scorpion* and its sequels.


Getting started
---------------

```python
from sasori import MatrixFactorization
from sasori.datasets import Movielens
from sklearn.model_selection import train_test_split

# Load the movielens-100k dataset (download it if needed).
data = Movielens.load'ml-100k')

# Split dataset into train and test set.
Xtrain, Xtest = train_test_split(data, test_size=0.1)

# Use a basic matrix factorization approach.
model = MatrixFactorization(verbose=True)

# Fit the model.
model.fit(Xtrain)

# Check RMSE on test dataset.
model.score(Xtest)
```


Bechnchmarks
------------

Performing a regularized biased matrix factorization with 100 features.

| Movielens 100k           |   RMSE |   MAE | Time    | Size | Predict oos |
|:-------------------------|-------:|------:|:--------|:-----|------------:|
| MF (sasori)              |     na |    na |      na |   na |         yes |
| SVD (surprise)           |  0.934 | 0.737 | 0:00:11 |   na |          no |
| na (spotlight)           |     na |    na |      na |   na |          na |


Installation
------------

Installation with pip (you will need a C compiler):

    $ pip install --user sasori 
