import os
import sys
import time
import pickle

from sklearn.model_selection import RandomizedSearchCV, train_test_split

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(get_script_path()+"/..")

from animerec.models import MatrixFactorization as MF


SEED = 123


# Load Dataset.
pickle_in = open("data/scores.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in.close()


# Perform Randomized Grid Search on Train dataset.
param_grid = {
    'n_features': [10, 50, 200],
    'reg': [0.1, 1, 10, 100]
}

Xtrain, Xtest = train_test_split(X, test_size=0.1, shuffle=True,
                                 random_state=SEED)
grid_search = RandomizedSearchCV(MF(verbose=True), param_grid, n_iter=5, cv=5,
                                 random_state=SEED, verbose=50)

t0 = time.time()
grid_search.fit(Xtrain)


# Print Diagnostics.
print("done in %0.3fs" % (time.time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# Save Model.
pickle_out = open("model/grid_search.pickle", "wb")
pickle.dump(grid_search, pickle_out)
pickle_out.close()
