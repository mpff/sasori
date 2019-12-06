import os
import sys
import time
import pickle


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(get_script_path()+"/..")


from sklearn.model_selection import GridSearchCV, train_test_split

from animerec.models import MatrixFactorization as MF


param_grid = {
    'n_features': [5, 10, 25, 50, 100, 150, 200],
    'reg': [0.1, 1, 10, 50]
}


# Load Dataset
pickle_in = open("data/scores.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in.close()


Xtrain, Xtest = train_test_split(X, test_size=0.1, shuffle=True, random_state=123)
grid_search = GridSearchCV(MF(), param_grid, cv=5, n_jobs=-1, verbose=3)

t0 = time.time()
grid_search.fit(Xtrain)

print("done in %0.3fs" % (time.time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

pickle_out = open("model/grid_search.pickle", "wb")
pickle.dump(grid_search, pickle_out)
pickle_out.close()
