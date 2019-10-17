import os
import sys
import pickle


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

sys.path.append(get_script_path()+"/..")


from animerec import models


k = 40
reg = 20.
steps = 50

pickle_in = open("data/scores.pickle", "rb")
X = pickle.load(pickle_in)
pickle_in.close()

model = models.MatrixFactorization(k=k, reg=reg)
model.fit(X,steps=steps)

pickle_out = open("model/model.pickle", "wb")
pickle.dump(model,pickle_out)
pickle_out.close()
