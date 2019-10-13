import datasets, utils
import pickle
from datetime import datetime

k = 40
reg = 20.
steps = 10

data = datasets.MyAnimeList()
model = utils.MatrixFactorization(k=k, reg=reg)
model.fit(data.X,steps=steps)

tstamp = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
opath = f"{tstamp}_k={k}_reg={reg:.1f}.pickle"

pickle_out = open("model.pickle", "wb")
pickle.dump(model,pickle_out)
pickle_out.close()
