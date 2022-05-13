from random import seed
from statistics import mean, mode

from sklearn.utils import shuffle
import dataset1
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np

dataset = dataset1.read_file()
data, target = dataset["観測点"], dataset["観測値"]

#npでreshpeしろって怒られたからつけた。
rep_data = np.array(data).reshape(-1,1)

model = LinearRegression()

k_folds = 8
#reg = model.fit(rep_data, target)

score = cross_val_score(model, rep_data, target, cv=KFold(n_splits=k_folds, shuffle=True))
average = mean(score)
print(f"score: {score}")
print(f"average: {average}")




