from cProfile import label
import math
import doctest
from random import seed
from re import X
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

def ture_function(x):
    """ Return a ture value
    >>> ture_function(0)
    0.0
    """
    y = math.sin(math.pi * x * 0.8) * 10

    return y

def get_plot_dataset(n=20):

    #観測点から、ture_danctionにより、真値を取り出す。
    X = obs_function(n)
    X_t = np.zeros(len(X))
    for i, x in enumerate(X):
        X_t[i] = ture_function(x)

    #ex1のデータセット
    magerX=np.arange(-1,1,0.1)
    t = np.zeros(len(X))
    for i, x in enumerate(magerX):
        t[i] = ture_function(x)

    df = pd.DataFrame(columns=["観測点","真値"])
    df["観測点"]=X
    df["真値"]=X_t

    #ex3
    #20のサンプルから正規分布(loc=0.0, scale=2.0)/2のノイズを真値に付与。ノイス+真値。
    obs_t = np.random.normal(0.0, 2.0, 20)/2 + X_t
    df=df.assign(観測値=obs_t)

    return df

def plot(df):
    """  Plot the ture value

    """
    magerX=np.arange(-1,1,0.1)
    ex1 = np.zeros(len(magerX))
    for i, x in enumerate(magerX):
        ex1[i] = ture_function(x)
    
    x = df["観測点"]
    t = df["真値"]
    n_t = df["観測値"]
    #ex1.1のplot
    #plt.scatter(magerX, ex1, label=f"ex1_1 functionn n={len(ex1)}", color="red")
    plt.plot(magerX, ex1, label=f"ex1_1 function n={len(ex1)}", color="red")

    #ex1.2のplot
    plt.scatter(x, t, label=f"observer's function n={len(t)}")

    #ex3
    plt.scatter(x, n_t, label=f"ex3 function n={len(t)}")

    plt.legend()
    plt.show()


def obs_function(n=20):
    """ Return the observer value
    >>> len(obs_function(20))
    20
     """
    np.random.seed(0)
    obs_X = np.random.rand(n)
    return obs_X

if __name__ == "__main__":
    #doctest.testmod()
    dataset = get_plot_dataset(20)
    #print(dataset)
    plot(dataset)
    