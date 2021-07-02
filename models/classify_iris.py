import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import mglearn

if __name__ == "__main__":
    iris_dataset = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
    iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60,
                                     alpha=.8, cmap=mglearn.cm3)
