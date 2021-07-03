import sys
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    iris_dataset = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'],
                                                        random_state=np.random)

    # Produces a dataframe and scatter matrix which can be used for visualisation needs

    iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20},
                                     s=60, alpha=.8, cmap=mglearn.cm3)

    # Using k-Nearest Neighbors to produce a classification algorithm and fitting training dataset
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    # Algorithm makes predictions for test data and outputs predictions

    y_prediction = knn.predict(X_test)
    print("Test set predictions:\n {}".format(y_prediction))

    # Output test set accuracy in predictions

    print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
