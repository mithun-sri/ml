from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import mglearn
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Standard lasso regression
    lasso = Lasso().fit(X_train, y_train)
    print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

    # Lasso regression with alpha = 0.01 and increased max iterations

    lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
    print("\nTraining set score: {:.2f}".format(lasso001.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

    # Lasso regression with alpha = 0.0001 and increased max iterations
    # Demonstrates that decreasing alpha too much can lead to overfitting of training set

    lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
    print("\nTraining set score: {:.2f}".format((lasso00001.score(X_train, y_train))))
    print("Test set score: {:.2f}".format((lasso00001.score(X_test, y_test))))
    print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

    # Ridge regression
    ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)

    # Plotting lasso regression

    plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
    plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
    plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")

    plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
    plt.legend(ncol=2, loc=(0, 1.05))
    plt.ylim(-25, 25)
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.show()
