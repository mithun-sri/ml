from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn


def one_dim_linear_reg():
    X, y = mglearn.datasets.make_wave(n_samples=60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    lr = LinearRegression().fit(X_train, y_train)
    print("lr.coef_: {}".format(lr.coef_))
    print("lr.intercept_:{}".format(lr.intercept_))
    print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))


def boston_multi_dim():
    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    lr = LinearRegression().fit(X_train, y_train)
    print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))


def ridge_reg():
    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Alpha = 1

    ridge = Ridge(alpha=1).fit(X_train, y_train)
    print("Training set score (alpha = 1): {:.2f}".format(ridge.score(X_train, y_train)))
    print("Test set score (alpha = 1): {:.2f}".format(ridge.score(X_test, y_test)))

    # Alpha = 10

    ridge10 = Ridge(alpha=10).fit(X_train, y_train)
    print("Training set score (alpha = 10): {:.2f}".format(ridge10.score(X_train, y_train)))
    print("Test set score (alpha = 10): {:.2f}".format(ridge10.score(X_test, y_test)))

    # Alpha = 0.1

    ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
    print("Training setit(X_t score (alpha = 0.1): {:.2f}".format(ridge01.score(X_train, y_train)))
    print("Test set score (alpha = 0.1): {:.2f}".format(ridge01.score(X_test, y_test)))

    lr = LinearRegression().fit(X_train, y_train)

    plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
    plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
    plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

    plt.plot(lr.coef_, 'o', label="LinearRegression")
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.ylim(-25, 25)
    plt.legend()
    plt.show()

    mglearn.plots.plot_ridge_n_samples()
    plt.show()


def get_req():
    print("1. One-dimensional linear regression")
    print("2. Multi-dimensional linear regression")
    print("3. Ridge regression")
    req = int(input("Enter command: "))
    if req == 1:
        one_dim_linear_reg()
    elif req == 2:
        boston_multi_dim()
    elif req == 3:
        ridge_reg()
    else:
        print("\nIncorrect input!\n")
        get_req()


if __name__ == "__main__":
    get_req()
