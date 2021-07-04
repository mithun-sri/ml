import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Forms a scatter plot using the wave dataset

X, y = mglearn.datasets.make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def create_plot():
    plt.plot(X, y, 'o')
    # Sets the range of the y-interval and plots the graph
    plt.ylim(-3, 3)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.show()


def one_n_regression():
    # KNN regression for 1 neighbor
    mglearn.plots.plot_knn_regression(n_neighbors=1)
    plt.show()


def three_n_regression():
    # KNN regression for 3 neighbors
    mglearn.plots.plot_knn_regression(n_neighbors=3)
    plt.show()

    # Make predictions on 3 neighbor regression
    reg = KNeighborsRegressor(n_neighbors=3)
    reg.fit(X_train, y_train)
    print("Test set predictions:\n{}".format(reg.predict(X_test)))
    print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))


def analyse_reg():
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    # create 1,000 data points, evenly spaced between -3 and 3
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)
    for n_neighbors, ax in zip([1, 3, 9], axes):
        # make predictions using 1, 3 or 9 neighbors
        reg = KNeighborsRegressor(n_neighbors=n_neighbors)
        reg.fit(X_train, y_train)
        ax.plot(line, reg.predict(line))
        ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
        ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)

        ax.set_title(
            "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
                n_neighbors, reg.score(X_train, y_train), reg.score(X_test, y_test)
            )
        )
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
    axes[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")
    plt.show()


def get_request():
    print("1. Create plot")
    print("2. 1-nearest neighbor regression")
    print("3. 3-nearest neighbor regression")
    print("4. Analyse regression")
    req = int(input("Enter command: "))
    if req == 1:
        create_plot()
    elif req == 2:
        one_n_regression()
    elif req == 3:
        three_n_regression()
    elif req == 4:
        analyse_reg()
    else:
        print("\nInvalid input. Try again.\n")
        get_request()


if __name__ == "__main__":
    get_request()
