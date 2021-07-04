import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Generate dataset

X, y = mglearn.datasets.make_forge()

# Plotting dataset


def plot_dataset():
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(["Class 0", "Class 1"], loc=4)
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    mglearn.plots.plot_knn_classification(n_neighbors=3)
    plt.show()
    print("X.shape: {}".format(X.shape))

# Displaying decision boundaries for 1, 3 and 9 neighbor kNN classifications


def display_decision_boundaries():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for n_neighbors, ax in zip([1, 3, 9], axes):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title("{} neighbor(s)".format(n_neighbors))
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
    axes[0].legend(loc=3)
    plt.show()


def get_request():
    print("1. Plot dataset")
    print("2. Display decision boundaries")
    req = int(input("Enter task: "))
    if req == 1:
        plot_dataset()
    if req == 2:
        display_decision_boundaries()
    else:
        print("Incorrect input.\n")
        get_request()
    exit()


if __name__ == "__main__":
    get_request()

