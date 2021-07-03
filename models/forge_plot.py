import mglearn
import matplotlib.pyplot as plt

# Generate dataset

X, y = mglearn.datasets.make_forge()

# Plotting dataset

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
print("X.shape: {}".format(X.shape))
