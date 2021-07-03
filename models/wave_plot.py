import mglearn
import matplotlib.pyplot as plt

# Forms a scatter plot using the wave dataset

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')

# Sets the range of the y-interval and plots the graph

plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()
