from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import pandas as pd

k_range = range(1, 31)

knn = KNeighborsClassifier()

iris = load_iris()

X = iris.data
y = iris.target

# create a parameter grid: map thhe parameter names to the values that should
# be searched
param_grid = dict(n_neighbors=k_range)

# instantiate the grid
grid = GridSearchCV(
    knn, param_grid, cv=10, scoring='accuracy', return_train_score=False
)

# you can set n_jobs = -1 to run computations in parallel
# (if supported by your computer and OS)

# fit the grid with data
grid.fit(X, y)

# view complete results
results = pd.DataFrame(grid.cv_results_)[
    ['mean_test_score', 'std_test_score', 'params']
]

# print(results)

# examining the first result
print(grid.cv_results_['params'][0])
print(grid.cv_results_['mean_test_score'][0])

grid_mean_scores = grid.cv_results_['mean_test_score']

# plot the results
plt.plot(k_range, grid_mean_scores)
plt.xlabel('K')
plt.ylabel('cross validation accuracy')
plt.show()

# selecting the best model
print(grid.best_score_, grid.best_params_, grid.best_estimator_)

# you can also define weights parameter that determines how the
# kNN is weighted while making the prediction
# define it in the param_grid dictionary as a list of weight parameters
# eg. weight_opt = ['uniform', 'distance']
# 'uniform' is the default one
# and add it to the param_grid like
# param_grid = dict(n_neighbors=k_range, weights=weight_opt)

# use best parameters to train your model

# shortcut: GridSearchCV automatically refits the best model using all the data
# syntax: grid.predict([data])

# Reducing computational expense

param_dist = dict(n_neighbors=k_range)

rand = RandomizedSearchCV(
    knn, param_dist, cv=10, n_iter=10, scoring='accuracy', random_state=5,
    return_train_score=False
)
rand.fit(X, y)

print(pd.DataFrame(rand.cv_results_)[
    ['mean_test_score', 'std_test_score', 'params']
    ])

print(rand.best_score_, rand.best_params_)
