from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import metrics

# Review of the model evaluaation procedures
# Motivation: A way to choose between ML models
# Initial Idea: Train and test on the same data
# Alternative Idea:
#      1. split the datasent into two pieces, so that the model
#         can be trained and tested on different data
#      2. testing accuracy is a better estimate than training accuracy
#         of out-of-sample performance
#      3. But, provides high variance estimate since changng observations
#         happen to be in the testing set can significantly change testing acc.

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

# Here, the accuracy varies with change in random_state; ie, high variance
# exists therefore, to correct this, we have k-Fold cross-Vaidation

# Steps for K-fold cross - validation
# 1. Split the dataset into K equal partitions (or, 'folds')
# 2. Use fold 1 as the testing set and the union of the othe folds as the
#    training set
# 3. calculate testing accuracy
# 4. repeat 2 and 3 K times, using a different fold as the testing set each
#    time
# 5. use the average testing accuracy as the estimate of the out-of-sample
#    accuracy.

# IMPLEMENTATION given below

scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)
print(scores.mean())

# search for optimal K for kNN

k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)

plt.plot(k_range, k_scores)
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.show()
