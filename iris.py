from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

i = load_iris()

X = i.data
y = i.target

knn = KNeighborsClassifier(n_neighbors=10)
logreg = LogisticRegression(multi_class='auto', solver='lbfgs')

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.4,
                                                    random_state=4)

knn.fit(X_train, y_train)
logreg.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
y_pred_logi = logreg.predict(X_test)

print('kNN accuracy:', metrics.accuracy_score(y_test, y_pred_knn))
print('Logistic regression accuracy:', metrics.accuracy_score(y_test,
                                                              y_pred_logi))

k_range = range(1, 26)
scores = []
for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(k_range, scores)
plt.xlabel('K')
plt.ylabel('Accuracy')
# plt.show()
# we find accuracy is highest from K = 6 to 16
# so we choose k = 11 and retrain the classifier with the whole data

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X, y)
