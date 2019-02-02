from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy

cancer = load_breast_cancer()

data = numpy.c_[cancer.data, cancer.target]
columns = numpy.append(cancer.feature_names, ["target"])

data = pd.DataFrame(data, columns=columns)

logreg = LogisticRegression()

X = data[data.columns[:-1]]
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

logreg.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)

print(metrics.accuracy_score(y_pred_logreg, y_test))
print(metrics.f1_score(y_test, y_pred_logreg))
