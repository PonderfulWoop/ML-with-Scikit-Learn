import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import binarize

path = 'pima-indians-diabetes.data'
col_names = [
    'preganent', 'glucose', 'bp', 'skin', 'insulin', 'bmi',
    'pedigree', 'age', 'label'
]
pima = pd.read_csv(path, header=None, names=col_names)

print(pima.head())

# Question: Can we predict the diabetes status of a patient given
# their health measurements

# define X and y

feature_cols = ['preganent', 'insulin', 'bmi', 'age']
X = pima[feature_cols]
y = pima['label']

# splitting X and y

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))

# NULL accuracy: accuracy that could be achieved by always predicting
# the most frequent class

# Calculate NULL Accuracy:
null_acc = max(y_test.mean(), 1 - y_test.mean())  # works for bin. class. only

# for multi-class classification
null_acc = y_test.value_counts().head(1) / len(y_test)  # works for pd series

# Confusion Matrix: Table that describes the performance of a
# classification problem

# Example:
print(metrics.confusion_matrix(y_test, y_pred))

# PRINTS THE FOLLOWING as a list of lists:
#
#           |           |           |
#           | predicted | predicted |
#  n = 192  |     0     |     1     |
#
#  Actual: 0     118          12
#
#  Actual: 1     47           15
#
# Terminologies: True Positives (1,1), True Negatives (0,0),
#                False Positives(0,1), False Negatives(1,0)

confusion = metrics.confusion_matrix(y_test, y_pred)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

# Classification Accuracy:
acc = (TP + TN) / float(TP + TN + FP + FN)  # equal to metrics.accuracy_score()

# Classifiaction Error:
error = (FP + FN) / float(TP + TN + FP + FN)
# equal to 1-metrics.accuracy_score()

# Sensitivity: when the actual value is +ve, how
#              often is the prediction correct (also called, recall)
sens = TP/float(TP + FN)
# or, sens = metrics.recall_score(y_test, y_pred)

# Specificity: when the actual value is -ve, how often is the
#              prediction correct
spec = TN/float(TN + FP)

# False Positive rate: when the actual value is -ve,
#                      how often is the prediction correct
false_positive = FP/float(TN + FP)

# Precision: When a +ve is predicted, how often is the
#            prediction correct
prec = TP/float(TP + FP)
# or, prec = metrics.precision_score(y_test, y_pred_class)

###########################################################################

# Adjusting the classifiaction threshold

prob = logreg.predict_proba(X_test)[0:10, :]
# prints 10 rows and 2 cols where col:1 is the prob of a
# sample being classified as non-diabetic and col:2 is the prob of a
# sample being classified as diabetic.
# both values in a row add up to 1.

# NOTE: we have 2 cols as there are only 2 labels: 0 and 1 to classify into

# NOTE: Classifiaction thershold: the prob over which the algo classifies as 1

y_prob = logreg.predict_proba(X_test)[:, 1]  # prob of 1 (bcuz col 2)

# predict diabetes if the predicted probabilty is greater than 0.3
y_prob_class = binarize([y_prob], 0.3)[0]

print(y_prob_class)
