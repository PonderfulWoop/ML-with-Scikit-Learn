import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.linear_model import LinearRegression

# to set a column of a table as its index, use index_col= 'index of the column'
# in the read_csv method

data = pd.read_csv('Advertising.csv', index_col=0)
print(data.head())

sns.pairplot(
    data,
    x_vars=['TV', 'Radio', 'Newspaper'],
    y_vars='Sales', height=4, aspect=0.4
)

# plt.show()

features_col = ['TV', 'Radio', 'Newspaper']

X = data[features_col]

y = data['Sales']

# or can also use, y = data.Sales if there are no spaces in bet. the col name

# X_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
# default spit is 75% train and 25% test

linear = LinearRegression()

scores = cross_val_score(linear, X, y, cv=10, scoring='neg_mean_squared_error')
print(scores)
# linear.fit(X_train, y_train)

# linear.coef_ now has the coefficients of the features, linear.intercept_ now
# has the intercept; using zip function to pair the coeff.

# pair = list(zip(features_col, linear.coef_))

# y_pred = linear.predict(x_test)

# error = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
# result in error = 1.40465142303

# feature selection
# delete features to decrease the error by changing X and fitting it again and
# predicting it.

# fixing -ve values
mse_scores = -scores

# convert mse to rmse
rmse_scores = np.sqrt(mse_scores)

print(rmse_scores.mean())

# 10 fold cross-validation with the two features (excluding newspaper)
features_col = ['TV', 'Radio']
X = data[features_col]
print(np.sqrt(-cross_val_score(
    linear, X, y, cv=10, scoring='neg_mean_squared_error')).mean())

# note that we get a mean of 1.68 which is better than 1.69,
# so we conclude that model excluding 'newspaper' is a better model
