import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = "http://bit.ly/w-data"
df = pd.read_csv(data)
print(df.head())
df.plot(x="Hours", y="Scores", style="o")
plt.xlabel("Hours Studied-->")
plt.ylabel("Score Obtained-->")
plt.title("Study Hours vs Score")
plt.show()
X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3, random_state=0)
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)
line = regressor.coef_ * X + regressor.intercept_
plt.scatter(X, Y)
plt.plot(X, line)
plt.show()
Y_pred = regressor.predict(X_Test)
prediction = pd.DataFrame({'Actual': Y_Test, 'Predicted': Y_pred})
print(prediction)
print(metrics.mean_absolute_error(Y_Test, Y_pred))
print("Enter the Study Hours for prediction:")
hours = float(input()) # Enter a value for prediction
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
