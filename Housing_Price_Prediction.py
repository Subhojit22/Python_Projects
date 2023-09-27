# Prediction of pricing of houses using ML and visualization of the data.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

housing = sklearn.datasets.fetch_california_housing()    # Here we are importing our required dataset into an array.

# print(housing)               #BY Activating these line of codes 
# print(housing.keys())        #you can see various attributes of your dataset 
# print(housing.data)
# print(housing.target)
# print(housing.frame)
# print(housing.target_names)
# print(housing.feature_names)
# print(housing.DESCR)

# Creating a structured dataframe using pandas
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Including the target column with the dataframe for understanding
df['PRICE'] = housing.target
print(df.head())
print(df.shape)
correlation = df.corr()

# Creating a heatmap of the dataframe
plt.figure(figsize=(12, 12))
sns.heatmap(correlation, cbar=True, square=True, fmt='.2f', annot=True, annot_kws={'size': 10}, cmap='Reds')
plt.show()

# Dropping the target column to create target set
X = df.drop(['PRICE'], axis=1)
Y = df['PRICE']
# print(X)        #Activate this to see X and Y
# print(Y)

# Splitting the dataset into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)

# Loading the model
model1 = XGBRegressor()
#model2 = LinearRegression()      # You can use different models according to the accuracy of the model.
#model3 = SVR()                   # Here I've used XGBRegresor because it was showing more accuracy than other ones
#model4 = RandomForestRegressor()

# Training the model
model1.fit(X_train, Y_train)
#model2.fit(X_train, Y_train)
#model3.fit(X_train, Y_train)
#model4.fit(X_train, Y_train)

# Storing prediction given by the model into different arrays.
Y_pred1 = model1.predict(X_test)
# Y_pred2 = model2.predict(X_test)    # 
# Y_pred3 = model3.predict(X_test)
# Y_pred4 = model4.predict(X_test)

# Comparing between the Actual vs Predicted data.
df1 = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred1})
# df2 = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred2})
# df3 = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred3})
# df4 = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred4})
print("This is XGBRegressor Model ML Output")
print(df1)
r2_score1 = metrics.r2_score(Y_test, Y_pred1)
print("R2 Score of this model:", r2_score1)
# print("This is LinerRegression Model ML Output")
# print(df2)
# r2_score2 = metrics.r2_score(Y_test, Y_pred2)
# print("R2 Score of this model:", r2_score2)
# print("This is an SVM Regressor Model ML Output")
# print(df3)
# r2_score3 = metrics.r2_score(Y_test, Y_pred3)
# print("R2 Score of this model:", r2_score3)
# print("This is a Random Forrest Regressor Model ML Output")
# print(df4)
# r2_score4 = metrics.r2_score(Y_test, Y_pred4)
# print("R2 Score of this model:", r2_score4)

# If you activate and run these block of codes you will see that XGBRegressor has the best R2 value among others so, we will use XGBRegressor.

# In this section we will test our prediction with testing data and calculate R2 score to
# measure model accuracy.
mean_abs_error = metrics.mean_absolute_error(Y_test, Y_pred1)
mean_sq_error = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred1))
print("Mean absolute error : ", mean_abs_error)          # More the error is closer to 0 more accurate the models are.
print("Mean Square root error : ", mean_sq_error)        # So try to use different volumes of testing and training data or different models to optimize the findings.

# Visualization using ScatterPlot
plt.scatter(Y_test, Y_pred1)
plt.xlabel("Actual Price ")
plt.ylabel("Predicted Price ")
plt.show()

# Visulaization using Pairplot
sns.pairplot(df, x_vars=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population'], y_vars='PRICE', height=9, aspect=0.5, kind='reg')    # You can use other feature names also.
plt.show()

# Using different visualization tools like box plot or KDE you can visualize different kinds of data. Go have fun playing with those. 

# Create a DataFrame with features for a single house
new_data = pd.DataFrame({
    'MedInc': [6.5456],
    'HouseAge': [57.0],
    'AveRooms': [5.23669],
    'AveBedrms': [1.73256],
    'Population': [500.0],
    'AveOccup': [2.7569],
    'Latitude': [37.78],
    'Longitude': [-122.23]
})

# Make a prediction for the single house
predicted_price = model1.predict(new_data)
print("Predicted value for given values is:",float(predicted_price))
# 'predicted_price' now contains the predicted price for the single house
