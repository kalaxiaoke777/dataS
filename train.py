import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("./ML_data/world-happiness-report-2017.csv")

train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_name = "Economy..GDP.per.Capita."
output_name = "Happiness.Score"


x_train = train_data[[input_name]].values
y_train = train_data[[output_name]].values

x_test = test_data[input_name]
y_test = test_data[output_name]

plt.scatter(x_train, y_train, label="Training data")
plt.scatter(x_test, y_test, label="test data")
plt.ylabel(output_name)
plt.xlabel(input_name)
plt.legend()
plt.show()

#
from projects.linear import linearRegression

#
linear = linearRegression(x_train, y_train, 0.01, 1)

ceta, cost = linear.train(0.001, 50000)


print(ceta)
