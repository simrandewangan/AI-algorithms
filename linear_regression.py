import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[5], [8], [10], [12]])
y = np.array([2, 3.8, 4.5, 6])

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict
X_new = np.array([[5]])
y_pred = model.predict(X_new)

# Plot
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.scatter(X_new, y_pred, color='red', marker='x')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

print(f'Predicted value for X={X_new[0][0]}: {y_pred[0]}')