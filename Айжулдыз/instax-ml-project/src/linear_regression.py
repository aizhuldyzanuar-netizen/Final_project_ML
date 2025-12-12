import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features)
        self.intercept = 0

        for _ in range(self.epochs):
            y_predicted = self.predict(X)
            # Calculate gradients
            coeff_gradient = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            intercept_gradient = (1/n_samples) * np.sum(y_predicted - y)

            # Update coefficients and intercept
            self.coefficients -= self.learning_rate * coeff_gradient
            self.intercept -= self.learning_rate * intercept_gradient

    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def plot_results(self, X, y):
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Data points')
        plt.plot(X, self.predict(X), color='red', label='Regression line')
        plt.title('Linear Regression Results')
        plt.xlabel('Features')
        plt.ylabel('Target')
        plt.legend()
        plt.show()

    def plot_mse_vs_epochs(self, mse_values):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(mse_values)), mse_values, color='green')
        plt.title('MSE vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.show()