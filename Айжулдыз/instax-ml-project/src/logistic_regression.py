def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, l2_reg=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features)
        self.intercept = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.coefficients) + self.intercept
            y_predicted = sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + (self.l2_reg * self.coefficients)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights
            self.coefficients -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.coefficients) + self.intercept
        y_predicted = sigmoid(linear_model)
        return np.where(y_predicted >= 0.5, 1, 0)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.coefficients) + self.intercept
        return sigmoid(linear_model)

    def score(self, X, y):
        y_predicted = self.predict(X)
        return np.mean(y_predicted == y)