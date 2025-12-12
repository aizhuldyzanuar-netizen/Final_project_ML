import unittest
import numpy as np
from src.linear_regression import LinearRegression
from src.logistic_regression import LogisticRegression

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.model = LinearRegression(learning_rate=0.01, epochs=1000)
        self.X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        self.y = np.array([1, 2, 2, 3])

    def test_fit(self):
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.coefficients)
        self.assertIsNotNone(self.model.intercept)

    def test_predict(self):
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(np.array([[3, 5]]))
        self.assertEqual(predictions.shape, (1,))

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.model = LogisticRegression(learning_rate=0.01, epochs=1000)
        self.X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
        self.y = np.array([0, 1, 1, 0])

    def test_fit(self):
        self.model.fit(self.X, self.y)
        self.assertIsNotNone(self.model.weights)

    def test_predict(self):
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(np.array([[0.5, 0.5]]))
        self.assertIn(predictions, [0, 1])

if __name__ == '__main__':
    unittest.main()