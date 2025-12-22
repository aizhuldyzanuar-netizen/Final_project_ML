import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    Сигмоидная функция активации.
    
    Формула: σ(z) = 1 / (1 + exp(-z))
    """
    # Защита от переполнения
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def log_loss(y_true, y_pred):
    """
    Логарифмическая функция потерь (binary cross-entropy).
    
    Формула: L = -mean(y * log(y_pred) + (1-y) * log(1-y_pred))
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class LogisticRegression:
    """
    Логистическая регрессия с градиентным спуском (batch или mini-batch).
    
    Формулы градиентов:
    - Для коэффициентов: dL/dw = (1/n) * X^T * (y_pred - y) + λ * w (L2 регуляризация)
    - Для intercept: dL/db = (1/n) * sum(y_pred - y)
    - Функция потерь: L = -mean(y * log(σ(z)) + (1-y) * log(1-σ(z))) + (λ/2) * ||w||^2
    где z = X * w + b, σ(z) - сигмоида
    """
    def __init__(self, learning_rate=0.01, epochs=1000, l2_reg=0.0, batch_size=None):
        """
        Параметры:
        ----------
        learning_rate : float
            Скорость обучения
        epochs : int
            Количество эпох
        l2_reg : float
            Коэффициент L2 регуляризации
        batch_size : int или None
            Размер батча. Если None, используется batch gradient descent
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.coefficients = None
        self.intercept = None
        self.losses = []  # История потерь по эпохам
        
    def fit(self, X, y):
        """
        Обучение модели градиентным спуском.
        
        Параметры:
        ----------
        X : array-like, shape (n_samples, n_features)
            Признаки
        y : array-like, shape (n_samples,)
            Целевая переменная (0 или 1)
        """
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features)
        self.intercept = 0.0
        self.losses = []
        
        # Если batch_size не указан, используем весь датасет
        if self.batch_size is None:
            self.batch_size = n_samples
        
        for epoch in range(self.epochs):
            # Mini-batch градиентный спуск
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Линейная комбинация
                linear_model = np.dot(X_batch, self.coefficients) + self.intercept
                # Применение сигмоиды
                y_predicted = sigmoid(linear_model)
                
                # Вычисление градиентов
                batch_size_actual = X_batch.shape[0]
                # Градиент для коэффициентов с L2 регуляризацией
                dw = (1 / batch_size_actual) * np.dot(X_batch.T, (y_predicted - y_batch)) + (self.l2_reg * self.coefficients)
                # Градиент для intercept
                db = (1 / batch_size_actual) * np.sum(y_predicted - y_batch)
                
                # Обновление параметров
                self.coefficients -= self.learning_rate * dw
                self.intercept -= self.learning_rate * db
            
            # Вычисление и сохранение потерь на всей выборке
            linear_model_full = np.dot(X, self.coefficients) + self.intercept
            y_pred_full = sigmoid(linear_model_full)
            loss = log_loss(y, y_pred_full)
            # Добавляем L2 регуляризацию к функции потерь
            if self.l2_reg > 0:
                loss += (self.l2_reg / 2) * np.sum(self.coefficients ** 2)
            self.losses.append(loss)

    def predict(self, X):
        """
        Предсказание классов (0 или 1).
        
        Параметры:
        ----------
        X : array-like, shape (n_samples, n_features)
            Признаки
            
        Возвращает:
        -----------
        y_pred : array, shape (n_samples,)
            Предсказанные классы
        """
        linear_model = np.dot(X, self.coefficients) + self.intercept
        y_predicted = sigmoid(linear_model)
        return np.where(y_predicted >= 0.5, 1, 0)

    def predict_proba(self, X):
        """
        Предсказание вероятностей принадлежности к классу 1.
        
        Параметры:
        ----------
        X : array-like, shape (n_samples, n_features)
            Признаки
            
        Возвращает:
        -----------
        probabilities : array, shape (n_samples,)
            Вероятности принадлежности к классу 1
        """
        linear_model = np.dot(X, self.coefficients) + self.intercept
        return sigmoid(linear_model)
    
    def plot_loss_history(self):
        """График потерь по эпохам."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.losses)), self.losses, color='purple', linewidth=2)
        plt.title('Log Loss vs Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Log Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_coefficients(self):
        """Возвращает коэффициенты и intercept."""
        return self.coefficients, self.intercept

    def score(self, X, y):
        """
        Вычисление точности (accuracy).
        
        Параметры:
        ----------
        X : array-like
            Признаки
        y : array-like
            Истинные метки
            
        Возвращает:
        -----------
        accuracy : float
            Точность модели
        """
        y_predicted = self.predict(X)
        return np.mean(y_predicted == y)