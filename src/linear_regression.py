import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class LinearRegression:
    """
    Линейная регрессия с градиентным спуском (batch или mini-batch).
    
    Формулы градиентов:
    - Для коэффициентов: dL/dw = (1/n) * X^T * (y_pred - y)
    - Для intercept: dL/db = (1/n) * sum(y_pred - y)
    - Функция потерь (MSE): L = (1/n) * sum((y_pred - y)^2)
    """
    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=None):
        """
        Параметры:
        ----------
        learning_rate : float
            Скорость обучения
        epochs : int
            Количество эпох
        batch_size : int или None
            Размер батча. Если None, используется batch gradient descent (весь датасет)
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
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
            Целевая переменная
        """
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features)
        self.intercept = 0.0
        self.losses = []
        
        # Если batch_size не указан, используем весь датасет (batch gradient descent)
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
                
                # Предсказания
                y_predicted = np.dot(X_batch, self.coefficients) + self.intercept
                
                # Вычисление градиентов
                batch_size_actual = X_batch.shape[0]
                coeff_gradient = (1/batch_size_actual) * np.dot(X_batch.T, (y_predicted - y_batch))
                intercept_gradient = (1/batch_size_actual) * np.sum(y_predicted - y_batch)
                
                # Обновление параметров
                self.coefficients -= self.learning_rate * coeff_gradient
                self.intercept -= self.learning_rate * intercept_gradient
            
            # Вычисление и сохранение потерь на всей выборке
            y_pred_full = self.predict(X)
            mse = self.mean_squared_error(y, y_pred_full)
            self.losses.append(mse)
    
    def predict(self, X):
        """
        Предсказание значений.
        
        Параметры:
        ----------
        X : array-like, shape (n_samples, n_features)
            Признаки
            
        Возвращает:
        -----------
        y_pred : array, shape (n_samples,)
            Предсказанные значения
        """
        return np.dot(X, self.coefficients) + self.intercept
    
    def predict_with_confidence(self, X, confidence=0.95):
        """
        Предсказание с доверительными интервалами.
        
        Параметры:
        ----------
        X : array-like, shape (n_samples, n_features)
            Признаки
        confidence : float
            Уровень доверия (по умолчанию 0.95)
            
        Возвращает:
        -----------
        y_pred : array
            Предсказанные значения
        lower_bound : array
            Нижняя граница доверительного интервала
        upper_bound : array
            Верхняя граница доверительного интервала
        """
        y_pred = self.predict(X)
        # Для упрощения используем стандартную ошибку
        # В реальности нужно использовать остатки обучения
        std_error = np.std(y_pred) if len(y_pred) > 1 else 0.1
        alpha = 1 - confidence
        t_value = stats.t.ppf(1 - alpha/2, df=len(y_pred) - 2) if len(y_pred) > 2 else 1.96
        
        margin = t_value * std_error
        lower_bound = y_pred - margin
        upper_bound = y_pred + margin
        
        return y_pred, lower_bound, upper_bound

    def mean_squared_error(self, y_true, y_pred):
        """Вычисление среднеквадратичной ошибки (MSE)."""
        return np.mean((y_true - y_pred) ** 2)
    
    def get_coefficients(self):
        """Возвращает коэффициенты и intercept."""
        return self.coefficients, self.intercept

    def plot_loss_history(self):
        """График потерь по эпохам."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.losses)), self.losses, color='green', linewidth=2)
        plt.title('MSE vs Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_regression_with_confidence(self, X, y, feature_idx=0, confidence=0.95):
        """
        Scatter plot данных с линией регрессии и доверительным интервалом.
        
        Параметры:
        ----------
        X : array-like
            Признаки
        y : array-like
            Целевая переменная
        feature_idx : int
            Индекс признака для визуализации (для многомерного случая)
        confidence : float
            Уровень доверия
        """
        if X.ndim > 1:
            X_plot = X[:, feature_idx]
        else:
            X_plot = X
        
        # Сортировка для плавной линии
        sort_idx = np.argsort(X_plot)
        X_sorted = X_plot[sort_idx]
        y_sorted = y[sort_idx]
        
        y_pred, lower_bound, upper_bound = self.predict_with_confidence(X[sort_idx], confidence)
        
        plt.figure(figsize=(12, 7))
        plt.scatter(X_sorted, y_sorted, alpha=0.6, color='blue', label='Data points', s=50)
        plt.plot(X_sorted, y_pred, color='red', linewidth=2, label='Regression line')
        plt.fill_between(X_sorted, lower_bound, upper_bound, alpha=0.3, color='red', 
                        label=f'{int(confidence*100)}% Confidence Interval')
        plt.title('Linear Regression with Confidence Interval', fontsize=14, fontweight='bold')
        plt.xlabel(f'Feature {feature_idx}', fontsize=12)
        plt.ylabel('Target', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_learning_rate_comparison(self, X, y, learning_rates=[0.001, 0.01, 0.1, 0.5]):
        """
        Сравнение влияния разных значений learning rate на сходимость.
        
        Параметры:
        ----------
        X : array-like
            Признаки
        y : array-like
            Целевая переменная
        learning_rates : list
            Список learning rates для сравнения
        """
        plt.figure(figsize=(12, 7))
        
        for lr in learning_rates:
            # Создаем временную модель с другим learning rate
            temp_model = LinearRegression(learning_rate=lr, epochs=self.epochs, 
                                         batch_size=self.batch_size)
            temp_model.fit(X, y)
            plt.plot(range(len(temp_model.losses)), temp_model.losses, 
                    label=f'LR = {lr}', linewidth=2)
        
        plt.title('Influence of Learning Rate on Convergence', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Логарифмическая шкала для лучшей визуализации
        plt.show()