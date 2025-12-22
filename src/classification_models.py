"""
Модуль с дополнительными моделями классификации.
Используется sklearn для Decision Tree и Random Forest.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

class DecisionTreeModel:
    """
    Обертка для Decision Tree из sklearn.
    Выбрана как простая для объяснения и визуализации модель.
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        """
        Параметры:
        ----------
        max_depth : int или None
            Максимальная глубина дерева
        min_samples_split : int
            Минимальное количество образцов для разделения узла
        min_samples_leaf : int
            Минимальное количество образцов в листе
        random_state : int
            Seed для воспроизводимости
        """
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.feature_importances_ = None
        
    def fit(self, X, y):
        """Обучение модели."""
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        return self
    
    def predict(self, X):
        """Предсказание классов."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Предсказание вероятностей."""
        return self.model.predict_proba(X)[:, 1]  # Вероятность класса 1
    
    def score(self, X, y):
        """Точность модели."""
        return self.model.score(X, y)
    
    def get_feature_importances(self):
        """Возвращает важность признаков."""
        return self.feature_importances_

class RandomForestModel:
    """
    Обертка для Random Forest из sklearn.
    Выбрана для повышения качества классификации.
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, random_state=42):
        """
        Параметры:
        ----------
        n_estimators : int
            Количество деревьев в ансамбле
        max_depth : int или None
            Максимальная глубина деревьев
        min_samples_split : int
            Минимальное количество образцов для разделения узла
        min_samples_leaf : int
            Минимальное количество образцов в листе
        random_state : int
            Seed для воспроизводимости
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.feature_importances_ = None
        
    def fit(self, X, y):
        """Обучение модели."""
        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        return self
    
    def predict(self, X):
        """Предсказание классов."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Предсказание вероятностей."""
        return self.model.predict_proba(X)[:, 1]  # Вероятность класса 1
    
    def score(self, X, y):
        """Точность модели."""
        return self.model.score(X, y)
    
    def get_feature_importances(self):
        """Возвращает важность признаков."""
        return self.feature_importances_

