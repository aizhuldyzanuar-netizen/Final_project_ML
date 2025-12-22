"""
Модуль для вычисления метрик классификации и регрессии.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)

def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Вычисление всех метрик классификации.
    
    Параметры:
    ----------
    y_true : array-like
        Истинные метки
    y_pred : array-like
        Предсказанные метки
    y_pred_proba : array-like, optional
        Предсказанные вероятности (для ROC AUC)
        
    Возвращает:
    -----------
    metrics : dict
        Словарь с метриками
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # ROC AUC только для бинарной классификации
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix'):
    """
    Визуализация матрицы ошибок.
    
    Параметры:
    ----------
    y_true : array-like
        Истинные метки
    y_pred : array-like
        Предсказанные метки
    class_names : list, optional
        Названия классов
    title : str
        Заголовок графика
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return cm

def plot_roc_curve(y_true, y_pred_proba, title='ROC Curve'):
    """
    Визуализация ROC кривой (только для бинарной классификации).
    
    Параметры:
    ----------
    y_true : array-like
        Истинные метки
    y_pred_proba : array-like
        Предсказанные вероятности класса 1
    title : str
        Заголовок графика
    """
    if len(np.unique(y_true)) != 2:
        print("ROC кривая доступна только для бинарной классификации")
        return None
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return fpr, tpr, roc_auc

def print_metrics_summary(metrics, model_name='Model'):
    """
    Вывод сводки метрик.
    
    Параметры:
    ----------
    metrics : dict
        Словарь с метриками
    model_name : str
        Название модели
    """
    print(f"\n{'='*50}")
    print(f"Metrics for {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"{'='*50}\n")

def compare_models(metrics_dict):
    """
    Сравнение метрик нескольких моделей.
    
    Параметры:
    ----------
    metrics_dict : dict
        Словарь вида {model_name: metrics_dict}
    """
    models = list(metrics_dict.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Создаем данные для сравнения
    comparison_data = {metric: [metrics_dict[model][metric] for model in models] 
                      for metric in metric_names}
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metric_names):
        offset = (i - 1.5) * width
        ax.bar(x + offset, comparison_data[metric], width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    plt.show()

