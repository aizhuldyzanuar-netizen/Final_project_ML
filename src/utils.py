def plot_histogram(data, column, title='Histogram', xlabel='Value', ylabel='Frequency'):
    """Plot a histogram for a given column in the dataset."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=30, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def plot_boxplot(data, column, title='Boxplot', ylabel='Value'):
    """Plot a boxplot for a given column in the dataset."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data[column], vert=False)
    plt.title(title)
    plt.xlabel(ylabel)
    plt.grid(axis='x', alpha=0.75)
    plt.show()

def calculate_metrics(y_true, y_pred):
    """Calculate and return accuracy, precision, recall, and F1-score."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def plot_confusion_matrix(cm, classes):
    """Plot the confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()