import pandas as pd
import numpy as np

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def overview_missing_values(data):
    """Return an overview of missing values in the dataset."""
    return data.isnull().sum()

def feature_summary(data):
    """Return a summary of numeric and categorical features."""
    numeric_summary = data.describe()
    categorical_summary = data.describe(include=['object'])
    return numeric_summary, categorical_summary

def correlation_matrix(data):
    """Return the correlation matrix of the numeric features."""
    return data.corr()

def create_new_features(data):
    """Create new features such as month, day, season, revenue, and profit."""
    data['transaction_date'] = pd.to_datetime(data['transaction_date'])
    data['month'] = data['transaction_date'].dt.month
    data['day'] = data['transaction_date'].dt.day
    data['season'] = data['transaction_date'].dt.month % 12 // 3 + 1  # 1: Winter, 2: Spring, 3: Summer, 4: Fall
    data['revenue'] = data['quantity'] * data['price_per_unit']
    data['profit'] = data['revenue'] - data['cost']
    return data

def handle_missing_values(data):
    """Handle missing values in the dataset."""
    # Example: Fill missing values with the mean for numeric columns
    for column in data.select_dtypes(include=[np.number]).columns:
        data[column].fillna(data[column].mean(), inplace=True)
    
    # Example: Fill missing values for categorical columns with the mode
    for column in data.select_dtypes(include=[object]).columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    
    return data

def encode_categorical_features(data):
    """Encode categorical features using One-Hot Encoding."""
    data = pd.get_dummies(data, drop_first=True)
    return data

def split_data(data, target_column, test_size=0.2):
    """Split the dataset into features and target, and then into training and testing sets."""
    from sklearn.model_selection import train_test_split
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

def preprocess_data(file_path, target_column):
    """Load, clean, and preprocess the dataset."""
    data = load_data(file_path)
    data = handle_missing_values(data)
    data = create_new_features(data)
    data = encode_categorical_features(data)
    X_train, X_test, y_train, y_test = split_data(data, target_column)
    
    return X_train, X_test, y_train, y_test, data