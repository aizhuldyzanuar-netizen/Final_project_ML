# 🎓 Machine Learning Project

## Project Overview

This project demonstrates the implementation of machine learning algorithms from scratch, including linear and logistic regression with gradient descent, as well as comparative analysis of various classification models.

**Project Participants:**
- Айжулдыз
- Жанерке

**Data Source:**
Instax sales transaction dataset (instax_sales_transaction_data.csv)

---

## Project Structure

```
instax-ml-project/
├── data/
│   └── instax_sales_transaction_data[1].csv
├── notebooks/
│   └── notebook.ipynb          # Main notebook with demonstration
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Data preprocessing
│   ├── linear_regression.py     # Linear regression (from scratch)
│   ├── logistic_regression.py   # Logistic regression (from scratch)
│   ├── classification_models.py # Decision Tree and Random Forest (sklearn)
│   ├── metrics.py               # Metrics module
│   ├── widgets_interface.py     # Interactive interface
│   ├── experiments.py           # Experiments
│   └── utils.py                 # Utility functions
├── tests/
│   └── test_models.py           # Model tests
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## Installation and Setup

### Local Installation

1. **Clone the repository or download the project**

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
# or
venv\Scripts\activate  # For Windows
```

3. **Install dependencies:**
```bash
cd Айжулдыз/instax-ml-project
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

5. **Open the notebook:**
   - Navigate to the `notebooks/` folder
   - Open `notebook.ipynb`
   - Run all cells sequentially

### Cloud Setup (Google Colab, Kaggle, etc.)

1. Upload project files to cloud storage
2. Install dependencies in the first cell:
```python
!pip install numpy pandas matplotlib seaborn scikit-learn ipywidgets scipy
```
3. Upload data and run the notebook

---

## Dependencies

The project requires the following Python packages (listed in `requirements.txt`):

- **NumPy** - for numerical computations and algorithm implementation
- **Pandas** - for data manipulation
- **Matplotlib** - for visualization
- **Seaborn** - for enhanced visualization
- **Scikit-learn** - for additional models and metrics
- **Ipywidgets** - for interactive interface
- **Jupyter** - for notebook work
- **SciPy** - for statistical functions (confidence intervals)

---

## Main Project Components

### 1. Linear Regression (Implementation from Scratch)

**File:** `src/linear_regression.py`

**Features:**
- ✅ Gradient descent (batch and mini-batch)
- ✅ Loss tracking by epochs
- ✅ Learning rate impact visualization
- ✅ Confidence intervals
- ✅ Scatter plot with regression line

**Gradient Formulas:**
- For coefficients: `dL/dw = (1/n) × X^T × (y_pred - y)`
- For intercept: `dL/db = (1/n) × Σ(y_pred - y)`
- Loss function (MSE): `L = (1/n) × Σ(y_pred - y)²`

**Usage:**
```python
from src.linear_regression import LinearRegression

model = LinearRegression(learning_rate=0.01, epochs=500, batch_size=32)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.plot_loss_history()
model.plot_regression_with_confidence(X_test, y_test)
```

### 2. Logistic Regression (Implementation from Scratch)

**File:** `src/logistic_regression.py`

**Features:**
- ✅ Sigmoid and log-loss function
- ✅ Gradient descent with L2 regularization
- ✅ Mini-batch training
- ✅ Loss tracking

**Formulas:**
- Sigmoid: `σ(z) = 1 / (1 + exp(-z))`
- Log-loss function: `L = -mean(y × log(σ(z)) + (1-y) × log(1-σ(z))) + (λ/2) × ||w||²`
- Gradients:
  - `dL/dw = (1/n) × X^T × (y_pred - y) + λ × w`
  - `dL/db = (1/n) × Σ(y_pred - y)`

**Usage:**
```python
from src.logistic_regression import LogisticRegression

model = LogisticRegression(learning_rate=0.01, epochs=500, batch_size=32, l2_reg=0.01)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
model.plot_loss_history()
```

### 3. Classification Models

**File:** `src/classification_models.py`

**Available Models:**
- **Decision Tree** (sklearn) - simple for explanation and visualization
- **Random Forest** (sklearn) - for improved quality

**Usage:**
```python
from src.classification_models import DecisionTreeModel, RandomForestModel

# Decision Tree
dt = DecisionTreeModel(max_depth=10)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# Random Forest
rf = RandomForestModel(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

### 4. Metrics

**File:** `src/metrics.py`

**Available Metrics:**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC AUC (for binary classification)

**Usage:**
```python
from src.metrics import (
    calculate_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    print_metrics_summary,
    compare_models
)

metrics = calculate_classification_metrics(y_true, y_pred, y_pred_proba)
print_metrics_summary(metrics, "Model Name")
plot_confusion_matrix(y_true, y_pred)
plot_roc_curve(y_true, y_pred_proba)
```

### 5. Interactive Interface

**File:** `src/widgets_interface.py`

**Capabilities:**
- Data loading via widget
- Model selection (Logistic Regression, Decision Tree, Random Forest)
- Parameter configuration:
  - Learning rate
  - Epochs
  - Batch size
  - Max depth (for trees)
  - N estimators (for Random Forest)
- Display of metrics and plots

**Usage:**
```python
from src.widgets_interface import create_interactive_interface

create_interactive_interface()
```

---

## Notebook Contents

The notebook `notebooks/notebook.ipynb` contains:

1. **Title Page** - information about participants and project
2. **Data Loading and Preprocessing**
3. **Exploratory Data Analysis (EDA)**
4. **Linear Regression** - full demonstration with plots
5. **Logistic Regression** - training and evaluation
6. **Classification** - comparison of three models
7. **Experiments and Metrics** - parameter impact analysis
8. **Interactive Interface** - widgets for experiments

---

## Technical Implementation Details

### Gradient Descent

All gradient descent algorithms are implemented **manually** using NumPy:
- ✅ Batch gradient descent (entire dataset)
- ✅ Mini-batch gradient descent (configurable batch size)
- ✅ Gradient formulas presented in code comments

### Optimizers

- Implemented SGD (Stochastic Gradient Descent) and mini-batch
- L2 regularization for logistic regression
- Momentum and Adam can be added optionally

### Documentation

- All functions have docstrings
- Gradient formulas described in comments
- Usage examples in the notebook

---

## Running Tests

```bash
python -m pytest tests/test_models.py
```

or

```bash
python tests/test_models.py
```

---

## Project Requirements Checklist

✅ **Linear Regression:**
- [x] Gradient descent implementation (batch/mini-batch)
- [x] Loss function (MSE)
- [x] Loss plot by epochs
- [x] Learning rate impact
- [x] Found coefficients and intercept
- [x] Scatter plot with regression line and confidence interval

✅ **Logistic Regression:**
- [x] Log-loss function and sigmoid
- [x] Gradient descent training
- [x] L2 regularization

✅ **Classification:**
- [x] Binary classification
- [x] Logistic Regression (custom implementation)
- [x] Decision Tree (sklearn)
- [x] Random Forest (sklearn)

✅ **Experiments and Metrics:**
- [x] Accuracy, Precision, Recall, F1-score
- [x] Confusion Matrix
- [x] ROC AUC (for binary classification)
- [x] Model comparison
- [x] Experiments with learning rate, epochs, batch size

✅ **Interface:**
- [x] Interactive widgets
- [x] Data loading
- [x] Model selection
- [x] Gradient descent parameters
- [x] Display of plots and metrics
- [x] Title page

---

## Troubleshooting

### Issue: Modules not found
**Solution:** Make sure you're in the correct directory and the path to `src` is added to `sys.path`. The notebook automatically handles this, but if issues persist, check that you're running from the `notebooks/` directory.

### Issue: Data not loading
**Solution:** Check the data file path in the notebook. Default: `../data/instax_sales_transaction_data[1].csv`

### Issue: Widgets not displaying
**Solution:** Make sure `ipywidgets` is installed and Jupyter extensions are enabled:
```bash
jupyter nbextension enable --py widgetsnbextension
```

---

## License

This project is created for educational purposes.

---

## Contact

For questions and suggestions, please create an issue in the repository.

---

**Good luck learning machine learning! 🚀**
