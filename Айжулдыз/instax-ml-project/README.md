# instax-ml-project

## Overview
The Instax ML Project is a comprehensive machine learning project that utilizes sales transaction data to build predictive models. The project includes data preprocessing, linear regression, logistic regression, and model evaluation, all implemented from scratch using Python and NumPy.

## Project Structure
```
instax-ml-project
├── data
│   └── instax_sales_transaction_data.csv
├── notebooks
│   └── notebook.ipynb
├── src
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── experiments.py
│   ├── widgets_interface.py
│   └── utils.py
├── tests
│   └── test_models.py
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## Installation
To set up the project, clone the repository and install the required packages. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage
1. **Data Loading and Preprocessing**: Use the `preprocessing.py` module to load and clean the dataset. This includes handling missing values, converting dates, and encoding categorical features.
2. **Model Training**: The `linear_regression.py` and `logistic_regression.py` modules contain implementations of linear and logistic regression, respectively. You can train these models using the functions provided.
3. **Experiments**: The `experiments.py` module allows you to run experiments by modifying parameters such as learning rate and number of epochs.
4. **Interactive Interface**: The `widgets_interface.py` module provides an interactive interface using ipywidgets for model selection and parameter tuning.
5. **Testing**: The `test_models.py` file contains unit tests to ensure the correctness of the implemented models.

## Requirements
The project requires the following Python packages:
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Ipywidgets

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.