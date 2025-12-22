"""
Interactive interface with widgets for training and evaluating models.
"""
from ipywidgets import (
    FileUpload, Dropdown, FloatSlider, IntSlider, Button, Output,
    VBox, HBox, HTML, Tab, Accordion
)
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io

try:
    from src.linear_regression import LinearRegression
    from src.logistic_regression import LogisticRegression
    from src.classification_models import DecisionTreeModel, RandomForestModel
    from src.metrics import (
        calculate_classification_metrics, plot_confusion_matrix,
        plot_roc_curve, print_metrics_summary, compare_models
    )
    from src.preprocessing import load_data, handle_missing_values, create_new_features, encode_categorical_features
except ImportError:
    # For cases when modules are imported directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.linear_regression import LinearRegression
    from src.logistic_regression import LogisticRegression
    from src.classification_models import DecisionTreeModel, RandomForestModel
    from src.metrics import (
        calculate_classification_metrics, plot_confusion_matrix,
        plot_roc_curve, print_metrics_summary, compare_models
    )
    from src.preprocessing import load_data, handle_missing_values, create_new_features, encode_categorical_features

class ModelTrainer:
    """Class for managing model training through widgets."""
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_reg = None
        self.y_test_reg = None
        self.trained_models = {}
        self.metrics_results = {}
        
    def load_and_preprocess_data(self, file_path=None, file_content=None):
        """Load and preprocess data."""
        if file_content is not None:
            # Load from uploaded file
            self.data = pd.read_csv(io.BytesIO(file_content))
        elif file_path is not None:
            # Load from path
            self.data = load_data(file_path)
        else:
            raise ValueError("Must specify file_path or file_content")
        
        # Preprocessing
        self.data = handle_missing_values(self.data)
        try:
            self.data = create_new_features(self.data)
        except:
            pass  # Skip if create_new_features fails (e.g., no date column)
        
        # Determine target variable for regression
        if 'total_revenue' in self.data.columns:
            target_reg = 'total_revenue'
        elif 'revenue' in self.data.columns:
            target_reg = 'revenue'
        else:
            # Use first numeric column
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            target_reg = numeric_cols[0] if len(numeric_cols) > 0 else None
        
        # Encode categorical features
        self.data = encode_categorical_features(self.data)
        
        # Split into features and target variable
        if target_reg and target_reg in self.data.columns:
            X = self.data.drop(columns=[target_reg])
            y_reg = self.data[target_reg]
            
            # Create classification task (binary)
            median_val = np.median(y_reg.values) if hasattr(y_reg, 'values') else np.median(y_reg)
            y_class = (y_reg > median_val).astype(int)
            
            # Split into train/test
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train_reg, self.y_test_reg = train_test_split(
                X, y_reg, test_size=0.2, random_state=42
            )
            _, _, self.y_train, self.y_test = train_test_split(
                X, y_class, test_size=0.2, random_state=42
            )
            self.X_train = self.X_train.values
            self.X_test = self.X_test.values
        else:
            raise ValueError("Failed to determine target variable")

def create_interactive_interface():
    """
    Create a fully functional interactive interface.
    """
    try:
        trainer = ModelTrainer()
        
        # Title
        title = HTML("<h1 style='text-align: center; color: #2c3e50;'>🤖 Interactive ML Models Demonstration</h1>")
        
        # Data upload widgets
        upload_widget = FileUpload(
            accept='.csv',
            multiple=False,
            description='Upload Data'
        )
        
        # Task type selection
        task_type = Dropdown(
            options=['Classification', 'Regression'],
            value='Classification',
            description='Task:',
            style={'description_width': 'initial'}
        )
        
        # Model selection
        model_dropdown = Dropdown(
            options=['Logistic Regression', 'Decision Tree', 'Random Forest'],
            value='Logistic Regression',
            description='Model:',
            style={'description_width': 'initial'}
        )

        # Parameters for gradient descent
        learning_rate_slider = FloatSlider(
            value=0.01,
            min=0.0001,
            max=1.0,
            step=0.001,
            description='Learning Rate:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        epochs_slider = IntSlider(
            value=100,
            min=10,
            max=1000,
            step=10,
            description='Epochs:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        batch_size_slider = IntSlider(
            value=32,
            min=1,
            max=256,
            step=1,
            description='Batch Size:',
            style={'description_width': 'initial'},
            continuous_update=False
        )

        # Parameters for Decision Tree
        max_depth_slider = IntSlider(
            value=10,
            min=1,
            max=50,
            step=1,
            description='Max Depth:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
        
        # Parameters for Random Forest
        n_estimators_slider = IntSlider(
            value=100,
            min=10,
            max=500,
            step=10,
            description='N Estimators:',
            style={'description_width': 'initial'},
            continuous_update=False
        )
    
        # Buttons
        load_button = Button(
            description='Load Data',
            button_style='info',
            icon='upload'
        )
        
        train_button = Button(
            description='Train Model',
            button_style='success',
            icon='play'
        )
        
        show_metrics_button = Button(
            description='Show Metrics',
            button_style='warning',
            icon='chart-bar'
        )
        
        show_loss_button = Button(
            description='Show Loss Plot',
            button_style='info',
            icon='line-chart'
        )
        
        compare_models_button = Button(
            description='Compare Models',
            button_style='primary',
            icon='balance-scale'
        )
    
        # Output area
        output_area = Output()
        
        def on_load_button_clicked(b):
            with output_area:
                clear_output(wait=True)
                if len(upload_widget.value) > 0:
                    file_name = list(upload_widget.value.keys())[0]
                    file_content = upload_widget.value[file_name]['content']
                    try:
                        trainer.load_and_preprocess_data(file_content=file_content)
                        print(f"✅ Data loaded successfully!")
                        print(f"Training set size: {trainer.X_train.shape}")
                        print(f"Test set size: {trainer.X_test.shape}")
                        print(f"Number of features: {trainer.X_train.shape[1]}")
                    except Exception as e:
                        print(f"❌ Error loading data: {e}")
                else:
                    print("⚠️ Please upload a CSV file")
    
        def on_train_button_clicked(b):
            with output_area:
                clear_output(wait=True)
                if trainer.X_train is None:
                    print("⚠️ Please load data first!")
                    return
                
                model_name = model_dropdown.value
                print(f"🔄 Training model: {model_name}...")
                
                try:
                    if model_name == 'Logistic Regression':
                        model = LogisticRegression(
                            learning_rate=learning_rate_slider.value,
                            epochs=epochs_slider.value,
                            batch_size=batch_size_slider.value if batch_size_slider.value < len(trainer.X_train) else None
                        )
                        model.fit(trainer.X_train, trainer.y_train)
                        trainer.trained_models[model_name] = model
                        
                        # Calculate metrics
                        y_pred = model.predict(trainer.X_test)
                        y_pred_proba = model.predict_proba(trainer.X_test)
                        metrics = calculate_classification_metrics(trainer.y_test, y_pred, y_pred_proba)
                        trainer.metrics_results[model_name] = metrics
                        
                        print(f"✅ Model trained!")
                        print_metrics_summary(metrics, model_name)
                        
                    elif model_name == 'Decision Tree':
                        model = DecisionTreeModel(max_depth=max_depth_slider.value)
                        model.fit(trainer.X_train, trainer.y_train)
                        trainer.trained_models[model_name] = model
                        
                        y_pred = model.predict(trainer.X_test)
                        y_pred_proba = model.predict_proba(trainer.X_test)
                        metrics = calculate_classification_metrics(trainer.y_test, y_pred, y_pred_proba)
                        trainer.metrics_results[model_name] = metrics
                        
                        print(f"✅ Model trained!")
                        print_metrics_summary(metrics, model_name)
                        
                    elif model_name == 'Random Forest':
                        model = RandomForestModel(
                            n_estimators=n_estimators_slider.value,
                            max_depth=max_depth_slider.value
                        )
                        model.fit(trainer.X_train, trainer.y_train)
                        trainer.trained_models[model_name] = model
                        
                        y_pred = model.predict(trainer.X_test)
                        y_pred_proba = model.predict_proba(trainer.X_test)
                        metrics = calculate_classification_metrics(trainer.y_test, y_pred, y_pred_proba)
                        trainer.metrics_results[model_name] = metrics
                        
                        print(f"✅ Model trained!")
                        print_metrics_summary(metrics, model_name)
                        
                except Exception as e:
                    print(f"❌ Error during training: {e}")
                    import traceback
                    traceback.print_exc()
    
        def on_show_metrics_button_clicked(b):
            with output_area:
                clear_output(wait=True)
                if not trainer.trained_models:
                    print("⚠️ Please train a model first!")
                    return
                
                model_name = model_dropdown.value
                if model_name not in trainer.trained_models:
                    print(f"⚠️ Model {model_name} is not trained yet!")
                    return
                
                model = trainer.trained_models[model_name]
                metrics = trainer.metrics_results[model_name]
                
                print_metrics_summary(metrics, model_name)
                
                # Confusion matrix
                y_pred = model.predict(trainer.X_test)
                plot_confusion_matrix(trainer.y_test, y_pred, 
                                    title=f'Confusion Matrix - {model_name}')
                
                # ROC curve (if binary classification)
                if len(np.unique(trainer.y_test)) == 2:
                    y_pred_proba = model.predict_proba(trainer.X_test)
                    plot_roc_curve(trainer.y_test, y_pred_proba,
                                 title=f'ROC Curve - {model_name}')
        
        def on_show_loss_button_clicked(b):
            with output_area:
                clear_output(wait=True)
                model_name = model_dropdown.value
                if model_name not in trainer.trained_models:
                    print(f"⚠️ Model {model_name} is not trained yet!")
                    return
                
                model = trainer.trained_models[model_name]
                
                if hasattr(model, 'losses') and model.losses:
                    model.plot_loss_history()
                else:
                    print("⚠️ Loss plot is only available for Logistic Regression")
        
        def on_compare_models_button_clicked(b):
            with output_area:
                clear_output(wait=True)
                if len(trainer.metrics_results) < 2:
                    print("⚠️ Train at least 2 models for comparison!")
                    return
                
                compare_models(trainer.metrics_results)
                print("\n📊 Comparison completed!")
    
        # Bind events
        load_button.on_click(on_load_button_clicked)
        train_button.on_click(on_train_button_clicked)
        show_metrics_button.on_click(on_show_metrics_button_clicked)
        show_loss_button.on_click(on_show_loss_button_clicked)
        compare_models_button.on_click(on_compare_models_button_clicked)
        
        # Group widgets
        data_section = VBox([
            HTML("<h3>📁 Data Loading</h3>"),
            upload_widget,
            load_button
        ])
        
        model_section = VBox([
            HTML("<h3>🎯 Model Selection and Parameters</h3>"),
            task_type,
            model_dropdown,
            learning_rate_slider,
            epochs_slider,
            batch_size_slider,
            max_depth_slider,
            n_estimators_slider
        ])
        
        actions_section = VBox([
            HTML("<h3>⚙️ Actions</h3>"),
            HBox([train_button, show_metrics_button]),
            HBox([show_loss_button, compare_models_button])
        ])
        
        # Display interface
        interface = VBox([
            title,
            Accordion(children=[
                data_section,
                model_section,
                actions_section
            ], titles=['Data Loading', 'Model Parameters', 'Actions']),
            output_area
        ])
        
        try:
            display(interface)
        except Exception as e:
            print(f"⚠️ Warning: Could not display interactive widgets: {e}")
            print("You can still use the models programmatically.")
            print("To fix this, try running: jupyter nbextension enable --py widgetsnbextension")
        
        return trainer
    
    except Exception as e:
        print(f"⚠️ Error creating interactive interface: {e}")
        print("Falling back to simple interface...")
        print("\nYou can use the models directly in code:")
        print("  from src.logistic_regression import LogisticRegression")
        print("  model = LogisticRegression()")
        print("  model.fit(X_train, y_train)")
        return None

def create_widgets():
    """Alias for backward compatibility."""
    return create_interactive_interface()