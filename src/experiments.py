def run_experiments(model, X_train, y_train, X_val, y_val, learning_rates, epochs, batch_sizes):
    results = []
    
    for lr in learning_rates:
        for epoch in epochs:
            for batch_size in batch_sizes:
                # Train the model with the given parameters
                model.train(X_train, y_train, learning_rate=lr, epochs=epoch, batch_size=batch_size)
                
                # Evaluate the model
                metrics = model.evaluate(X_val, y_val)
                
                # Store the results
                results.append({
                    'learning_rate': lr,
                    'epochs': epoch,
                    'batch_size': batch_size,
                    'metrics': metrics
                })
    
    return results

def plot_metrics(results):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    # Plotting metrics vs learning rate
    plt.figure(figsize=(12, 6))
    for metric in df['metrics'][0].keys():
        plt.plot(df['learning_rate'], df['metrics'].apply(lambda x: x[metric]), label=metric)
    
    plt.title('Metrics vs Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid()
    plt.show()

    # Plotting loss curves
    plt.figure(figsize=(12, 6))
    for batch_size in df['batch_size'].unique():
        subset = df[df['batch_size'] == batch_size]
        plt.plot(subset['epochs'], subset['metrics'].apply(lambda x: x['loss']), label=f'Batch Size: {batch_size}')
    
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()