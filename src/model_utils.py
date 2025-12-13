def create_model(model_type, **kwargs):
    """
    Create a machine learning model based on the specified type.

    Parameters:
    - model_type (str): The type of model to create (e.g., 'linear_regression', 'decision_tree').
    - **kwargs: Additional parameters for model initialization.

    Returns:
    - model: An instance of the specified model.
    """
    if model_type == 'linear_regression':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(**kwargs)
    elif model_type == 'decision_tree':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(**kwargs)
    else:
        raise ValueError(f"Model type '{model_type}' is not supported.")
    
    return model

def train_model(model, X_train, y_train):
    """
    Train the specified model using the provided training data.

    Parameters:
    - model: The model to train.
    - X_train (array-like): Training features.
    - y_train (array-like): Training labels.

    Returns:
    - model: The trained model.
    """
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    """
    Save the trained model to the specified file path.

    Parameters:
    - model: The model to save.
    - filepath (str): The path where the model will be saved.
    """
    import joblib
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load a model from the specified file path.

    Parameters:
    - filepath (str): The path from which to load the model.

    Returns:
    - model: The loaded model.
    """
    import joblib
    model = joblib.load(filepath)
    return model