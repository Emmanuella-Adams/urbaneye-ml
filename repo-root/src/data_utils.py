def load_data(file_path):
    """Load data from a specified file path."""
    import pandas as pd
    return pd.read_csv(file_path)

def save_data(data, file_path):
    """Save data to a specified file path."""
    data.to_csv(file_path, index=False)

def preprocess_data(data):
    """Preprocess the data by handling missing values and encoding categorical variables."""
    # Example preprocessing steps
    data.fillna(method='ffill', inplace=True)
    data = pd.get_dummies(data)
    return data

def split_data(data, target_column, test_size=0.2):
    """Split the data into features and target, and then into training and testing sets."""
    from sklearn.model_selection import train_test_split
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)