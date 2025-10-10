from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_data(df):
    """
    Perform preprocessing:
    - Encode gender
    - Remove highly correlated features
    - Separate features and target
    """
    df.replace({'male': 0, 'female': 1}, inplace=True)
    
    # Remove columns with data leakage
    to_remove = ['Weight', 'Duration']
    df.drop(to_remove, axis=1, inplace=True)
    
    # Separate features and target
    X = df.drop(['User_ID', 'Calories'], axis=1)
    y = df['Calories'].values
    
    return X, y

def scale_features(X_train, X_val):
    """
    Standardize the features for stable and fast training
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    return X_train_scaled, X_val_scaled, scaler
