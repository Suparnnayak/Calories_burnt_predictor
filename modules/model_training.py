from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae
import joblib

def train_models(X_train, Y_train, X_val, Y_val):
    """
    Train multiple ML models and return performance
    """
    models = [LinearRegression(), XGBRegressor(), Lasso(), RandomForestRegressor(), Ridge()]
    results = {}
    
    for model in models:
        model.fit(X_train, Y_train)
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        
        results[str(model)] = {
            'Training MAE': mae(Y_train, train_preds),
            'Validation MAE': mae(Y_val, val_preds),
            'model': model
        }
    return results

def save_model(model, scaler, model_path='artifacts/xgb_model.pkl', scaler_path='artifacts/scaler.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
