from modules.data_ingestion import load_data
from modules.data_preprocessing import preprocess_data, scale_features
from modules.eda import scatter_plots, distribution_plots
from modules.model_training import train_models, save_model
from sklearn.model_selection import train_test_split

# Load Data
df = load_data('data/calories.csv')

# EDA
features_for_scatter = ['Age', 'Height', 'Weight', 'Duration']
scatter_plots(df, features_for_scatter)
features_for_dist = df.select_dtypes(include='float').columns
distribution_plots(df, features_for_dist)

# Preprocessing
X, y = preprocess_data(df)
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.1, random_state=22)

X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)

# Train Models
results = train_models(X_train_scaled, Y_train, X_val_scaled, Y_val)

# Print Model Performance
for model_name, metrics in results.items():
    print(model_name)
    print('Training MAE:', metrics['Training MAE'])
    print('Validation MAE:', metrics['Validation MAE'])
    print('---')

# Save best model (XGB example)
best_model_name = min(results, key=lambda x: results[x]['Validation MAE'])
best_model = results[best_model_name]['model']
save_model(best_model, scaler)
print(f"Best model '{best_model_name}' saved successfully in 'artifacts/'")