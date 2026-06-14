import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Reconfigure stdout to support UTF-8 characters (like emojis) in Windows terminal
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Intern Information
INTERN_ID = "CITS3911"
DEVELOPER_GITHUB = "https://github.com/zakkyaj"

def generate_synthetic_data(filepath, n_samples=500, seed=42):
    """Generates a realistic synthetic housing dataset and saves it to a CSV file."""
    np.random.seed(seed)
    
    # Generate realistic features
    area = np.random.randint(800, 4500, size=n_samples)      # Square footage
    bedrooms = np.random.randint(1, 6, size=n_samples)        # 1 to 5 bedrooms
    bathrooms = np.random.randint(1, 4, size=n_samples)       # 1 to 3 bathrooms
    parking = np.random.randint(0, 4, size=n_samples)         # 0 to 3 parking spaces
    
    # Base price calculation with realistic weightings and noise
    # Base price: $50,000
    # $120 per sq ft
    # $20,000 per bedroom
    # $35,000 per bathroom
    # $12,000 per parking space
    noise = np.random.normal(0, 15000, size=n_samples)        # Random noise std dev $15,000
    
    price = 50000 + (120 * area) + (20000 * bedrooms) + (35000 * bathrooms) + (12000 * parking) + noise
    price = np.round(price, -2) # Round to nearest 100
    
    # Create DataFrame
    df = pd.DataFrame({
        'Area': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Parking': parking,
        'Price': price
    })
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[{INTERN_ID}] Clean housing dataset successfully generated at '{filepath}' with {n_samples} samples.")

def main():
    print("=" * 60)
    print(f"🏠 SMART HOUSE PRICE PREDICTOR - MODEL TRAINING")
    print(f"Intern ID: {INTERN_ID} | Github: {DEVELOPER_GITHUB}")
    print("=" * 60)
    
    data_path = os.path.join("data", "housing.csv")
    model_dir = "models"
    model_path = os.path.join(model_dir, "model.pkl")
    
    # 1. Generate dataset if not present
    if not os.path.exists(data_path):
        generate_synthetic_data(data_path)
    
    # 2. Load dataset
    print(f"[{INTERN_ID}] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 3. Define features and target
    X = df[['Area', 'Bedrooms', 'Bathrooms', 'Parking']]
    y = df['Price']
    
    # 4. Split data (80% train, 20% test)
    print(f"[{INTERN_ID}] Splitting dataset into train and test sets (80-20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Train Random Forest Regressor
    print(f"[{INTERN_ID}] Initializing Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    print(f"[{INTERN_ID}] Fitting model on training data...")
    model.fit(X_train, y_train)
    
    # 6. Evaluate Model
    print(f"[{INTERN_ID}] Evaluating model performance on test set...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("-" * 60)
    print(f"📊 MODEL METRICS REPORT")
    print(f"Intern ID: {INTERN_ID}")
    print(f"Mean Absolute Error (MAE) : ${mae:,.2f}")
    print(f"Mean Squared Error (MSE)   : ${mse:,.2f}")
    print(f"Root Mean Sq. Error (RMSE) : ${rmse:,.2f}")
    print(f"R-squared (R2 Accuracy)    : {r2:.4f} ({r2 * 100:.2f}%)")
    print("-" * 60)
    
    # 7. Save Model and Metadata
    os.makedirs(model_dir, exist_ok=True)
    
    payload = {
        'model': model,
        'metrics': {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        },
        'intern_id': INTERN_ID,
        'developer_github': DEVELOPER_GITHUB,
        'features': list(X.columns)
    }
    
    print(f"[{INTERN_ID}] Exporting trained model and metadata to {model_path}...")
    joblib.dump(payload, model_path)
    print(f"[{INTERN_ID}] Model training completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
