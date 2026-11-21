

# HOUSE PRICE PREDICTION - MODEL TRAINING

import pandas as pd
import joblib


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
housing = fetch_california_housing(as_frame=True)

df = housing.frame

print("Dataset loaded successfully!")
print(df.head())

# 2. Separate features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# 4. Create Linear Regression model
model = LinearRegression()

# 5. Train model
model.fit(X_train, y_train)

print("Model training completed!")

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate model
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("------------------------")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 8. Save model
joblib.dump(
    {
        "model": model,
        "feature_names": list(X.columns),
        "metrics": {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }
    },
    "house_price_model.joblib"
)

print("\nModel saved successfully!")
