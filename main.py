import logging
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from fastapi import FastAPI, HTTPException

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset (replace with your dataset)
data = pd.read_csv("equipment_failure_data.csv")

# Preprocess Data
def preprocess_data(data):
    # Remove unnecessary columns
    data = data.drop(columns=["timestamp", "sensor_id"])
    
    # Feature engineering: Create new features
    data["rolling_mean"] = data.groupby("equipment_id")["value"].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = ["value", "rolling_mean"]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(data[numerical_features])
    data["pca_feature_1"] = pca_features[:, 0]
    data["pca_feature_2"] = pca_features[:, 1]
    
    # Convert categorical features to one-hot encoding
    data = pd.get_dummies(data, columns=["equipment_type"])
    
    return data

# Split Data
def split_data(data):
    X = data.drop(columns=["target"])
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train Model with Grid Search
def train_model(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10]
    }
    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Predict Equipment Failure
def predict_failure(model, data):
    predictions = model.predict(data)
    return predictions

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report

# Initialize FastAPI
app = FastAPI()

# Endpoint to Predict Equipment Failure
@app.post("/predict/")
async def predict_failure(data: dict):
    try:
        # Load the trained model with best hyperparameters from grid search
        trained_model = train_model(X_train, y_train)
        preprocessed_data = preprocess_data(data)
        predictions = predict_failure(trained_model, preprocessed_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    # Preprocess the data
    preprocessed_data = preprocess_data(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(preprocessed_data)
    
    # Train the model
    trained_model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluation_report = evaluate_model(trained_model, X_test, y_test)
    print(evaluation_report)
    
    # Initialize the FastAPI server
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
