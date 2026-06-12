import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def train_and_save():
    print("Starting model training pipeline...")
    
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(BASE_DIR, "dataset", "salary_prediction.csv")
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    data = pd.read_csv(dataset_path)
    print(f"Dataset loaded. Total rows: {len(data)}")
    
    # Encode text columns
    label_encoders = {}
    categorical_columns = [
        "job_title",
        "education_level",
        "industry",
        "company_size",
        "location",
        "remote_work"
    ]
    
    print("Fitting LabelEncoders...")
    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
        print(f"Encoded '{column}' with classes: {list(le.classes_)}")
        
    # Features and target
    features = [
        "job_title",
        "experience_years",
        "education_level",
        "skills_count",
        "industry",
        "company_size",
        "location",
        "remote_work",
        "certifications"
    ]
    
    X = data[features]
    y = data["salary"]
    
    # Train model
    print("Training RandomForestRegressor (n_estimators=40, max_depth=12)...")
    model = RandomForestRegressor(
        n_estimators=40,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    print("Model training completed successfully!")
    
    # Define save paths
    model_save_path = os.path.join(BASE_DIR, "salary_model.joblib")
    encoders_save_path = os.path.join(BASE_DIR, "label_encoders.joblib")
    
    # Save the objects
    print(f"Saving model to: {model_save_path}")
    joblib.dump(model, model_save_path)
    
    print(f"Saving label encoders to: {encoders_save_path}")
    joblib.dump(label_encoders, encoders_save_path)
    
    print("Training pipeline finished successfully!")

if __name__ == "__main__":
    train_and_save()
