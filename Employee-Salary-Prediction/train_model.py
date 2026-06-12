import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv(r"C:\Users\LENOVO\Documents\pythonprojects\Employee-Salary-Prediction\dataset\salary_prediction.csv")

# Convert text columns into numbers
label_cols = [
    "job_title",
    "education_level",
    "industry",
    "company_size",
    "location",
    "remote_work"
]

encoder = LabelEncoder()

for col in label_cols:
    data[col] = encoder.fit_transform(data[col])

# Input features
X = data[
    [
        "experience_years",
        "education_level",
        "skills_count",
        "industry",
        "company_size",
        "location",
        "remote_work",
        "certifications",
        "job_title"
    ]
]

# Output
y = data["salary"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict salaries
y_pred = model.predict(X_test)

# Check accuracy
score = r2_score(y_test, y_pred)

print("Model Accuracy:", round(score * 100, 2), "%")