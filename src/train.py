import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
df = pd.read_csv("data/customer_churn.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())


# 2. Remove customer ID
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)


# 3. Convert TotalCharges to numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )


# 4. Convert Churn to 0 and 1
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# 5. Separate input and output
X = df.drop("Churn", axis=1)
y = df["Churn"]


# 6. Find numerical and categorical columns
categorical_columns = X.select_dtypes(
    include=["object"]
).columns

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns


# 7. Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# 8. Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 9. Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])


# 10. Create ML model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# 11. Create complete ML pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# 12. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 13. Train model
print("\nTraining model...")
pipeline.fit(X_train, y_train)

print("Training completed!")


# 14. Make predictions
y_pred = pipeline.predict(X_test)


# 15. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 16. Save model
joblib.dump(
    pipeline,
    "models/model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/model.pkl")