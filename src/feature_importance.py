import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/customer_churn.csv")

# Load trained model
pipeline = joblib.load("models/model.pkl")

# Remove customer ID
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)
# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
# Remove target
X = df.drop("Churn", axis=1)

# Get preprocessing step
preprocessor = pipeline.named_steps["preprocessor"]

# Get model
model = pipeline.named_steps["model"]

# Transform data
X_transformed = preprocessor.transform(X)

# Get feature names
feature_names = preprocessor.get_feature_names_out()

# Get importance
importance = model.feature_importances_

# Create dataframe
feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort
feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(feature_df.head(15))

# Plot top 15
top_features = feature_df.head(15)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importance")

plt.tight_layout()
plt.show()