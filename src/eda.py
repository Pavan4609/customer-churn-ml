import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import os

# Create reports folder
os.makedirs("reports", exist_ok=True)

# Load dataset
df = pd.read_csv("data/customer_churn.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())


# 1. Churn Distribution - Bar Graph
df["Churn"].value_counts().plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("reports/churn_distribution.png")
#plt.show()
plt.close()


# 2. Monthly Charges - Histogram
df["MonthlyCharges"].plot(kind="hist", bins=30)

plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig("reports/monthly_charges.png")
#plt.show()
plt.close()


# 3. Tenure - Histogram
df["tenure"].plot(kind="hist", bins=30)

plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig("reports/tenure_distribution.png")
#plt.show()
plt.close()

# 4. Churn by Contract Type

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"]
)

contract_churn.plot(kind="bar")

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()

plt.savefig("reports/churn_by_contract.png")
plt.close()
# 5. Churn by Internet Service

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"]
)

internet_churn.plot(kind="bar")

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()

plt.savefig("reports/churn_by_internet_service.png")
plt.close()
# 6. Churn by Payment Method

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"]
)

payment_churn.plot(kind="bar")

plt.title("Customer Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Churn")
plt.tight_layout()

plt.savefig("reports/churn_by_payment_method.png")
plt.close()
# 7. Monthly Charges vs Churn - Box Plot

df.boxplot(
    column="MonthlyCharges",
    by="Churn"
)

plt.title("Monthly Charges vs Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()

plt.savefig("reports/monthly_charges_vs_churn.png")
plt.close()
# 8. Tenure vs Churn - Box Plot

df.boxplot(
    column="tenure",
    by="Churn"
)

plt.title("Customer Tenure vs Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()

plt.savefig("reports/tenure_vs_churn.png")
plt.close()
# 9. Correlation Heatmap

numeric_df = df.select_dtypes(include=["number"])

correlation = numeric_df.corr()

plt.figure(figsize=(8, 6))

plt.imshow(correlation, aspect="auto")
plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig("reports/correlation_heatmap.png")
plt.close()
print("\nGraphs saved successfully in reports folder!")