import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
)

# Load dataset
print("Loading dataset...")
df = pd.read_csv("DATA/outputs/dividend_classification_dataset.csv")

# Prepare features and target
print("Preprocessing data...")
X = df.drop(columns=["Ticker", "Quarter", "Class"])
y = df["Class"]
feature_names = X.columns

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Determine number of splits such that each test set has 2 instances
n_splits = len(df) // 2
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

y_true_all, y_pred_all, y_prob_all = [], [], []

print(f"Starting cross-validation with {n_splits} folds (2 test instances per fold)...")
for i, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), start=1):
    print(f"Training fold {i}...")
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)
    y_prob_all.extend(y_prob)

    print(f"Fold {i} complete.")

# Evaluation
print("\nModel Evaluation:")
accuracy = accuracy_score(y_true_all, y_pred_all)
roc_auc = roc_auc_score(y_true_all, y_prob_all)
conf_matrix = confusion_matrix(y_true_all, y_pred_all)
class_report = classification_report(y_true_all, y_pred_all, target_names=["No Dividend", "Dividend"])

print(f"\nRandom Forest Accuracy: {accuracy:.4f}")
print(f"Random Forest ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["No Dividend", "Dividend"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix RandomForestML model")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true_all, y_prob_all)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Predicted probability distribution
plt.figure()
plt.hist([p for i, p in enumerate(y_prob_all) if y_true_all[i] == 0], bins=20, alpha=0.7, label='No Dividend')
plt.hist([p for i, p in enumerate(y_prob_all) if y_true_all[i] == 1], bins=20, alpha=0.7, label='Dividend')
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Probability Distribution by Class")
plt.legend()
plt.grid(True)
plt.show()


# Amazon feature values
amazon_data = {
    "Gross Margin": 0.49,
    "Operating Margin": 0.11,
    "Net Margin": 0.10,
    "Interest Coverage Ratio": 34.02,
    "Revenue Growth": 0.085,
    "R&D Intensity": 0.15,
    "SG&A / Revenue": 0.08,
    "Quick Ratio": 0.85,
    "Debt to Equity": 0.18,
    "Retention Ratio": 0.29,
}

# Ensure the order of features matches training set
amazon_df = pd.DataFrame([amazon_data])[feature_names]

# Apply imputation and scaling
amazon_imputed = imputer.transform(amazon_df)
amazon_scaled = scaler.transform(amazon_imputed)

# Train final model on all data
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_scaled, y)

# Predict probability for Amazon
amazon_prob = final_model.predict_proba(amazon_scaled)[0][1]
amazon_class = final_model.predict(amazon_scaled)[0]

print(f"\nPredicted probability Amazon initiates a dividend (Class 1): {amazon_prob:.4f}")
print(f"Predicted class: {'Dividend' if amazon_class == 1 else 'No Dividend'}")