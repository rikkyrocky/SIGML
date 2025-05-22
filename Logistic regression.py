import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
)
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("DATA/outputs/dividend_classification_dataset.csv")

# Prepare features and target
X = df.drop(columns=["Ticker", "Quarter", "Class"])
y = df["Class"]
feature_names = X.columns

# Step 1: Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Step 2: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Step 3: Setup LOOCV
loo = LeaveOneOut()

# Step 4: Build pipeline (Feature selection + Logistic Regression)
selector = SelectKBest(score_func=f_classif, k=9)
clf = LogisticRegression(max_iter=1000)
pipeline = Pipeline([
    ("select", selector),
    ("clf", clf)
])

# Step 5: Perform LOOCV
y_true, y_pred, y_prob = [], [], []

for train_idx, test_idx in loo.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline.fit(X_train, y_train)
    y_true.append(y_test.values[0])
    y_pred.append(pipeline.predict(X_test)[0])
    y_prob.append(pipeline.predict_proba(X_test)[0][1])

# Step 6: Evaluate model
accuracy = accuracy_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)
cm = confusion_matrix(y_true, y_pred)

print(f"\nLOOCV Accuracy: {accuracy:.4f}")
print(f"LOOCV ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["No Dividend", "Dividend"]))

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Dividend", "Dividend"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
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
plt.hist([p for i, p in enumerate(y_prob) if y_true[i] == 0], bins=20, alpha=0.7, label='No Dividend')
plt.hist([p for i, p in enumerate(y_prob) if y_true[i] == 1], bins=20, alpha=0.7, label='Dividend')
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Probability Distribution by Class")
plt.legend()
plt.grid(True)
plt.show()

# Feature importance (mean F-statistic over all folds)
selector.fit(X_scaled, y)
scores = selector.scores_
selected_indices = selector.get_support(indices=True)
selected_features = feature_names[selected_indices]
selected_scores = scores[selected_indices]

# Plot feature importance
plt.figure()
plt.barh(selected_features, selected_scores)
plt.xlabel("F-score")
plt.title("Top 9 Selected Features by ANOVA F-statistic")
plt.grid(True)
plt.tight_layout()
plt.show()


# === Predict Amazon ===

# Step 1: Create Amazon's input data using the same features
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

# Ensure correct order of columns
amazon_df = pd.DataFrame([amazon_data])[feature_names]

# Step 2: Apply same preprocessing: impute -> scale -> select features
amazon_imputed = imputer.transform(amazon_df)
amazon_scaled = scaler.transform(amazon_imputed)

# Refit selector and classifier on full dataset
pipeline.fit(X_scaled, y)

# Select features from Amazon
amazon_selected = selector.transform(amazon_scaled)

# Step 3: Predict
amazon_prob = clf.predict_proba(amazon_selected)[0][1]
amazon_class = clf.predict(amazon_selected)[0]

print(f"\nPredicted probability Amazon initiates a dividend (Class 1): {amazon_prob:.4f}")
print(f"Predicted class: {'Dividend' if amazon_class == 1 else 'No Dividend'}")
