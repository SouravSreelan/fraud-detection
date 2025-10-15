import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

print("Loading dataset...") 
df = pd.read_csv("data/creditcard.csv")
print("Dataset shape:", df.shape)
print(df["Class"].value_counts())
X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X["scaled_amount"] = scaler.fit_transform(X["Amount"].values.reshape(-1, 1))
X["scaled_time"] = scaler.fit_transform(X["Time"].values.reshape(-1, 1))
X = X.drop(["Amount", "Time"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print("After SMOTE:", y_res.value_counts())

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_res, y_res)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label="XGBoost (AUC = %.3f)" % roc_auc)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Fraud Detection")
plt.legend()
plt.show()






