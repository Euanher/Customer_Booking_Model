import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_score,
                             recall_score, f1_score, accuracy_score,
                             confusion_matrix)

# ---------- 1. Load (full 50,000-row file) ----------
df = pd.read_csv("customer_booking.csv", encoding="ISO-8859-1")

y = df["booking_complete"]
X = df.drop(columns=["booking_complete"])

# ---------- 2. Encode categoricals ----------
# route + booking_origin are high-cardinality -> one-hot is fine for RF,
# but consider target/frequency encoding if memory is tight.
cat_cols = ["sales_channel", "trip_type", "flight_day",
            "route", "booking_origin"]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# ---------- 3. Stratified split ----------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- 4. Train RandomForest ----------
rf = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=5,
    class_weight="balanced",   # ~85/15 imbalance -> don't let majority dominate
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_tr, y_tr)

# ---------- 5. Evaluate ----------
p     = rf.predict_proba(X_te)[:, 1]
pred  = (p >= 0.5).astype(int)

print(f"ROC-AUC   {roc_auc_score(y_te, p):.3f}")       # ~0.80
print(f"Precision {precision_score(y_te, pred):.3f}")  # ~0.38
print(f"Recall    {recall_score(y_te, pred):.3f}")     # ~0.55
print(f"F1        {f1_score(y_te, pred):.3f}")         # ~0.45
print(f"Accuracy  {accuracy_score(y_te, pred):.3f}")   # ~0.82
print(f"Baseline  {1 - y_te.mean():.3f}")              # ~0.85 (majority class)
print("Confusion matrix:\n", confusion_matrix(y_te, pred))

# ---------- 6. Gini importance -> top-10 chart ----------
imp = (pd.Series(rf.feature_importances_, index=X.columns)
         .sort_values(ascending=False).head(10))

fig, ax = plt.subplots(figsize=(8, 4))
imp.iloc[::-1].plot.barh(ax=ax, color="#1f77b4")
ax.set_xlabel("Gini importance")
ax.set_title("Top-10 feature importances — RandomForest")
plt.tight_layout()
plt.savefig("gini_top10.png", dpi=150)
plt.show()
