import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, os

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ------- LOAD DATA ---------------
master = pd.read_csv("master_cleaned.csv")
print("Shape:", master.shape)
print("Columns:", list(master.columns))

if "is_won" not in master.columns:
    print("ERROR: is_won missing -- run data_cleaning.py first")
    exit()
print(f"Total deals: {len(master)}")
print(f"Win rate: {master['is_won'].mean()*100:.1f}%")

# ------- FEATURES ---------------
NUM_FEATURES = ["deal_duration_days", "sales_price", "revenue", "employees", "agent_win_rate"]
TARGET = "is_won"
CAT_FEATURES = []
for col in ["product", "regional_office", "series"]:
    if col in master.columns:
        CAT_FEATURES.append(col)
print(f"Categorical features found: {CAT_FEATURES}")

# -------- ENCODE TEXT COLUMNS ------------
le_dict = {}
for col in CAT_FEATURES:
    master[col] = master[col].fillna("Unknown")
    le = LabelEncoder()
    master[col] = le.fit_transform(master[col])
    le_dict[col] = le
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
print(f"All features: {ALL_FEATURES}")
model_data = master[ALL_FEATURES + [TARGET]].dropna()
X = model_data[ALL_FEATURES]
y = model_data[TARGET]
print(f"Dataset for model: {len(model_data)} rows")

# -------- TRAIN TEST SPLIT ---------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify =y, random_state=42)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Train win rate: {y_train.mean()*100:.1f}%")

# ------- TRAIN RANDOM FOREST -----------------
rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_prob = rf_clf.predict_proba(X_test)[:,1]
print("✓ Random Forest trained")

# -------- TRAIN XGBOOST -----------
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

xgb_clf = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    scale_pos_weight=neg/pos,
    eval_metric="logloss",
    random_state=42,
    verbosity=0)
xgb_clf.fit(X_train, y_train)
xgb_pred = xgb_clf.predict(X_test)
xgb_prob = xgb_clf.predict_proba(X_test)[:,1]
print("✓ XGBoost trained")

# -------- EVALUATE --------------
print("\n==== XGBOOST RESULTS ====")
print(classification_report(y_test, xgb_pred, target_names=["Lost", "Won"]))
print(f"AUC-ROC: {roc_auc_score(y_test, xgb_prob):.3f}")
cv = cross_val_score(xgb_clf, X, y, cv=5, scoring="accuracy")
print(f"\n5-fold CV Accuracy: {cv.mean():.3f} +- {cv.std():.3f}")

# -------- SAVE MODELS ----------

joblib.dump(rf_clf, "clf_random_forest.pkl")
joblib.dump(xgb_clf, "clf_xgboost.pkl")
joblib.dump(le_dict, "label_encoders.pkl")
print("✓ Models saved")

# -------- SAVE PREDICTIONS ---------
pd.DataFrame({
    "actual" : y_test.values,
    "rf_predicted" : rf_pred,
    "rf_win_prob" : rf_prob.round(2),
    "xgb_predicted" : xgb_pred,
    "xgb_win_prob" : xgb_prob.round(2),
}).to_csv("classification_results.csv", index=False)
print("✓ classification_results.csv saved")

# -------- CHART: FEATURE IMPORTANCE ---------
fig, ax = plt.subplots(figsize=(8, 5))
feat_imp = pd.Series(xgb_clf.feature_importances_, index=ALL_FEATURES).sort_values()
feat_imp.plot(kind="barh", color="#534AB7", edgecolor="none", ax=ax)
ax.set_title("What Predicts a Won Deal at Dhyey Consulting?", fontsize=13, fontweight="bold")
ax.set_xlabel("Feature Importance Score")
plt.tight_layout()
plt.savefig("chart8_clf_importance.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("✓ chart8_clf_importance.png saved")

# ------- CHART: WIN PROBABILITY DISTRIBUTION -------
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(xgb_prob[y_test == 1], bins=20, alpha=0.6, color="#1D9E75", label="Actual Won deals")
ax.hist(xgb_prob[y_test == 0], bins=20, alpha=0.6, color="#D85A30", label="Actual Lost deals")
ax.set_title("XGBoost — Predicted Win Probability Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted probability of winning (0=lost, 1=won)")
ax.set_ylabel("Number of deals")
ax.legend()
plt.tight_layout()
plt.savefig("chart9_win_probability.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("✓ chart9_win_probability.png saved")
print("\n✓ CLASSIFICATION COMPLETE")
print("Next → run python powerbi.py")