import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import joblib, os

from sklearn.linear_model  import LinearRegression
from sklearn.ensemble      import RandomForestRegressor
from xgboost               import XGBRegressor
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ------- LOAD DATA ----------
master = pd.read_csv("master_cleaned.csv")
print("Columns:", list(master.columns))
print("Shape:", master.shape)

if "is_won" not in master.columns:
    print("ERROR: is_won column missing — run data_cleaning.py first")
    exit()

won = master[master["is_won"] == 1].copy()
print(f"Won deals: {len(won)}")

# ------ MONTHLY TIME SERIES -----------
monthly = (won.groupby("close_yearmonth").agg(
    total_revenue  = ("close_value",        "sum"),
    num_deals      = ("opportunity_id",     "count"),
    avg_deal_value = ("close_value",        "mean"),
    avg_duration   = ("deal_duration_days", "mean"),
).reset_index().sort_values("close_yearmonth"))

print("\nMonthly revenue (INR):")
print(monthly[["close_yearmonth","total_revenue","num_deals"]].to_string(index=False))

# ------- LAG FEATURES -------------------
monthly["time_index"]     = range(1, len(monthly)+1)
monthly["lag_1"]          = monthly["total_revenue"].shift(1)
monthly["lag_2"]          = monthly["total_revenue"].shift(2)
monthly["lag_3"]          = monthly["total_revenue"].shift(3)
monthly["rolling_mean_3"] = monthly["total_revenue"].shift(1).rolling(3).mean()
monthly = monthly.dropna()
print(f"\nUsable months for modelling: {len(monthly)}")

# ------- FEATURES AND TARGET ----------
FEATURES = ["time_index", "lag_1", "lag_2", "lag_3",
            "rolling_mean_3", "num_deals", "avg_deal_value", "avg_duration"]
TARGET = "total_revenue"

X = monthly[FEATURES]
y = monthly[TARGET]

# ------ TRAIN TEST SPLIT ----------
split      = int(len(monthly) * 0.75)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"Train: {monthly['close_yearmonth'].iloc[0]} to {monthly['close_yearmonth'].iloc[split-1]}")
print(f"Test : {monthly['close_yearmonth'].iloc[split]} to {monthly['close_yearmonth'].iloc[-1]}")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=3,
                   subsample=0.8, colsample_bytree=0.8,
                   random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("\n=== MODEL RESULTS ===")
for name, pred in [("Linear Regression", lr_pred),
                   ("Random Forest",     rf_pred),
                   ("XGBoost",           xgb_pred)]:
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    print(f"\n{name}:")
    print(f"  MAE  = ₹{mae:,.0f}")
    print(f"  RMSE = ₹{rmse:,.0f}")
    print(f"  R²   = {r2:.3f}")

# ------ SAVE MODELS ----------
joblib.dump(lr,     "model_linear_regression.pkl")
joblib.dump(rf,     "model_random_forest.pkl")
joblib.dump(xgb,    "model_xgboost.pkl")
joblib.dump(scaler, "model_scaler.pkl")
print("\n✓ Models saved")

# ------ SAVE PREDICTIONS -----------
test_months = monthly["close_yearmonth"].iloc[split:].values
pd.DataFrame({
    "month"         : test_months,
    "actual"        : y_test.values,
    "lr_predicted"  : lr_pred.round(0),
    "rf_predicted"  : rf_pred.round(0),
    "xgb_predicted" : xgb_pred.round(0),
}).to_csv("forecast_results.csv", index=False)
print("✓ forecast_results.csv saved")

# ------- CHART: ACTUAL VS PREDICTED ----------
all_months   = monthly["close_yearmonth"].values
model_names  = ["Linear Regression", "Random Forest", "XGBoost"]
model_preds  = [lr_pred, rf_pred, xgb_pred]
model_colors = ["#378ADD", "#1D9E75", "#D85A30"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for i in range(3):
    ax    = axes[i]
    name  = model_names[i]
    pred  = model_preds[i]
    color = model_colors[i]
    ax.plot(all_months, monthly["total_revenue"], color="gray", linewidth=2, marker="o", markersize=5, label="Actual")
    ax.plot(test_months, pred, color=color, linewidth=2.5, linestyle="--", marker="o", markersize=6, label="Predicted")
    ax.set_title(name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Month (2025)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"₹{x/1e7:.1f}Cr"))
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=45)

plt.suptitle("Dhyey Consulting — Revenue Forecast 2025", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("chart6_forecast.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("✓ chart6_forecast.png saved")

# ------ CHART: FEATURE IMPORTANCE --------------
fig, ax = plt.subplots(figsize=(8, 5))
feat_imp = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values()
feat_imp.plot(kind="barh", color="#D85A30", edgecolor="none", ax=ax)
ax.set_title("XGBoost Feature Importance\n(what drives monthly revenue predictions)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("chart7_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
print("✓ chart7_feature_importance.png saved")
print("\n✓ FORECASTING COMPLETE")
print("Next → run python classification.py")