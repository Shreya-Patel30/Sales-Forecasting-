import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ----- LOAD DATA --------------
master = pd.read_csv("master_cleaned.csv")
print("Shape:", master.shape)
if "is_won" not in master.columns:
    print("ERROR: is_won missing -- run data_cleaning.py first")
    exit()
won = master[master["is_won"]==1].copy()
print(f"Won deals: {len(won)}")

# ------- EXPORT 1: MONTHLY REVENUE ----------
monthly = (won.groupby("close_yearmonth").agg(total_revenue = ("close_value", "sum"), number_of_deals = ("opportunity_id", "count"), avg_deal_value = ("close_value", "mean"),).reset_index().sort_values("close_yearmonth"))
monthly["avg_deal_value"] = monthly["avg_deal_value"].round(0)

if os.path.exists("forecast_results.csv"):
    fc = pd.read_csv("forecast_results.csv")
    fc = fc.rename(columns={"month": "close_yearmonth"})
    monthly = monthly.merge(
        fc[["close_yearmonth", "xgb_predicted"]],
        on="close_yearmonth", how="left")
monthly = monthly.rename(columns={"xgb_redicted": "xgb_forecast_inr"})
monthly.to_csv("powerbi_monthly_revenue.csv", index=False)
print("✓ powerbi_monthly_revenue.csv saved")
print(monthly.to_string(index=False))

# -------- EXPORT 2: AGENT PERFORMANCE -----------
agent_cols = ["sales_agent", "regional_office"]
if "manager" in master.columns:
    agent_cols.append("manager")
agent = (master.groupby(agent_cols).agg(
    total_deals = ("opportunity_id", "count"),
    won_deals = ("is_won", "sum"),
    total_revenue = ("close_value", "sum"),
    avg_deal_val = ("close_value", "mean"),
).reset_index())
agent["win_rate_pct"] = (agent["won_deals"]/ agent["total_deals"] * 100).round(1)
agent["total_revenue"] = agent["total_revenue"].round(0)
agent["avg_deal_val"] = agent["avg_deal_val"].round(0)
agent.sort_values("total_revenue", ascending=False).to_csv("powerbi_agents.csv", index=False)
print("✓ powerbi_agents.csv saved")

# ------- EXPORT 3: PRODUCT PERFORMANCE ------------
prod_cols = ["product"]
if "series" in master.columns:
    prod_cols.append("series")
prod = (master.groupby(prod_cols).agg(
    total_deals = ("opportunity_id", "count"),
    won_deals = ("is_won", "sum"),
    total_revenue = ("close_value", "sum"),
    list_price = ("sales_price", "first"),
).reset_index())
prod["win_rate_pct"] = (prod["won_deals"]/ prod["total_deals"]*100).round(1)
prod["total_revenue"] = prod["total_revenue"].round(0)
prod.sort_values("total_revenue", ascending=False).to_csv("powerbi_products.csv", index=False)
print("✓ powerbi_products.csv saved")

# ------ EXPORT 4: DEAL DETAIL ----------
keep_cols = [
    "opportunity_id", "sales_agent", "regional_office", "product", "account", "deal_stage", "is_won", "close_yearmonth", "close_month", "close_quarter", "deal_duration_days", "close_value", "sales_price", "discount_rate", "agent_win_rate"
]

for col in ["manager", "series", "sector", "revenue", "employees", "office_loaction"]:
    if col in master.columns:
        keep_cols.append(col)
detail = master[keep_cols].copy()
detail["discount_rate"] = (detail["discount_rate"] * 100).round(1)
detail["agent_win_rate"] = (detail["agent_win_rate"] * 100).round(1)
detail.to_csv("powerbi_deals.csv", index=False)
print("✓ powerbi_deals.csv saved")

# ------- SUMMARY ----------
print("\n" + "="*45)
print("POWER BI EXPORT COMPLETE")
print("="*45)
print(f"\nTotal Won Revenue: ₹{won['close_value'].sum():,.0f}")
print(f"Overall Win Rate: {master['is_won'].mean()*100:.1f}%")
print(f"Total Deals : {len(master)}")
print(f"\nFiles saved:")
print("1. powerbi_monthly_revenue.csv")
print("2. powerbi_agents.csv")
print("3. powerbi_products.csv")
print("4. powerbi_deals.csv")
print("\nImport all 4 files into Power BI Desktop")
print("Build 5 pages:")
print("Page 1 -- Executive Overview")
print("Page 2 -- Revenue Forecast")
print("Page 3 -- Agent Analysis")
print("Page 4 -- Service Performance")
print("Page 5 -- Deal Detail Table")