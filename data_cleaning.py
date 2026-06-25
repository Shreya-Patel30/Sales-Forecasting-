import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── LOAD ALL FILES ─────────────────────────────────────────
pipeline = pd.read_csv("sales_pipeline.csv", sep=";")
accounts = pd.read_csv("accounts.csv",        sep=";")
products = pd.read_csv("products.csv",        sep=";")
teams    = pd.read_csv("sales_teams.csv",     sep=";")

print("✓ Files loaded")
print("Pipeline:", pipeline.shape)  
print("Teams   :", teams.shape)     

# ── FIX DATE COLUMNS ───────────────────────────────────────
pipeline["engage_date"] = pd.to_datetime(pipeline["engage_date"], dayfirst=True, errors="coerce")
pipeline["close_date"]  = pd.to_datetime(pipeline["close_date"],  dayfirst=True, errors="coerce")
print("✓ Dates converted")

# ── SHIFT DATES FROM 2016/2017 TO 2024/2025 ───────────────
pipeline["engage_date"] = pipeline["engage_date"] + pd.Timedelta(days=2922)
pipeline["close_date"]  = pipeline["close_date"]  + pd.Timedelta(days=2922)
print("✓ Dates shifted to 2024/2025")
print("  Sample:", pipeline["close_date"].dropna().head(3).dt.date.tolist())

# ── RENAME SALES AGENTS TO INDIAN NAMES ───────────────────
agent_name_map = {
    "Moses Frase"       : "Arjun Sharma",
    "Darcel Schlecht"   : "Priya Patel",
    "James Ascencio"    : "Rahul Mehta",
    "Maureen Marcano"   : "Sneha Joshi",
    "Hayden Neloms"     : "Vikram Singh",
    "Rosalina Dieter"   : "Pooja Desai",
    "Versie Hillebrand" : "Amit Verma",
    "Daniell Hammack"   : "Neha Gupta",
    "Elease Gluck"      : "Suresh Kumar",
    "Violet Mclelland"  : "Anjali Shah",
    "Kami Bicknell"     : "Rohan Nair",
    "Gladys Colclough"  : "Kavita Reddy",
    "Niesha Huffines"   : "Deepak Yadav",
    "Anna Snelling"     : "Meera Iyer",
    "Vicki Laflamme"    : "Ritesh Jain",
    "Markita Hansen"    : "Divya Pillai",
    "Zane Levy"         : "Karan Malhotra",
    "Summer Sewald"     : "Sunita Rao",
    "Cara Losch"        : "Chetan Pandey",
    "Celia Rouche"      : "Anita Mishra",
    "Rocco Neubert"     : "Manish Tiwari",
    "Melvin Marxen"     : "Rajesh Agarwal",
    "Dustin Brinkmann"  : "Sanjay Bhatt",
    "Donn Cantrell"     : "Nikhil Soni",
    "Lajuana Vencill"   : "Preeti Kulkarni",
    "Casimira Puppo"    : "Ravi Saxena",
    "Marty Mcfarland"   : "Gaurav Tomar",
    "Wilburn Farley"    : "Ashish Dubey",
    "Kenna Purnell"     : "Shruti Bose",
    "Garret Dahl"       : "Tarun Kapoor",
    "Cecily Lampkin"    : "Usha Naik",
    "Bertie Lupo"       : "Vinod Chavan",
    "Loreta Tweed"      : "Pallavi Hegde",
    "Reed Clapper"      : "Aakash Trivedi",
}

manager_name_map = {
    "Dustin Brinkmann" : "Sanjay Bhatt",
    "Melvin Marxen"    : "Rajesh Agarwal",
    "Cara Losch"       : "Chetan Pandey",
    "Rocco Neubert"    : "Manish Tiwari",
    "Celia Rouche"     : "Anita Mishra",
    "Summer Sewald"    : "Sunita Rao",
}

pipeline["sales_agent"] = pipeline["sales_agent"].map(agent_name_map).fillna(pipeline["sales_agent"])
teams["sales_agent"]    = teams["sales_agent"].map(agent_name_map).fillna(teams["sales_agent"])
teams["manager"]        = teams["manager"].map(manager_name_map).fillna(teams["manager"])
print("✓ Agent and manager names changed to Indian names")

# ── FIX PRODUCT NAMES ─────────────────────────────────────
pipeline["product"] = pipeline["product"].str.strip().str.lower().str.replace(" ", "", regex=False)
products["product"] = products["product"].str.strip().str.lower().str.replace(" ", "", regex=False)

product_map = {
    "gtxpro"       : "D365 Business Central",
    "gtxpluspro"   : "D365 Finance & Operations",
    "gtxbasic"     : "D365 Sales",
    "gtxplusbasic" : "D365 Field Services",
    "mgadvanced"   : "D365 Project Management",
    "mgspecial"    : "Power BI Dashboard",
    "gtk500"       : "Power Apps Development",
}

pipeline["product"] = pipeline["product"].map(product_map).fillna(pipeline["product"])
products["product"] = products["product"].map(product_map).fillna(products["product"])
print("✓ Products renamed to Dhyey service names")
print("  Services:", sorted(pipeline["product"].unique()))

# ── FIX OFFICE LOCATION TO INDIAN CITIES ──────────────────
india_cities = [
    "Mumbai", "Ahmedabad", "Vadodara", "Surat",
]

np.random.seed(42)
accounts["office_location"] = np.random.choice(india_cities, size=len(accounts))
print("✓ Office locations changed to Indian cities")

# ── KEEP ONLY CLOSED DEALS ─────────────────────────────────
closed = pipeline[pipeline["deal_stage"].isin(["Won", "Lost"])].copy()
closed["account"] = closed["account"].fillna("Unknown")
closed = closed[closed["close_value"] >= 0]
print(f"✓ Closed deals: {len(closed)} rows")

# ── CLEAN ACCOUNTS TABLE ───────────────────────────────────
accounts = accounts.drop(columns=["subsidiary_of"])
accounts["sector"] = (accounts["sector"]
                      .str.strip()
                      .str.lower()
                      .str.replace("technolgy", "technology", regex=False))
print("✓ Accounts cleaned")

# ── JOIN ALL TABLES ────────────────────────────────────────
master = pd.merge(closed,  teams,    on="sales_agent", how="left")
master = pd.merge(master,  products, on="product",     how="left")
master = pd.merge(master,  accounts, on="account",     how="left")
print(f"✓ Tables joined → shape: {master.shape}")

# ── FEATURE ENGINEERING ────────────────────────────────────
master["deal_duration_days"] = (master["close_date"] - master["engage_date"]).dt.days
master.loc[master["deal_duration_days"] < 0, "deal_duration_days"] = np.nan

master["close_month"]     = master["close_date"].dt.month
master["close_year"]      = master["close_date"].dt.year
master["close_quarter"]   = master["close_date"].dt.quarter
master["close_yearmonth"] = master["close_date"].dt.to_period("M").astype(str)

master["discount_rate"] = np.where(
    master["sales_price"] > 0,
    (master["sales_price"] - master["close_value"]) / master["sales_price"],
    0
)
master["discount_rate"]  = master["discount_rate"].clip(0, 1)
master["is_won"]         = (master["deal_stage"] == "Won").astype(int)
master["agent_win_rate"] = master.groupby("sales_agent")["is_won"].transform("mean").round(2)

for col in ["deal_duration_days", "revenue", "employees", "sales_price", "discount_rate"]:
    if col in master.columns:
        master[col] = master[col].fillna(master[col].median())

print("✓ Features engineered")

# ── CONVERT USD TO INR ─────────────────────────────────────
USD_TO_INR = 83
master["close_value"] = (master["close_value"] * USD_TO_INR).round(0)
master["sales_price"] = (master["sales_price"] * USD_TO_INR).round(0)
master["revenue"]     = (master["revenue"]     * USD_TO_INR).round(0)
print("✓ Currency converted to INR (×83)")

# ── SAVE ───────────────────────────────────────────────────
master.to_csv("master_cleaned.csv", index=False, sep=",")

print("\n" + "="*45)
print("STEP 1 COMPLETE")
print("="*45)
print(f"Shape          : {master.shape}")
print(f"Years in data  : {sorted(master['close_year'].unique())}")
print(f"Won            : {master['is_won'].sum()}")
print(f"Lost           : {(master['is_won']==0).sum()}")
print(f"Win %          : {master['is_won'].mean()*100:.1f}%")
print(f"Sample agents  : {master['sales_agent'].unique()[:4].tolist()}")
print(f"Sample cities  : {master['office_location'].dropna().unique()[:4].tolist()}")
print(f"Sample dates   : {master['close_yearmonth'].unique()[:4].tolist()}")
print(f"\nNext → run python eda.py")