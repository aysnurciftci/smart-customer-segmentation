import pandas as pd
import os
import kagglehub
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. VERIYI YUKLEME VE TEMIZLEME KISMI
def load_and_clean_data():
    local_file = "online_retail.csv"
    if os.path.exists(local_file):
        print("-> Veri seti yerel diskten yukleniyor...")
        file_path = local_file
    else:
        print("-> Veri seti Kaggle uzerinden indiriliyor...")
        path = kagglehub.dataset_download("ulrikthygepedersen/online-retail-dataset")
        file_path = os.path.join(path, "online_retail.csv")

    df = pd.read_csv(file_path, encoding="ISO-8859-1", on_bad_lines='skip', low_memory=False)
    df.columns = df.columns.str.replace(' ', '')
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors='coerce')
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors='coerce')
    df = df.dropna(subset=['Quantity', 'UnitPrice'])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df

print("ETL ve K-Means Segmentasyon Sistemi Baslatiliyor...")
df = load_and_clean_data()
max_date = df["InvoiceDate"].max()
cutoff_purchase = max_date - dt.timedelta(days=30)
df_features_purchase = df[df["InvoiceDate"] <= cutoff_purchase]

rfm = df_features_purchase.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (cutoff_purchase - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
})
rfm.columns = ["Recency", "Frequency", "Monetary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)
print("K-Means Kumeleme tamamlandi.")
