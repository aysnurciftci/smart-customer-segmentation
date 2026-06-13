import pandas as pd
import os
import kagglehub
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

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

df = load_and_clean_data()
max_date = df["InvoiceDate"].max()

cutoff_purchase = max_date - dt.timedelta(days=30)
df_features_purchase = df[df["InvoiceDate"] <= cutoff_purchase]
active_in_target_purchase = df[df["InvoiceDate"] > cutoff_purchase]["CustomerID"].unique()

cutoff_churn = max_date - dt.timedelta(days=90)
active_in_target_churn = df[df["InvoiceDate"] > cutoff_churn]["CustomerID"].unique()

rfm = df_features_purchase.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (cutoff_purchase - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
})
rfm.columns = ["Recency", "Frequency", "Monetary"]

rfm["CLV"] = rfm["Monetary"] * rfm["Frequency"]
rfm["Avg_Order_Value"] = rfm["Monetary"] / rfm["Frequency"]

customer_age = df_features_purchase.groupby("CustomerID")["InvoiceDate"].agg(
    lambda x: (cutoff_purchase - x.min()).days
)
rfm["Customer_Age"] = customer_age

product_variety = df_features_purchase.groupby("CustomerID")["Description"].nunique()
rfm["Product_Variety"] = product_variety

rfm['Likely_to_Buy'] = rfm.index.isin(active_in_target_purchase).astype(int)
rfm['Is_Churn'] = (~rfm.index.isin(active_in_target_churn)).astype(int)

X = rfm[["Recency", "Frequency", "Monetary", "CLV", "Avg_Order_Value", "Customer_Age", "Product_Variety"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

# ML Training
y_buy = rfm['Likely_to_Buy']
y_churn = rfm['Is_Churn']

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y_buy, test_size=0.2, random_state=42, stratify=y_buy)
buy_model = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
buy_model.fit(X_train_b, y_train_b)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_churn, test_size=0.2, random_state=42, stratify=y_churn)
churn_model = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
churn_model.fit(X_train_c, y_train_c)
print("Modeller egitildi.")
