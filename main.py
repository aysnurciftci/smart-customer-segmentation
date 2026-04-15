import pandas as pd
import os
import kagglehub
import datetime as dt

path = kagglehub.dataset_download("ulrikthygepedersen/online-retail-dataset")
file_path = os.path.join(path, "online_retail.csv")
df = pd.read_csv(file_path, encoding="ISO-8859-1")

print(df.head())
print(df.columns)
print(df.info())
print(df.isnull().sum())#id kısmındaki veri kaybı kontrolü

df = df.dropna(subset=['CustomerID'])
df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df = df[~df["InvoiceNo"].str.startswith("C")]
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
print("Temizlenmiş veri boyutu:", df.shape)#satır-sütun sayısını vericek

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["Day"] = df["InvoiceDate"].dt.day

print("Total Harcama:", df["TotalPrice"].sum())
print("Müşteri Sayısı:", df["CustomerID"].nunique())
print("Sipariş Sayısı:", df["InvoiceNo"].nunique())
print(df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10))
print(df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(10))
print(df.groupby("Month")["TotalPrice"].sum())

snapshot_date = df["InvoiceDate"].max() + dt.timedelta(days=1)#aynı gün verilen sipariş için 0 gün sorunu içim çözüm
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
})
rfm.columns = ["Recency", "Frequency", "Monetary"]
print(rfm.head(10))
print(rfm.describe())#değerleri görmek için printle

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia)
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
kmeans = KMeans(n_clusters=4, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
print(rfm.groupby("Cluster").mean())

def segment(cluster):
    if cluster==0:
       return "Low Value"
    elif cluster==1:
       return "VIP"
    elif cluster==2:
       return "Regular"
    else:
        return "Lost"
rfm["Segment"]=rfm["Cluster"].apply(segment)
rfm["Segment"].value_counts().plot(kind="bar")
plt.title("Customer Segments")
plt.show()
rfm["Cluster"].value_counts().plot(kind="bar")
plt.title("Cluster Distribution")
plt.show()
plt.scatter(rfm["Recency"], rfm["Monetary"], c=rfm["Cluster"])
plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.title("Customer Segments")
plt.show()