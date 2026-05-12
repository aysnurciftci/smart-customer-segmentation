import numpy as np
import pandas as pd
import os
import kagglehub
import datetime as dt

path = kagglehub.dataset_download("ulrikthygepedersen/online-retail-dataset")
file_path = os.path.join(path, "online_retail.csv")
df = pd.read_csv(file_path, encoding="ISO-8859-1", on_bad_lines='skip')

print(df.head())
print(df.columns)
print(df.info())
print(df.isnull().sum())#id kısmındaki veri kaybı kontrolü

df = df.dropna(subset=['CustomerID'])
df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df = df[~df["InvoiceNo"].str.startswith("C")]

df["Quantity"] = pd.to_numeric(df["Quantity"], errors='coerce')
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors='coerce')
df = df.dropna(subset=['Quantity', 'UnitPrice'])

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
# Tahmin için hedef değişken (Label) oluşturma
# Örnek: Recency değeri medyan değerden küçükse (yani yakın zamanda geldiyse) 1, değilse 0.
rfm['Likely_to_Buy'] = ((rfm['Recency'] < rfm['Recency'].mean()) &
                        (rfm['Frequency'] > rfm['Frequency'].median())).astype(int)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(42) # Her seferinde aynı sonucu almak için
mask = np.random.rand(len(rfm)) < 0.05 # %5'lik bir gürültü payı
rfm.loc[mask, 'Likely_to_Buy'] = 1 - rfm.loc[mask, 'Likely_to_Buy']

# X: Girdi verileri, y: Tahmin edilecek sonuç
X = rfm[['Recency', 'Frequency', 'Monetary']]
y = rfm['Likely_to_Buy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelin eğitilmesi
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

def get_customer_top_products(df, customer_id):
    top_products = df[df['CustomerID'] == customer_id].groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(5)
    return top_products

import streamlit as st
st.set_page_config(page_title="Müşteri Analiz Paneli", layout="wide")
st.title("📊 Akıllı Müşteri Segmentasyonu ve Tahmin Sistemi")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📈 Genel Bakış", "🔍 Müşteri Sorgulama", "⚙️ Model Performansı"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Satış", f"{df['TotalPrice'].sum():,.0f} TL")
    col2.metric("Müşteri Sayısı", df['CustomerID'].nunique())
    col3.metric("VIP Müşteri Sayısı", rfm[rfm['Segment'] == 'VIP'].shape[0])

    st.markdown("### 📈 Genel Müşteri Dağılımı")
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.subheader("Segment Dağılım Grafiği")
        st.bar_chart(rfm['Segment'].value_counts())

    with c_right:
        st.subheader("Segment Bazlı Ortalamalar")
        # .round(2) ekleyerek sayıları temizliyoruz
        st.write(rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(2))

    st.markdown("### ⚠️ Kritik İş Uyarıları")
    lost_customers = rfm[rfm['Segment'] == 'Lost'].shape[0]
    vip_count = rfm[rfm['Segment'] == 'VIP'].shape[0]

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.warning(
            f"**Dikkat:** {lost_customers} müşteri 'Lost' (Kaybedilmiş) kategorisinde. Acil geri kazanma senaryosu uygulanmalı.")
    with col_u2:
        st.success(f"**Fırsat:** {vip_count} VIP müşteriniz var. Onlara özel sadakat programı başlatılabilir.")

with tab2:
    st.subheader("👤 Bireysel Müşteri Derin Analizi")

    selected_id = st.selectbox("Analiz edilecek Müşteri ID", rfm.index, key="detail_search")

    if st.button("Detaylı Raporu Hazırla"):
        cust_data = rfm.loc[[selected_id]]
        pred = rf_model.predict(cust_data[['Recency', 'Frequency', 'Monetary']])[0]

        # Üst Panel: Özet Bilgiler
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Mevcut Segment", cust_data['Segment'].values[0])
        with c2:
            prob = "Yüksek" if pred == 1 else "Düşük"
            st.metric("Gelecek Ay Olasılığı", prob)
        with c3:
            # Basit bir Churn Riski hesaplama (Recency yüksekse risk yüksektir)
            risk = "Yüksek" if cust_data['Recency'].values[0] > rfm['Recency'].mean() else "Düşük"
            st.metric("Müşteri Kaybetme Riski", risk)

        st.markdown("---")

        # Alt Panel: Ürün ve Harcama Analizi
        left_col, right_col = st.columns(2)

        with left_col:
            st.write("### 🛍️ En Çok Tercih Ettiği Ürünler")
            top_p = get_customer_top_products(df, selected_id)
            if not top_p.empty:
                st.table(top_p)
            else:
                st.warning("Bu müşteriye ait ürün verisi bulunamadı.")

        with right_col:
            st.write("### 💰 Harcama İstatistikleri")
            st.write(f"- Toplam Harcama: **{cust_data['Monetary'].values[0]:,.2f} TL**")
            st.write(f"- Toplam İşlem Sayısı: **{int(cust_data['Frequency'].values[0])}**")
            st.write(
                f"- Ortalama Sepet Değeri: **{(cust_data['Monetary'].values[0] / cust_data['Frequency'].values[0]):,.2f} TL**")

with tab3:
    st.subheader("Makine Öğrenmesi Başarım Raporu")
    st.metric("Model Doğruluk Oranı (Accuracy)", f"%{accuracy * 100:.2f}")
    st.write("Bu model, Random Forest Classifier algoritması kullanılarak eğitilmiştir.")

with st.sidebar:
    st.header("🔍 Müşteri Sorgulama")
    st.success(f"🤖 Model Başarısı: %{accuracy * 100:.2f}")  # Bunu ekledik
    st.markdown("---")

    selected_id = st.selectbox("Analiz edilecek Müşteri ID", rfm.index)
    if st.button("Tahmin Et"):
        cust_data = rfm.loc[[selected_id], ['Recency', 'Frequency', 'Monetary']]
        pred = rf_model.predict(cust_data)[0]
        seg = rfm.loc[selected_id, 'Segment']

        st.markdown("---")
        st.subheader(f"Müşteri No: {selected_id}")
        st.success(f"**Segment:** {seg}")
        st.info(f"**Gelecek Ay Alışveriş Olasılığı:** {'Yüksek' if pred == 1 else 'Düşük'}")
        st.write(f"📅 Son gelişten beri: {int(cust_data['Recency'].values[0])} gün")
        st.write(f"🛒 Toplam sipariş: {int(cust_data['Frequency'].values[0])} adet")

