import pandas as pd
import os
import kagglehub
import datetime as dt
from flask import Flask, jsonify, send_from_directory
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

app = Flask(__name__)

# Global değişkenleri tanımlıyoruz
rfm_data = None
df_global = None
model_purchase = None
model_churn = None
cached_segment_products = {}  # segment bazlı popüler ürünleri tutacak sözlük
X_global_columns = []
X_test_b_global, y_test_b_global = None, None
X_test_c_global, y_test_c_global = None, None
ml_metrics_cache = {}


# 1. VERİYİ YÜKLEME VE TEMİZLEME KISMI
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

    df.columns = df.columns.str.replace(' ', '')     # sütun isimlerindeki boşluklari silelim



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


def initialize_system():
    global rfm_data, df_global, model_purchase, model_churn, cached_segment_products, X_global_columns
    global X_test_b_global, y_test_b_global, X_test_c_global, y_test_c_global
    global ml_metrics_cache

    print("\n" + "=" * 60)
    print(" [SISTEM BASLATILIYOR] Veriler yukleniyor ve temizleniyor...")
    print("=" * 60)

    df = load_and_clean_data()
    df_global = df

    print(f"-> Ham e-ticaret verisi basariyla islendi. Satir Sayisi: {df.shape[0]}")

    max_date = df["InvoiceDate"].max()

    # Model 1 (Gelecek Satın Alma): Son 30 gün test dönemi
    cutoff_purchase = max_date - dt.timedelta(days=30)
    df_features_purchase = df[df["InvoiceDate"] <= cutoff_purchase]
    active_in_target_purchase = df[df["InvoiceDate"] > cutoff_purchase]["CustomerID"].unique()

    # Model 2 (Churn Tahmini): Son 90 gün işlem yapmama durumu
    cutoff_churn = max_date - dt.timedelta(days=90)
    active_in_target_churn = df[df["InvoiceDate"] > cutoff_churn]["CustomerID"].unique()

    # Ana rfm tablosunu gruplayarak oluşturuyoruz
    rfm = df_features_purchase.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (cutoff_purchase - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    })
    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # --- Özellik mühendisliği kısmı ---
    rfm["CLV"] = rfm["Monetary"] * rfm["Frequency"]
    rfm["Avg_Order_Value"] = rfm["Monetary"] / rfm["Frequency"]

    customer_age = df_features_purchase.groupby("CustomerID")["InvoiceDate"].agg(
        lambda x: (cutoff_purchase - x.min()).days
    )
    rfm["Customer_Age"] = customer_age

    product_variety = df_features_purchase.groupby("CustomerID")["Description"].nunique()
    rfm["Product_Variety"] = product_variety

    # Hedef değişkenlerimizi (etiketleri) atayalım
    rfm['Likely_to_Buy'] = rfm.index.isin(active_in_target_purchase).astype(int)
    rfm['Is_Churn'] = (~rfm.index.isin(active_in_target_churn)).astype(int)

    # K-MEANS SEGMENTASYON KISMI
    print("\n[MUHENDISLIK] 7 Boyutlu genisletilmis ozellik seti matrise donusturuluyor...")
    X = rfm[["Recency", "Frequency", "Monetary", "CLV", "Avg_Order_Value", "Customer_Age", "Product_Variety"]]
    X_global_columns = X.columns.tolist()

    # Verileri ölçeklendirip kmeans modelini çalıştırıyoruz
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[MUHENDISLIK] K-Means kumeleme mimarisi tetikleniyor...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    # Rfm değerlerine göre kümeleri sıralayalım
    cluster_stats = rfm.groupby("Cluster").mean()
    cluster_scores = (
            cluster_stats["Frequency"].rank() +
            cluster_stats["Monetary"].rank() -
            cluster_stats["Recency"].rank(ascending=False)
            ).sort_values()

    cluster_mapping = {old_cluster: idx for idx, old_cluster in enumerate(cluster_scores.index)}
    rfm["Ordered_Cluster"] = rfm["Cluster"].map(cluster_mapping)

    def assign_segment(cluster_id):
        if cluster_id == 3:
            return "VIP"
        elif cluster_id == 2:
            return "Regular"
        elif cluster_id == 1:
            return "Low Value"
        else:
            return "Lost"

    rfm["Segment"] = rfm["Ordered_Cluster"].apply(assign_segment)

    print("\n--- SEGMENTASYON DAGILIM RAPORU (K-MEANS) ---")
    for seg_name in ["Lost", "Low Value", "Regular", "VIP"]:
        count = rfm[rfm["Segment"] == seg_name].shape[0]
        print(f" -> {seg_name:<12} Segmentindeki Tekil Musteri: {count}")

    # --- segment bazlı popüler ürünleri önbellege alalım ---
    print("\n[MUHENDISLIK] Segment bazli urun oneri motoru matrisi onbellege aliniyor...")
    for segment in rfm["Segment"].unique():
        peer_ids = rfm[rfm["Segment"] == segment].index
        top_products = df[df["CustomerID"].isin(peer_ids)].groupby("Description")["Quantity"].sum().sort_values(
            ascending=False).index.tolist()
        cached_segment_products[segment] = top_products

    # MODELLERİN EĞİTİLMESİ (SATIN ALMA VE CHURN TAHMİNİ)
    print("\n[EGITIM] Optimize edilmis Random Forest hiperparametreleri ile egitim basladi...")
    y_buy = rfm['Likely_to_Buy']
    y_churn = rfm['Is_Churn']

    # 1. model: satın alma tahmin modeli
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y_buy, test_size=0.2, random_state=42,
                                                                stratify=y_buy)
    X_test_b_global, y_test_b_global = X_test_b, y_test_b

    buy_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    buy_model.fit(X_train_b, y_train_b)

    # model basari metrikleri
    y_pred_b = buy_model.predict(X_test_b)
    y_prob_b = buy_model.predict_proba(X_test_b)[:, 1]
    acc_b = accuracy_score(y_test_b, y_pred_b)
    auc_b = roc_auc_score(y_test_b, y_prob_b)
    f1_b = f1_score(y_test_b, y_pred_b)

    # 2. model: churn tahmin modeli
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_churn, test_size=0.2, random_state=42,
                                                                stratify=y_churn)
    X_test_c_global, y_test_c_global = X_test_c, y_test_c

    churn_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    churn_model.fit(X_train_c, y_train_c)

    # Model başarı metrikleri
    y_pred_c = churn_model.predict(X_test_c)
    y_prob_c = churn_model.predict_proba(X_test_c)[:, 1]
    acc_c = accuracy_score(y_test_c, y_pred_c)
    auc_c = roc_auc_score(y_test_c, y_prob_c)
    f1_c = f1_score(y_test_c, y_pred_c)

    model_purchase = buy_model
    model_churn = churn_model
    rfm_data = rfm

    # Metrikleri bir kez hesaplayıp cache sozlüğüne yaziyoruz (hızlı çalışsın diye)
    features_b = [{'feature': f, 'importance': float(i)} for f, i in zip(X_global_columns, buy_model.feature_importances_)]
    tn_b, fp_b, fn_b, tp_b = map(int, confusion_matrix(y_test_b, y_pred_b).ravel())
    
    features_c = [{'feature': f, 'importance': float(i)} for f, i in zip(X_global_columns, churn_model.feature_importances_)]
    tn_c, fp_c, fn_c, tp_c = map(int, confusion_matrix(y_test_c, y_pred_c).ravel())

    ml_metrics_cache = {
        "purchase": {
            "accuracy": float(acc_b * 100),
            "roc_auc": float(auc_b),
            "f1_score": float(f1_b * 100),
            "feature_importances": features_b,
            "confusion_matrix": {"tn": tn_b, "fp": fp_b, "fn": fn_b, "tp": tp_b}
        },
        "churn": {
            "accuracy": float(acc_c * 100),
            "roc_auc": float(auc_c),
            "f1_score": float(f1_c * 100),
            "feature_importances": features_c,
            "confusion_matrix": {"tn": tn_c, "fp": fp_c, "fn": fn_c, "tp": tp_c}
        }
    }

    print("\n--- MODEL EGITIM PERFORMANS RAPORU ---")
    print("- Satin Alma Tahmin Modeli (30 Gunluk):")
    print(f"  * Dogruluk (Accuracy)  : %{acc_b * 100:.2f}")
    print(f"  * ROC-AUC Skoru        : {auc_b:.3f}")
    print(f"  * F1 Skoru             : %{f1_b * 100:.2f}")
    print("- Churn (Terk) Modeli (90 Gunluk):")
    print(f"  * Dogruluk (Accuracy)  : %{acc_c * 100:.2f}")
    print(f"  * ROC-AUC Skoru        : {auc_c:.3f}")
    print(f"  * F1 Skoru             : %{f1_c * 100:.2f}")
    print("=" * 60)
    print(" Tum modeller basariyla egitildi ve Flask REST API servisi hazir.")
    print("=" * 60 + "\n")


# Baslangıçta sunucu açılırken modeli eğitiyoruz
initialize_system()


def get_advanced_recommendations_optimized(customer_id, segment_name, top_n=5):
    customer_products = set(df_global[df_global['CustomerID'] == customer_id]['Description'].dropna().unique())
    peer_products = cached_segment_products.get(segment_name, [])

    recommendations = []
    for product in peer_products:
        if product not in customer_products:
            recommendations.append(product)
        if len(recommendations) >= top_n:
            break
    return recommendations if recommendations else ["Segment Genel İndirim Kataloğu"]


# 5. FLASK API ROTALARI

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    total_revenue = float(rfm_data['Monetary'].sum())
    customer_count = int(rfm_data.shape[0])
    vip_count = int(rfm_data[rfm_data['Segment'] == 'VIP'].shape[0])

    # segment bazli ortalama rfm degerleri
    avg_df = rfm_data.groupby('Segment').mean().reset_index()
    segment_averages = []
    for _, row in avg_df.iterrows():
        segment_averages.append({
            "Segment": row['Segment'],
            "Recency": round(float(row['Recency']), 1),
            "Frequency": round(float(row['Frequency']), 1),
            "Monetary": round(float(row['Monetary']), 2),
            "CLV": round(float(row['CLV']), 1)
        })

    # En yüksek CLV değerine sahip ilk 10 müşteri
    top_df = rfm_data.sort_values(by='CLV', ascending=False).head(10).reset_index()
    top_customers = []
    for _, row in top_df.iterrows():
        top_customers.append({
            "CustomerID": int(row['CustomerID']),
            "Segment": row['Segment'],
            "Recency": int(row['Recency']),
            "Frequency": int(row['Frequency']),
            "Monetary": float(row['Monetary']),
            "CLV": float(row['CLV'])
        })

    segment_counts = rfm_data['Segment'].value_counts().to_dict()

    # konsola log yazdiralim
    print(f"[API] Genel Yonetici Paneli verileri istendi. (Toplam Ciro: {total_revenue:,.2f} TL, VIP Sayisi: {vip_count})")

    return jsonify({
        "total_revenue": total_revenue,
        "customer_count": customer_count,
        "vip_count": vip_count,
        "segment_averages": segment_averages,
        "top_customers": top_customers,
        "segment_counts": segment_counts
    })


@app.route('/api/customer_list', methods=['GET'])
def get_customer_list():
    # konsola log yazdiralim
    print("[API] Musteri arama listesi yuklendi.")
    return jsonify({"customer_ids": [int(cid) for cid in rfm_data.index]})


@app.route('/api/customer/<int:customer_id>', methods=['GET'])
def get_customer_analysis(customer_id):
    if customer_id not in rfm_data.index:
        return jsonify({"error": "Müşteri bulunamadı"}), 404

    cust_row = rfm_data.loc[[customer_id]]
    segment_name = cust_row['Segment'].values[0]

    # Müşterinin 7 tane ozelligini modele verip tahmin aliyoruz
    features = cust_row[
        ["Recency", "Frequency", "Monetary", "CLV", "Avg_Order_Value", "Customer_Age", "Product_Variety"]]
    prob_buy = model_purchase.predict_proba(features)[0][1]
    prob_churn = model_churn.predict_proba(features)[0][1]

    recommendations = get_advanced_recommendations_optimized(customer_id, segment_name)

    # Karar destek eylem planı kuralları
    if prob_churn > 0.75:
        action_plan = f"PASİF MÜŞTERİ REAKTİVASYONU: Müşterinin churn (terk) riski %{prob_churn * 100:.1f} ile yüksek. Müşteriyi canlandırmak amacıyla özel indirim kuponu tanımlanmalı ve anlık bildirim gönderilmelidir."
    elif prob_buy > 0.75:
        action_plan = f"SATIŞ FIRSATI: Satın alma iştahı %{prob_buy * 100:.1f} ile çok yüksek! Listelenen akran ürünleri için anlık mobil push bildirimi gönderin."
    elif segment_name == "VIP":
        action_plan = "VIP KORUMA PROTOKOLÜ: Müşteri ilişkileri stabil. Hesap yöneticisi tarafından rutin özel sadakat avantajları sunulmaya devam edilmelidir."
    else:
        action_plan = "STABİL: Genel müşteri profili. Standart pazarlama takvimi ve haftalık bülten gönderimleri sürdürülebilir."

    # konsola log yazdiralim
    print(f"[API] Musteri ID {customer_id} detaylari istendi. (Segment: {segment_name}, Satin Alma Egilimi: %{prob_buy * 100:.1f}, Churn Riski: %{prob_churn * 100:.1f})")

    return jsonify({
        "segment": segment_name,
        "purchase_probability": float(prob_buy * 100),
        "churn_risk": float(prob_churn * 100),
        "recommendations": recommendations,
        "decision_support": action_plan,
        "monetary": float(cust_row['Monetary'].values[0]),
        "frequency": int(cust_row['Frequency'].values[0]),
        "clv": float(cust_row['CLV'].values[0])
    })


@app.route('/api/ml_metrics', methods=['GET'])
def get_ml_metrics():
    # konsola log yazdiralim
    print("[API] Cift Model Performans metrikleri istendi.")
    return jsonify(ml_metrics_cache)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
