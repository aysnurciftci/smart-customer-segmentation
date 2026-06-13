# Akıllı Müşteri Segmentasyonu ve Satın Alım Tahmini Sistemi

Bu proje, e-ticaret fatura hareketlerini işleyerek müşterilerin satın alma alışkanlıklarını analiz eden, onları gözetimsiz öğrenme yöntemleriyle segmentlere ayıran ve gelecekteki satın alma ile terk (churn) eğilimlerini tahmin eden uçtan uca bir **Karar Destek Karar ve Öneri Sistemidir**.

Proje kapsamında ham işlem verileri işlenerek 7 boyutlu özellik matrisine dönüştürülmüş, K-Means Kümeleme ve Random Forest Sınıflandırma algoritmaları kullanılarak karar destek kuralları tanımlanmış ve Flask tabanlı bir REST API ile web arayüzüne entegre edilmiştir.

---

## 📂 Proje Dizin Yapısı

Proje dizini, kod kalitesini ve modülerliği korumak amacıyla yalın ve standartlara uygun bir yapıda kurgulanmıştır:

```text
smart-customer-segmentation/
├── main.py               # Flask REST API & Makine Öğrenmesi İşlem Hattı (Pipeline)
├── index.html            # Yönetici Paneli & Kullanıcı Arayüzü (Frontend)
├── online_retail.csv     # Kaggle Online Retail Fatura Veri Seti (Veri Kaynağı)
├── requirements.txt      # Python Bağımlılık Paketleri ve Kütüphane Sürümleri
├── README.md             # Akademik ve Teknik Açıklama Dokümanı (Bu dosya)
└── .gitignore            # Git Sürüm Kontrolü Yoksayma Listesi
```

---

## 🛠️ Sistem Mimarisi ve Kullanılan Teknolojiler

Proje mimarisi veri toplama, önişleme, özellik mühendisliği, modelleme, API sunumu ve görselleştirme katmanlarından oluşur:
* **Veri İşleme ve Temizleme:** Python, Pandas, NumPy
* **Makine Öğrenmesi (ML) Altyapısı:** Scikit-learn (K-Means, RandomForestClassifier, StandardScaler)
* **API Katmanı (Backend):** Flask (RESTful yönlendirme ve JSON veri alışverişi)
* **Arayüz Katmanı (Frontend):** HTML5, Vanilla CSS3 (Bootstrap Grid Layout), JavaScript (Fetch API, Asenkron İstekler, Chart.js Grafikleri)

---

## ⚙️ Kurulum ve Çalıştırma Adımları

Sistemin yerel ortamda kurulumu ve koşturulması için aşağıdaki adımları takip ediniz:

### 1. Sanal Ortam (Virtual Environment) Yapılandırması
Bağımlılıkların izole bir ortamda kurulması için proje dizininde sanal ortam oluşturup aktifleştirin:

**Windows PowerShell/CMD:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Bağımlılıkların Kurulması
Gerekli kütüphaneleri yüklemek için:
```bash
pip install -r requirements.txt
```

### 3. Backend Sunucusunun Başlatılması
REST API servisini ve makine öğrenmesi hattını tetiklemek için:
```bash
python main.py
```
Sunucu başlatıldığında, yerel dizinde `online_retail.csv` dosyası varsa doğrudan okunur, aksi takdirde `kagglehub` kütüphanesi aracılığıyla veri seti otomatik olarak indirilerek işlenir. Modellerin eğitimi ve doğrulama süreçleri tamamlandıktan sonra uygulama **`http://127.0.0.1:5000`** adresi üzerinden yayına başlar.

---

## 🧠 Makine Öğrenmesi Hattı ve Metodoloji

Sistem veri temizleme adımından başlayarak tahmin çıktılarına kadar doğrusal bir boru hattı (pipeline) üzerinde çalışır:

### 1. Veri Önişleme (Data Preprocessing)
Ham e-ticaret fatura kayıtlarındaki gürültülü ve eksik veriler temizlenmiştir:
* İptal ve iade işlemlerini simgeleyen ve `InvoiceNo` değeri "C" ile başlayan faturalar filtrelenmiştir.
* `CustomerID` hücresi boş (NaN) olan satırlar veri kümesinden elenmiştir.
* `Quantity` (Miktar) ve `UnitPrice` (Birim Fiyat) değerleri sıfır veya negatif olan hatalı kayıtlar temizlenmiştir.

### 2. Özellik Mühendisliği (7 Boyutlu RFM & CLV Özellik Kümesi)
İşlemsel fatura satırları, müşteri bazında aşağıdaki 7 temel davranışsal özelliğe dönüştürülmüştür:
* **Recency:** Müşterinin son alışverişinden analiz tarihine kadar geçen gün sayısı.
* **Frequency:** Müşterinin gerçekleştirdiği tekil sipariş sayısı.
* **Monetary:** Müşterinin bıraktığı toplam ciro hacmi.
* **CLV (Customer Lifetime Value):** Yaşam Boyu Değer Skoru ($Monetary \times Frequency$).
* **Avg_Order_Value:** Müşterinin sipariş başına ortalama harcama tutarı.
* **Customer_Age:** Müşterinin sistemde aktif olduğu toplam gün sayısı.
* **Product_Variety:** Müşterinin satın aldığı benzersiz ürün çeşitliliği adedi.

### 3. K-Means Kümeleme ve Akıllı Segmentasyon
* Boyutsal büyüklük farklarının mesafe ölçümlerini etkilememesi için özellik seti `StandardScaler` ile ölçeklendirilmiştir.
* Optimal küme sayısı ($K=4$), Elbow (Dirsek) ve Silhouette katsayısı analizlerine göre belirlenmiştir.
* Kümelerin anlamsal olarak karıştırılmaması için RFM ağırlıklarına göre sıralama yapılmış ve şu sınıflara atanmıştır:
  * **VIP:** Sık gelen, yüksek ciro bırakan ve en yüksek CLV'ye sahip sadık kitle.
  * **Regular:** Standart frekansta alışveriş yapan ana müşteri tabanı.
  * **Low Value:** Düşük işlem hacmine sahip, seyrek gelen grup.
  * **Lost:** Uzun süredir işlem yapmamış pasif grup.

### 4. Zaman Bölümlemeli Doğrulama ve Sınıf Dengeli Random Forest
Gelecek bilgisinin geçmişe sızmasını (Data Leakage) önlemek amacıyla klasik rastgele bölme (Random Split) yerine **Zaman Bölümlemeli Doğrulama (Time-Split Validation)** uygulanmıştır. Veri kümesinin son 30 günü test periyodu olarak ayrılmıştır. 

Sınıf dağılımındaki dengesizliğe karşı modeller Random Forest algoritmasında `class_weight='balanced'` parametresiyle optimize edilmiştir:

| Tahmin Modeli | Doğruluk (Accuracy) | ROC-AUC | F1-Skoru |
| :--- | :---: | :---: | :---: |
| **Satın Alma Tahmin Modeli (30 Günlük)** | %71.01 | 0.731 | %58.89 |
| **Müşteri Churn Modeli (90 Günlük)** | %89.56 | 0.940 | %86.98 |

* **Skor Yorumu:** Churn tahmini, son geliş süresi (`Recency`) ile yüksek korelasyona sahip olduğu için karar sınırları belirgindir ve F1 skoru %87 seviyesindedir. Satın alma kararı ise yüksek varyanslı işlemler içerdiğinden veri örtüşmesi (feature overlap) fazladır ve F1 skoru %59 ile sınırlanmıştır. Bu metrikler, zaman sızıntısı engellendiği için gerçek dünya performansını birebir yansıtmaktadır.

---

## 🤖 Karar Destek ve Öneri Mantığı

Tahmin modellerinden üretilen olasılık skorları, kural tabanlı bir karar motoru ile iş aksiyonlarına dönüştürülür:
1. **Pasif Müşteri Reaktivasyonu (Churn Riski > %75):** Müşteri kaybı olasılığı kritik seviyededir. Sistem, akran öneri algoritmasından gelen ürünlerle birlikte özel reaktivasyon indirim teklifleri önerir.
2. **Satış Fırsatı (Satın Alma Eğilimi > %75):** Müşterinin satın alma iştahı yüksektir. İlgili segmentin en çok tercih ettiği fakat müşterinin henüz satın almadığı ürünler listelenerek anlık teklifler üretilir.
3. **VIP Koruma Protokolü:** VIP segmentindeki müşterilerin ilişkilerini korumak üzere özel sadakat programı kararları uygulanır.
4. **Stabil:** Genel müşteri profili için standart pazarlama takvimi uygulanmaya devam eder.

---

## 💻 Donanım ve Yazılım Gereksinimleri

### Yazılım Gereksinimleri
* **İşletim Sistemi:** Windows 10/11, macOS, Linux (tümü desteklenmektedir).
* **Python Sürümü:** Python 3.8 veya üzeri.
* **Web Tarayıcı:** Google Chrome, Mozilla Firefox, Microsoft Edge veya Safari (güncel sürümleri).

### Minimum Donanım Gereksinimleri
* **İşlemci (CPU):** Intel Core i3 / AMD Ryzen 3 (Paralel model eğitimi için çok çekirdekli işlemciler önerilir).
* **Bellek (RAM):** 8 GB RAM (Veri kümesinin tamamı bellekte işlendiği için).
* **Depolama:** ~200 MB kullanılabilir disk alanı (Veri setinin sıkıştırılmamış hali dahil).

---

## 🗄️ Veritabanı ve Çevresel Yapılandırma (.env) Durumu

* **Veritabanı:** Bu sistemde herhangi bir harici SQL veya NoSQL veritabanı motoru (PostgreSQL, MySQL, MongoDB vb.) **kullanılmamıştır.** Müşteri kayıtları, işlem hareketleri ve makine öğrenmesi özellikleri doğrudan Pandas DataFrame yapısı üzerinden **bellek içi (in-memory)** yönetilmektedir. Bu nedenle herhangi bir veritabanı kurulumu veya dump yükleme adımına ihtiyaç yoktur.
* **Model Dosyası:** Makine öğrenmesi modelleri sunucu her başlatıldığında ham veri üzerinden dinamik olarak anlık eğitilmektedir. Devasa boyutlu kalıcı model dosyalarına ihtiyaç yoktur.
* **.env (Çevresel Değişkenler):** Sistemde harici gizli anahtarlar, API tokenları veya veritabanı şifreleri bulunmadığı için `.env` veya `.env.example` yapılandırma dosyalarına gerek duyulmamıştır.

---

## 🔑 Arayüz Erişim Bilgileri (Demo)

Geliştirilen yönetici paneli ve analiz ekranları yerel ağda herkese açık olarak çalışacak şekilde tasarlanmıştır:
* **Giriş Bilgileri (Username/Password):** Arayüzde herhangi bir kullanıcı giriş (Login) yetkilendirme katmanı bulunmamaktadır. Tarayıcıdan `http://127.0.0.1:5000` adresine girildiğinde doğrudan dashboard ekranına erişim sağlanır.

---

## 📊 Veri Seti Erişimi ve Örnek Veri (Dataset & Sample Data)

### 1. Orijinal Veri Seti
Projede kullanılan orijinal veri setine Kaggle üzerinden erişebilirsiniz:
* **Veri Seti Linki:** [Kaggle - Online Retail Dataset](https://www.kaggle.com/datasets/ulrikthygepedersen/online-retail-dataset)
* **Boyut:** ~48 MB (541,909 satır fatura hareketi)

### 2. Örnek Veri Seti (`online_retail_sample.csv`)
Proje klasöründe yer alan **`online_retail_sample.csv`** dosyası, orijinal veri setinin ilk 500 satırını içeren küçültülmüş bir örnektir. 
* **Çevrimdışı Çalıştırma:** İnternet bağlantısı olmadığında veya sistemi hızlıca test etmek istediğinizde, bu dosyanın adını `online_retail.csv` olarak değiştirmeniz yeterlidir. Flask sunucusu dosyayı otomatik olarak tespit edip yerel üzerinden çalışacaktır.

### 3. Örnek Veri Yapısı (First 5 Rows Preview)
Veri setinin sütun yapısı ve ilk 5 satırının görünümü aşağıdaki gibidir:

| InvoiceNo | StockCode | Description | Quantity | InvoiceDate | UnitPrice | CustomerID | Country |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| 536365 | 85123A | WHITE HANGING HEART T-LIGHT HOLDER | 6 | 2010-12-01 08:26:00 | 2.55 | 17850.0 | United Kingdom |
| 536365 | 71053 | WHITE METAL LANTERN | 6 | 2010-12-01 08:26:00 | 3.39 | 17850.0 | United Kingdom |
| 536365 | 84406B | CREAM CUPID HEARTS COAT HANGER | 8 | 2010-12-01 08:26:00 | 2.75 | 17850.0 | United Kingdom |
| 536365 | 84029G | KNITTED UNION FLAG HOT WATER BOTTLE | 6 | 2010-12-01 08:26:00 | 3.39 | 17850.0 | United Kingdom |
| 536365 | 84029E | RED WOOLLY HOTTIE WHITE HEART. | 6 | 2010-12-01 08:26:00 | 3.39 | 17850.0 | United Kingdom |
