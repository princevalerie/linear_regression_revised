# 🏠 House Price Prediction — Linear Regression from Scratch

Aplikasi Streamlit untuk memprediksi harga rumah menggunakan **Ordinary Least Squares (OLS)** yang diimplementasikan dari nol tanpa library machine learning.

> Berdasarkan paper: *Predicting House Prices with a Linear Regression Model* — Liming Yan, 2024  
> Mata Kuliah: Komputasi Numerik

---

## 📋 Fitur Aplikasi

| Tab | Deskripsi |
|-----|-----------|
| 📊 Dataset Overview | Pratinjau data, distribusi harga, correlation matrix |
| 📈 Univariate Regression | Model dengan satu fitur (area), metrik + grafik + interpretasi |
| 📉 Multivariate Regression | Model dengan semua fitur, feature importance, perbandingan model |
| 🔮 Prediksi Manual | Input spesifikasi rumah → estimasi harga real-time |

---

## 🧮 Implementasi Matematika

Semua fungsi diimplementasikan manual menggunakan **NumPy** saja:

- **OLS**: `β = (XᵀX)⁻¹ Xᵀy`
- **MSE**: `(1/n) Σ(yᵢ - ŷᵢ)²`
- **R²**: `1 - SS_res / SS_tot`
- **Adjusted R²**: `1 - (1-R²)(n-1)/(n-p-1)`
- **Pearson Correlation**: dihitung manual per-kolom
- **Outlier Removal**: IQR method pada kolom `price`
- **Normalization**: Min-Max scaling

---

## 🗂️ Struktur File

```
├── app.py              # Aplikasi Streamlit utama
├── requirements.txt    # Dependensi Python
├── Housing.csv         # Dataset (Kaggle Housing Prices Dataset)
└── README.md           # Dokumentasi ini
```

---

## 🚀 Cara Menjalankan

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Jalankan aplikasi
streamlit run app.py
```

Pastikan file `Housing.csv` berada di direktori yang sama dengan `app.py`.

---

## 📦 Dependensi

| Library | Versi | Kegunaan |
|---------|-------|----------|
| streamlit | 1.35.0 | Framework UI web |
| numpy | 1.26.4 | Komputasi matriks OLS |
| pandas | 2.2.2 | Manipulasi dataset |
| matplotlib | 3.9.0 | Visualisasi grafik |

> ⚠️ **Tidak menggunakan** scikit-learn, scipy, atau library ML lainnya.

---

## 📊 Pipeline Model (sesuai paper)

```
Housing.csv
    │
    ▼
Data Cleaning
├── Encode categorical (yes/no → 0/1, furnishingstatus → 0/1/2)
├── Drop NaN
└── Remove outliers (IQR pada price)
    │
    ▼
Min-Max Normalization
    │
    ├─── Univariate OLS (area only) ──→ Evaluate (MSE, R², Adj R²)
    │
    └─── Multivariate OLS (all features) ──→ Evaluate + Feature Importance
                                                │
                                                └─── Manual Prediction Input
```

---

## 📝 Referensi

Yan, L. (2024). *Predicting House Prices with a Linear Regression Model*. Proceedings of the 2nd International Conference on Machine Learning and Automation. DOI: 10.54254/2755-2721/114/2024.18220
