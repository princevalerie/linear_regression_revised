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
| 📉 Multivariate + Outlier | Model semua fitur dengan outlier |
| 📉 Multivariate − Outlier | Model semua fitur tanpa outlier |
| 🌲 XGBoost | Bonus: model non-linear untuk perbandingan |
| ⚖️ Comparison | Perbandingan keempat model |
| 🔮 Prediksi Manual | Input spesifikasi rumah → estimasi harga |
| 🤖 Ask AI | Chatbot AI — tanya tentang dataset, kode, paper |

---

## 🤖 Ask AI — NVIDIA AI Chatbot

Tab Ask AI menggunakan **ChatNVIDIA** via LangChain untuk menjawab pertanyaan tentang proyek ini.

**Model:** `moonshotai/kimi-k2.6` via NVIDIA AI Endpoints

**Konteks RAG otomatis:**
- `Housing.csv` — dataset lengkap
- `main.py` — source code aplikasi
- `ask_ai_tab.py` — source code tab AI
- `README.md` — dokumentasi
- `*.pdf` — paper (teks diekstrak via PyPDF2)
- Output chart labels dari aplikasi

### Setup API Key

```bash
# Buat file .env di root folder:
NVIDIA_API_KEY=nvapi-...
```

Dapatkan API key di [build.nvidia.com](https://build.nvidia.com)

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
├── main.py             # Aplikasi Streamlit utama (8 tab)
├── ask_ai_tab.py       # Tab Ask AI — ChatNVIDIA RAG chatbot
├── requirements.txt    # Dependensi Python
├── Housing.csv         # Dataset (Kaggle Housing Prices Dataset)
├── *.pdf               # Paper referensi
├── .env                # API key (NVIDIA_API_KEY=nvapi-...)
└── README.md           # Dokumentasi ini
```

---

## 🚀 Cara Menjalankan

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Setup API key (untuk fitur Ask AI)
echo "NVIDIA_API_KEY=nvapi-..." > .env

# 3. Jalankan aplikasi
streamlit run main.py
```

---

## 📦 Dependensi

| Library | Kegunaan |
|---------|----------|
| streamlit | Framework UI web |
| numpy | Komputasi matriks OLS |
| pandas | Manipulasi dataset |
| matplotlib | Visualisasi grafik |
| seaborn | Visualisasi tambahan |
| xgboost | Model XGBoost |
| langchain-nvidia-ai-endpoints | ChatNVIDIA LLM |
| PyPDF2 | Ekstraksi teks dari PDF |
| python-dotenv | Baca file .env |

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
    ├─── Multivariate OLS (all features) ──→ Evaluate + Feature Importance
    │
    └─── XGBoost ──→ Evaluate + Feature Importance
                        │
                        └─── Manual Prediction Input
```

---

## 📝 Referensi

Yan, L. (2024). *Predicting House Prices with a Linear Regression Model*. Proceedings of the 2nd International Conference on Machine Learning and Automation. DOI: 10.54254/2755-2721/114/2024.18220
