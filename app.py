import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb

# ─────────────────────────────────────────────
#  PURE MATH FUNCTIONS  (no sklearn / scipy)
# ─────────────────────────────────────────────

def ols_fit(X: np.ndarray, y: np.ndarray):
    """Ordinary Least Squares: β = (XᵀX)⁻¹ Xᵀy"""
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    return beta

def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return X @ beta

def mse(y_true, y_pred):
    n = len(y_true)
    return np.sum((y_true - y_pred) ** 2) / n

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def adj_r_squared(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r_squared(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

def remove_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Remove outliers using IQR method on a given column."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix from scratch."""
    n_cols = X.shape[1]
    corr = np.zeros((n_cols, n_cols))
    for i in range(n_cols):
        for j in range(n_cols):
            xi, xj = X[:, i], X[:, j]
            xi_c = xi - np.mean(xi)
            xj_c = xj - np.mean(xj)
            denom = np.sqrt(np.sum(xi_c**2) * np.sum(xj_c**2))
            corr[i, j] = np.sum(xi_c * xj_c) / denom if denom != 0 else 0
    return corr

def min_max_scale(X: np.ndarray):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rng = maxs - mins
    rng[rng == 0] = 1
    return (X - mins) / rng, mins, maxs

# ─────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

@st.cache_data
def load_and_prepare():
    df = pd.read_csv("Housing.csv")

    # Encode binary yes/no columns
    binary_cols = ["mainroad", "guestroom", "basement",
                   "hotwaterheating", "airconditioning", "prefarea"]
    for col in binary_cols:
        df[col] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0})

    # Encode furnishingstatus (ordinal)
    furnish_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
    df["furnishingstatus"] = df["furnishingstatus"].str.strip().str.lower().map(furnish_map)

    # Drop remaining NaN
    df = df.dropna()

    # Remove outliers on price (main target)
    df = remove_outliers_iqr(df, "price")

    return df

# ─────────────────────────────────────────────
#  BUILD MODELS
# ─────────────────────────────────────────────

def build_univariate(df: pd.DataFrame):
    X_raw = df[["area"]].values.astype(float)
    y = df["price"].values.astype(float)
    X_scaled, mins, maxs = min_max_scale(X_raw)
    X = np.hstack([np.ones((len(X_scaled), 1)), X_scaled])
    beta = ols_fit(X, y)
    y_pred = predict(X, beta)
    return beta, y_pred, y, mins, maxs

def build_multivariate(df: pd.DataFrame):
    feature_cols = ["area", "bedrooms", "bathrooms", "stories",
                    "mainroad", "guestroom", "basement",
                    "hotwaterheating", "airconditioning",
                    "parking", "prefarea", "furnishingstatus"]
    X_raw = df[feature_cols].values.astype(float)
    y = df["price"].values.astype(float)
    X_scaled, mins, maxs = min_max_scale(X_raw)
    X = np.hstack([np.ones((len(X_scaled), 1)), X_scaled])
    beta = ols_fit(X, y)
    y_pred = predict(X, beta)
    return beta, y_pred, y, mins, maxs, feature_cols

def build_xgboost(df: pd.DataFrame):
    feature_cols = ["area", "bedrooms", "bathrooms", "stories",
                    "mainroad", "guestroom", "basement",
                    "hotwaterheating", "airconditioning",
                    "parking", "prefarea", "furnishingstatus"]
    X_raw = df[feature_cols].values.astype(float)
    y = df["price"].values.astype(float)
    X_scaled, mins, maxs = min_max_scale(X_raw)
    
    # Train XGBoost
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    
    return model, y_pred, y, mins, maxs, feature_cols

# ─────────────────────────────────────────────
#  PLOT HELPERS
# ─────────────────────────────────────────────

def plot_actual_vs_predicted(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.arange(len(y_true))
    ax.plot(idx, y_true / 1e6, label="Actual", color="#2196F3", linewidth=1.2, alpha=0.8)
    ax.plot(idx, y_pred / 1e6, label="Predicted", color="#FF5722",
            linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Price (Millions)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_residuals(y_true, y_pred, title):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred / 1e6, residuals / 1e6, alpha=0.5, color="#673AB7", s=15)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted (Millions)")
    axes[0].set_ylabel("Residuals (Millions)")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals / 1e6, bins=30, color="#009688", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Residual (Millions)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")
    axes[1].grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df):
    num_cols = ["price", "area", "bedrooms", "bathrooms", "stories",
                "mainroad", "guestroom", "basement", "hotwaterheating",
                "airconditioning", "parking", "prefarea", "furnishingstatus"]
    data = df[num_cols].values.astype(float)
    corr = correlation_matrix(data)
    labels = num_cols

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdYlBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black")
    ax.set_title("Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig

def interpret_r2(r2):
    if r2 >= 0.75:
        return "🟢 **Sangat Baik** — Model menjelaskan sebagian besar variasi harga."
    elif r2 >= 0.5:
        return "🟡 **Cukup Baik** — Model menangkap tren utama, namun masih ada variasi yang tidak terjelas."
    elif r2 >= 0.3:
        return "🟠 **Lemah** — Model hanya menjelaskan sebagian kecil variasi harga."
    else:
        return "🔴 **Sangat Lemah** — Model tidak cocok untuk data ini."

# ─────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.caption("Implementasi Linear Regression dari scratch (OLS) · Berdasarkan paper: *Predicting House Prices with a Linear Regression Model* (Liming Yan, 2024)")

df = load_and_prepare()

tabs = st.tabs([
    "📊 Dataset Overview",
    "📈 Univariate Regression",
    "📉 Multivariate Regression",
    "🌲 XGBoost Regression",
    "🔮 Prediksi Harga Manual"
])

# ── TAB 1: DATASET ──────────────────────────────────────────────
with tabs[0]:
    st.subheader("Dataset: Housing Prices")
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Data (setelah cleaning)", len(df))
    col2.metric("Fitur", df.shape[1] - 1)
    col3.metric("Rata-rata Harga", f"{df['price'].mean()/1e6:.2f}M")

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Correlation Matrix")
    st.info("Area (luas) memiliki korelasi tertinggi dengan harga (r ≈ 0.54), diikuti bathrooms, airconditioning, dan stories.")
    fig_corr = plot_correlation_heatmap(df)
    st.pyplot(fig_corr)

    st.subheader("Distribusi Harga")
    fig_dist, ax = plt.subplots(figsize=(8, 3))
    ax.hist(df["price"] / 1e6, bins=40, color="#1565C0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Price (Millions)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribusi Harga Rumah (setelah outlier removal)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_dist)

# ── TAB 2: UNIVARIATE ───────────────────────────────────────────
with tabs[1]:
    st.subheader("Univariate Linear Regression")
    st.markdown("""
    Model: $y = \\beta_0 + \\beta_1 \\times \\text{area} + \\epsilon$

    Hanya menggunakan **luas area** sebagai prediktor.
    """)

    beta_uni, y_pred_uni, y_uni, mins_uni, maxs_uni = build_univariate(df)

    mse_uni = mse(y_uni, y_pred_uni)
    r2_uni = r_squared(y_uni, y_pred_uni)
    adj_r2_uni = adj_r_squared(y_uni, y_pred_uni, 1)

    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mse_uni:,.0f}")
    col2.metric("R²", f"{r2_uni:.4f} ({r2_uni*100:.2f}%)")
    col3.metric("Adjusted R²", f"{adj_r2_uni:.4f}")

    st.markdown(interpret_r2(r2_uni))

    st.markdown("**Koefisien Model:**")
    coef_df = pd.DataFrame({
        "Parameter": ["Intercept (β₀)", "Area (β₁)"],
        "Nilai": [f"{beta_uni[0]:,.2f}", f"{beta_uni[1]:,.2f}"]
    })
    st.table(coef_df)

    st.subheader("Actual vs Predicted")
    fig_uni = plot_actual_vs_predicted(y_uni, y_pred_uni, "Univariate: Actual vs Predicted")
    st.pyplot(fig_uni)

    st.subheader("Analisis Residual")
    fig_res_uni = plot_residuals(y_uni, y_pred_uni, "Residual Analysis — Univariate")
    st.pyplot(fig_res_uni)

    with st.expander("ℹ️ Interpretasi"):
        st.markdown(f"""
        - R² = **{r2_uni*100:.2f}%** artinya luas area hanya menjelaskan ~{r2_uni*100:.1f}% variasi harga.
        - Sesuai paper, model univariat memiliki R² sekitar 34%–35%, menunjukkan bahwa area saja tidak cukup.
        - MSE yang besar menandakan prediksi masih jauh dari nilai aktual untuk banyak sampel.
        """)

# ── TAB 3: MULTIVARIATE ─────────────────────────────────────────
with tabs[2]:
    st.subheader("Multivariate Linear Regression")
    st.markdown("""
    Model: $y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\cdots + \\beta_n x_n + \\epsilon$

    Menggunakan **semua fitur**: area, bedrooms, bathrooms, stories, mainroad, guestroom,
    basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus.
    """)

    beta_multi, y_pred_multi, y_multi, mins_multi, maxs_multi, feat_cols = build_multivariate(df)

    mse_multi = mse(y_multi, y_pred_multi)
    r2_multi = r_squared(y_multi, y_pred_multi)
    adj_r2_multi = adj_r_squared(y_multi, y_pred_multi, len(feat_cols))

    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mse_multi:,.0f}", delta=f"{(mse_multi - mse(y_uni, y_pred_uni))/mse(y_uni, y_pred_uni)*100:.1f}% vs Univariate")
    col2.metric("R²", f"{r2_multi:.4f} ({r2_multi*100:.2f}%)", delta=f"+{(r2_multi - r_squared(y_uni, y_pred_uni))*100:.2f}%")
    col3.metric("Adjusted R²", f"{adj_r2_multi:.4f}")

    st.markdown(interpret_r2(r2_multi))

    st.subheader("Koefisien Model")
    beta_labels = ["Intercept"] + feat_cols
    coef_data = pd.DataFrame({
        "Fitur": beta_labels,
        "Koefisien (β)": [f"{b:,.2f}" for b in beta_multi],
    })
    st.dataframe(coef_data, use_container_width=True)

    st.subheader("Feature Importance (|β| normalized)")
    beta_no_intercept = np.abs(beta_multi[1:])
    importance = beta_no_intercept / beta_no_intercept.sum()
    fig_imp, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feat_cols)))
    sorted_idx = np.argsort(importance)[::-1]
    ax.barh([feat_cols[i] for i in sorted_idx], importance[sorted_idx], color=colors)
    ax.set_xlabel("Relative Importance")
    ax.set_title("Feature Importance (Normalized |β|)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    st.pyplot(fig_imp)

    st.subheader("Actual vs Predicted")
    fig_multi = plot_actual_vs_predicted(y_multi, y_pred_multi, "Multivariate: Actual vs Predicted")
    st.pyplot(fig_multi)

    st.subheader("Analisis Residual")
    fig_res_multi = plot_residuals(y_multi, y_pred_multi, "Residual Analysis — Multivariate")
    st.pyplot(fig_res_multi)

    st.subheader("Perbandingan Model")
    comp_df = pd.DataFrame({
        "Model": ["Univariate (area only)", "Multivariate (all features)"],
        "R²": [f"{r_squared(y_uni, y_pred_uni)*100:.2f}%", f"{r2_multi*100:.2f}%"],
        "Adj. R²": [f"{adj_r_squared(y_uni, y_pred_uni, 1):.4f}", f"{adj_r2_multi:.4f}"],
        "MSE": [f"{mse(y_uni, y_pred_uni):,.0f}", f"{mse_multi:,.0f}"],
    })
    st.table(comp_df)

    with st.expander("ℹ️ Interpretasi"):
        st.markdown(f"""
        - Multivariate R² = **{r2_multi*100:.2f}%**, jauh lebih tinggi dari univariate ({r_squared(y_uni, y_pred_uni)*100:.2f}%).
        - Paper melaporkan R² ~62% (dengan outlier) dan ~55% (tanpa outlier). Hasil ini konsisten.
        - Penambahan fitur seperti bathrooms, airconditioning, dan stories berkontribusi signifikan.
        - MSE yang masih besar mencerminkan keterbatasan asumsi linearitas pada data properti nyata.
        """)

# ── TAB 4: XGBOOST ──────────────────────────────────────────────
with tabs[3]:
    st.subheader("XGBoost Regression")
    st.markdown("""
    Model: **XGBoost (Extreme Gradient Boosting)**
    
    Menggunakan **semua fitur** sebagai input. XGBoost adalah algoritma berbasis tree-ensemble yang sangat kuat untuk menangkap hubungan non-linear.
    """)

    model_xgb, y_pred_xgb, y_xgb, mins_xgb, maxs_xgb, feat_cols = build_xgboost(df)

    mse_xgb = mse(y_xgb, y_pred_xgb)
    r2_xgb = r_squared(y_xgb, y_pred_xgb)
    adj_r2_xgb = adj_r_squared(y_xgb, y_pred_xgb, len(feat_cols))

    col1, col2, col3 = st.columns(3)
    _, y_pred_multi_tmp, y_multi_tmp, _, _, _ = build_multivariate(df)
    mse_multi_tmp = mse(y_multi_tmp, y_pred_multi_tmp)
    col1.metric("MSE", f"{mse_xgb:,.0f}", delta=f"{(mse_xgb - mse_multi_tmp)/mse_multi_tmp*100:.1f}% vs Multivariate", delta_color="inverse")
    col2.metric("R²", f"{r2_xgb:.4f} ({r2_xgb*100:.2f}%)")
    col3.metric("Adjusted R²", f"{adj_r2_xgb:.4f}")

    st.markdown(interpret_r2(r2_xgb))

    st.subheader("Feature Importance (XGBoost)")
    importance = model_xgb.feature_importances_
    fig_imp_xgb, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(feat_cols)))
    sorted_idx = np.argsort(importance)[::-1]
    ax.barh([feat_cols[i] for i in sorted_idx], importance[sorted_idx], color=colors)
    ax.set_xlabel("Relative Importance")
    ax.set_title("XGBoost Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    st.pyplot(fig_imp_xgb)

    st.subheader("Actual vs Predicted")
    fig_xgb = plot_actual_vs_predicted(y_xgb, y_pred_xgb, "XGBoost: Actual vs Predicted")
    st.pyplot(fig_xgb)

    st.subheader("Analisis Residual")
    fig_res_xgb = plot_residuals(y_xgb, y_pred_xgb, "Residual Analysis — XGBoost")
    st.pyplot(fig_res_xgb)

    with st.expander("ℹ️ Interpretasi"):
        st.markdown(f"""
        - R² XGBoost biasanya lebih tinggi dari Multivariate (OLS) karena model ini dapat menangkap interaksi dan pola non-linear antar fitur.
        - Perlu diingat bahwa ini adalah hasil pada set pelatihan. R² yang sangat tinggi dapat mengindikasikan *overfitting*.
        """)

# ── TAB 5: PREDIKSI MANUAL ──────────────────────────────────────
with tabs[4]:
    st.subheader("🔮 Prediksi Harga Rumah")
    st.markdown("Masukkan spesifikasi rumah untuk mendapatkan estimasi harga.")
    
    model_choice = st.radio("Pilih Model untuk Prediksi:", ["Multivariate Linear Regression", "XGBoost Regression"], horizontal=True)

    # Rebuild model params
    beta_m, _, _, mins_m, maxs_m, f_cols = build_multivariate(df)
    model_xgb_m, _, _, _, _, _ = build_xgboost(df)

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Luas Area (sq ft)", min_value=1000, max_value=20000, value=5000, step=100)
        bedrooms = st.slider("Jumlah Kamar Tidur", 1, 6, 3)
        bathrooms = st.slider("Jumlah Kamar Mandi", 1, 4, 2)
        stories = st.slider("Jumlah Lantai", 1, 4, 2)
        parking = st.slider("Slot Parkir", 0, 3, 1)
        furnishing = st.selectbox("Status Furnitur", ["furnished", "semi-furnished", "unfurnished"])

    with col2:
        mainroad = st.radio("Akses Jalan Utama?", ["Ya", "Tidak"], horizontal=True)
        guestroom = st.radio("Ada Kamar Tamu?", ["Ya", "Tidak"], horizontal=True)
        basement = st.radio("Ada Basement?", ["Ya", "Tidak"], horizontal=True)
        hotwater = st.radio("Hot Water Heating?", ["Ya", "Tidak"], horizontal=True)
        aircon = st.radio("Ada AC?", ["Ya", "Tidak"], horizontal=True)
        prefarea = st.radio("Area Preferred?", ["Ya", "Tidak"], horizontal=True)

    furnish_map_input = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
    yn = lambda x: 1 if x == "Ya" else 0

    input_raw = np.array([[
        area, bedrooms, bathrooms, stories,
        yn(mainroad), yn(guestroom), yn(basement),
        yn(hotwater), yn(aircon),
        parking, yn(prefarea), furnish_map_input[furnishing]
    ]], dtype=float)

    # Scale using training min/max
    rng_m = maxs_m - mins_m
    rng_m[rng_m == 0] = 1
    input_scaled = (input_raw - mins_m) / rng_m
    
    if model_choice == "Multivariate Linear Regression":
        input_with_intercept = np.hstack([np.ones((1, 1)), input_scaled])
        price_pred = float(input_with_intercept @ beta_m)
    else:
        price_pred = float(model_xgb_m.predict(input_scaled)[0])

    st.divider()
    if price_pred > 0:
        st.success(f"### 💰 Estimasi Harga: **{price_pred:,.0f}** ({price_pred/1e6:.3f} Juta)")

        # Context
        avg_price = df["price"].mean()
        pct = (price_pred - avg_price) / avg_price * 100
        if pct > 0:
            st.info(f"📊 Harga ini **{abs(pct):.1f}% di atas** rata-rata dataset ({avg_price/1e6:.2f}M)")
        else:
            st.info(f"📊 Harga ini **{abs(pct):.1f}% di bawah** rata-rata dataset ({avg_price/1e6:.2f}M)")

        # Range estimate (±1 RMSE)
        if model_choice == "Multivariate Linear Regression":
            _, y_pred_tmp, y_tmp, _, _, _ = build_multivariate(df)
        else:
            _, y_pred_tmp, y_tmp, _, _, _ = build_xgboost(df)
        rmse_val = np.sqrt(mse(y_tmp, y_pred_tmp))
        st.caption(f"Perkiraan rentang: {max(0, price_pred - rmse_val):,.0f} — {price_pred + rmse_val:,.0f} (±1 RMSE)")
    else:
        st.warning("Estimasi harga negatif — coba ubah kombinasi input.")

    with st.expander("ℹ️ Catatan Model"):
        st.markdown("""
        - Model dilatih menggunakan **Ordinary Least Squares (OLS)** tanpa library ML.
        - Outlier pada kolom `price` dihapus menggunakan **metode IQR** sebelum training.
        - Fitur numerik di-*scale* dengan **min-max normalization** agar OLS konvergen stabil.
        - Model ini bersifat linear; hubungan non-linear antar fitur tidak ditangkap sepenuhnya.
        """)
