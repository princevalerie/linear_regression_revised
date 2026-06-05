import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
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
    """Mean Square Error — Equation (4) from paper"""
    n = len(y_true)
    return np.sum((y_true - y_pred) ** 2) / n

def r_squared(y_true, y_pred):
    """Coefficient of Determination R² — Equation (5) from paper"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def adj_r_squared(y_true, y_pred, n_features):
    """Adjusted R-squared — accounts for number of predictors"""
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
    """Min-Max Scaling — as recommended in Section 2.2 of the paper"""
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rng = maxs - mins
    rng[rng == 0] = 1
    return (X - mins) / rng, mins, maxs

# ─────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING
#  Following exact methodology from Section 2.2
# ─────────────────────────────────────────────

NUMERIC_COLS = ["area", "bedrooms", "bathrooms", "stories", "parking"]
BINARY_COLS = ["mainroad", "guestroom", "basement",
               "hotwaterheating", "airconditioning", "prefarea"]
ORDINAL_COLS = ["furnishingstatus"]
CATEGORICAL_COLS = BINARY_COLS + ORDINAL_COLS

FEATURE_COLS = ["area", "bedrooms", "bathrooms", "stories",
                "mainroad", "guestroom", "basement",
                "hotwaterheating", "airconditioning",
                "parking", "prefarea", "furnishingstatus"]

@st.cache_data
def load_and_prepare():
    """
    Load dataset and apply preprocessing as described in the paper:
    1. Numeric columns: impute missing values with median
    2. Categorical columns: drop rows with missing values
    3. Encode binary (yes/no) and ordinal (furnishing) columns
    Returns TWO DataFrames: with outliers and without outliers.
    """
    df = pd.read_csv("Housing.csv")

    # --- Step 1: Handle Missing Values (Section 2.2) ---
    # Numeric: impute with median
    for col in NUMERIC_COLS:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Categorical: drop rows with missing values
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df = df.dropna(subset=[col])

    # --- Step 2: Encode categorical variables ---
    for col in BINARY_COLS:
        df[col] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0})

    furnish_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
    df["furnishingstatus"] = df["furnishingstatus"].str.strip().str.lower().map(furnish_map)

    # Drop any remaining NaN (safety net)
    df = df.dropna()

    # --- Step 3: Prepare two versions ---
    df_with_outliers = df.copy()

    # Remove outliers on price using IQR (Section 2.2)
    df_no_outliers = remove_outliers_iqr(df.copy(), "price")

    return df_with_outliers, df_no_outliers

# ─────────────────────────────────────────────
#  BUILD MODELS
# ─────────────────────────────────────────────

def build_univariate(df: pd.DataFrame):
    """Univariate model: y = β0 + β1 * area + ε (Equation 2)"""
    X_raw = df[["area"]].values.astype(float)
    y = df["price"].values.astype(float)
    X_scaled, mins, maxs = min_max_scale(X_raw)
    X = np.hstack([np.ones((len(X_scaled), 1)), X_scaled])
    beta = ols_fit(X, y)
    y_pred = predict(X, beta)
    return beta, y_pred, y, mins, maxs

def build_multivariate(df: pd.DataFrame):
    """Multivariate model: y = β0 + β1*x1 + ... + βn*xn + ε (Equation 3)"""
    X_raw = df[FEATURE_COLS].values.astype(float)
    y = df["price"].values.astype(float)
    X_scaled, mins, maxs = min_max_scale(X_raw)
    X = np.hstack([np.ones((len(X_scaled), 1)), X_scaled])
    beta = ols_fit(X, y)
    y_pred = predict(X, beta)
    return beta, y_pred, y, mins, maxs, FEATURE_COLS

def build_xgboost(df: pd.DataFrame):
    """XGBoost model — as suggested in Section 4 (future research) [6, 16]"""
    X_raw = df[FEATURE_COLS].values.astype(float)
    y = df["price"].values.astype(float)
    X_scaled, mins, maxs = min_max_scale(X_raw)
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    return model, y_pred, y, mins, maxs, FEATURE_COLS

# ─────────────────────────────────────────────
#  MATPLOTLIB GLOBAL STYLE
# ─────────────────────────────────────────────

plt.rcParams.update({
    'figure.facecolor': '#0E1117',
    'axes.facecolor': '#1A1D23',
    'axes.edgecolor': '#333842',
    'axes.labelcolor': '#FAFAFA',
    'text.color': '#FAFAFA',
    'xtick.color': '#B0B8C1',
    'ytick.color': '#B0B8C1',
    'grid.color': '#2A2E35',
    'grid.alpha': 0.5,
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'legend.facecolor': '#1A1D23',
    'legend.edgecolor': '#333842',
    'legend.fontsize': 9,
})

# Color palette
C_BLUE    = "#60A5FA"
C_CYAN    = "#22D3EE"
C_GREEN   = "#34D399"
C_ORANGE  = "#FB923C"
C_RED     = "#F87171"
C_PURPLE  = "#A78BFA"
C_PINK    = "#F472B6"
C_YELLOW  = "#FBBF24"
C_TEAL    = "#2DD4BF"

# ─────────────────────────────────────────────
#  PLOT HELPERS
# ─────────────────────────────────────────────

def plot_figure1_pairwise(df):
    """Figure 1: Pairwise scatter plots."""
    cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
    fig, axes = plt.subplots(len(cols), len(cols), figsize=(14, 14))

    for i, col_i in enumerate(cols):
        for j, col_j in enumerate(cols):
            ax = axes[i][j]
            if i == j:
                ax.hist(df[col_i], bins=20, color=C_BLUE, edgecolor="#0E1117", alpha=0.85)
            else:
                ax.scatter(df[col_j], df[col_i], alpha=0.35, s=6, color=C_CYAN, edgecolors='none')

            if i == len(cols) - 1:
                ax.set_xlabel(col_j, fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(col_i, fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=5)

    fig.suptitle("Figure 1 · Pairwise Relationships",
                 fontsize=14, fontweight="bold", color=C_BLUE, y=1.01)
    plt.tight_layout()
    return fig

def plot_figure2_correlation(df):
    """Figure 2: Correlation Matrix heatmap."""
    num_cols = ["price", "area", "bedrooms", "bathrooms", "stories",
                "mainroad", "guestroom", "basement", "hotwaterheating",
                "airconditioning", "parking", "prefarea", "furnishingstatus"]
    data = df[num_cols].values.astype(float)
    corr = correlation_matrix(data)
    labels = num_cols

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr[i, j]
            color = "white" if abs(val) > 0.4 else "#CCCCCC"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color, fontweight='bold' if abs(val) > 0.4 else 'normal')
    ax.set_title("Figure 2 · Correlation Matrix", fontsize=14, fontweight="bold", color=C_BLUE, pad=15)
    plt.tight_layout()
    return fig

def plot_figure3_comparison(y_uni, y_pred_uni, r2_uni,
                            y_multi, y_pred_multi, r2_multi,
                            y_no_out, y_pred_no_out, r2_no_out,
                            y_xgb, y_pred_xgb, r2_xgb):
    """Figure 3: Four-panel predicted vs actual including XGBoost."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    datasets = [
        (y_uni, y_pred_uni, f"(A) Univariate\nR² = {r2_uni*100:.2f}%", C_BLUE),
        (y_multi, y_pred_multi, f"(B) Multivariate + Outliers\nR² = {r2_multi*100:.2f}%", C_GREEN),
        (y_no_out, y_pred_no_out, f"(C) Multivariate − Outliers\nR² = {r2_no_out*100:.2f}%", C_ORANGE),
        (y_xgb, y_pred_xgb, f"(D) XGBoost\nR² = {r2_xgb*100:.2f}%", C_PURPLE),
    ]

    for ax, (y_true, y_pred, title, color) in zip(axes, datasets):
        idx = np.arange(len(y_true))
        ax.plot(idx, y_true / 1e6, label="Actual", color=color, linewidth=1.0, alpha=0.7)
        ax.plot(idx, y_pred / 1e6, label="Predicted", color=C_RED,
                linewidth=1.0, linestyle="--", alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Sample Index", fontsize=9)
        ax.set_ylabel("Price (Millions)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 3 · Predicted vs Actual Values",
                 fontsize=14, fontweight="bold", color=C_BLUE, y=1.02)
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(y_true, y_pred, title, color=C_CYAN):
    """Single actual vs predicted plot."""
    fig, ax = plt.subplots(figsize=(9, 4))
    idx = np.arange(len(y_true))
    ax.fill_between(idx, y_true / 1e6, alpha=0.12, color=color)
    ax.plot(idx, y_true / 1e6, label="Actual", color=color, linewidth=1.2, alpha=0.85)
    ax.plot(idx, y_pred / 1e6, label="Predicted", color=C_ORANGE,
            linewidth=1.2, linestyle="--", alpha=0.85)
    ax.set_title(title, fontsize=13, fontweight="bold", color=C_BLUE)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Price (Millions)")
    ax.legend(framealpha=0.8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig

def plot_residuals(y_true, y_pred, title):
    """Residual analysis plot."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_pred / 1e6, residuals / 1e6, alpha=0.45, color=C_PURPLE, s=12, edgecolors='none')
    axes[0].axhline(0, color=C_RED, linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_xlabel("Predicted (Millions)")
    axes[0].set_ylabel("Residuals (Millions)")
    axes[0].set_title("Residuals vs Predicted", color=C_PURPLE)
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(residuals / 1e6, bins=30, color=C_TEAL, edgecolor="#0E1117", alpha=0.85)
    axes[1].set_xlabel("Residual (Millions)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution", color=C_TEAL)
    axes[1].grid(True, alpha=0.25)
    plt.suptitle(title, fontsize=12, fontweight="bold", color=C_BLUE)
    plt.tight_layout()
    return fig

def plot_feature_importance(values, labels, title, cmap_name="Blues"):
    """Horizontal bar chart for feature importance."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sorted_idx = np.argsort(values)
    cmap = plt.colormaps[cmap_name]
    colors = [cmap(0.3 + 0.6 * i / len(labels)) for i in range(len(labels))]
    ax.barh([labels[i] for i in sorted_idx], values[sorted_idx],
            color=[colors[i] for i in range(len(sorted_idx))],
            edgecolor='none', height=0.65)
    ax.set_xlabel("Relative Importance")
    ax.set_title(title, fontsize=13, fontweight="bold", color=C_BLUE)
    ax.grid(True, alpha=0.2, axis="x")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig

def interpret_r2(r2):
    if r2 >= 0.75:
        return "🟢 **Sangat Baik** — Model menjelaskan sebagian besar variasi harga."
    elif r2 >= 0.5:
        return "🟡 **Cukup Baik** — Model menangkap tren utama, namun masih ada variasi yang tidak terjelaskan."
    elif r2 >= 0.3:
        return "🟠 **Lemah** — Model hanya menjelaskan sebagian kecil variasi harga."
    else:
        return "🔴 **Sangat Lemah** — Model tidak cocok untuk data ini."

# ─────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="House Price Predictor — Linear Regression",
    page_icon="🏠",
    layout="wide"
)

# ── Hero Header (Default Streamlit Style) ──────────────────────
st.title("🏠 House Price Prediction")
st.markdown("""
**Implementasi Linear Regression dari scratch (OLS)**  
*Berdasarkan paper: Predicting House Prices with a Linear Regression Model — Liming Yan, 2024*
""")


# Load data — two versions
df_with, df_no = load_and_prepare()

# Build ALL models upfront
beta_uni, y_pred_uni, y_uni, mins_uni, maxs_uni = build_univariate(df_with)
beta_multi, y_pred_multi, y_multi, mins_multi, maxs_multi, feat_cols = build_multivariate(df_with)
beta_no, y_pred_no, y_no, mins_no, maxs_no, feat_cols_no = build_multivariate(df_no)
model_xgb, y_pred_xgb, y_xgb, mins_xgb, maxs_xgb, feat_cols_xgb = build_xgboost(df_with)

# Calculate ALL metrics
mse_uni = mse(y_uni, y_pred_uni)
r2_uni = r_squared(y_uni, y_pred_uni)
adj_r2_uni = adj_r_squared(y_uni, y_pred_uni, 1)

mse_multi = mse(y_multi, y_pred_multi)
r2_multi = r_squared(y_multi, y_pred_multi)
adj_r2_multi = adj_r_squared(y_multi, y_pred_multi, len(feat_cols))

mse_no = mse(y_no, y_pred_no)
r2_no = r_squared(y_no, y_pred_no)
adj_r2_no = adj_r_squared(y_no, y_pred_no, len(feat_cols_no))

mse_xgb = mse(y_xgb, y_pred_xgb)
r2_xgb = r_squared(y_xgb, y_pred_xgb)
adj_r2_xgb = adj_r_squared(y_xgb, y_pred_xgb, len(feat_cols_xgb))

# Create 7 separate tabs using default Streamlit styling
tabs = st.tabs([
    "📊 Dataset",
    "📈 Univariate",
    "📉 Multi + Outlier",
    "📉 Multi − Outlier",
    "🌲 XGBoost",
    "⚖️ Comparison",
    "🔮 Prediction"
])




# ── TAB 1: DATASET ──────────────────────────────────────────────
with tabs[0]:
    st.subheader("Kaggle Housing Prices Dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data", 545)
    c2.metric("Dengan Outlier", len(df_with))
    c3.metric("Tanpa Outlier", len(df_no))
    c4.metric("Jumlah Fitur", len(FEATURE_COLS))

    with st.container():
        st.markdown("##### 📋 Langkah Preprocessing (Section 2.2)")
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""
            **1. Missing Values**
            - Numerik → imputasi **median**
            - Kategorikal → baris **di-drop**
            
            **2. Encoding**
            - `yes/no` → `1/0`
            - `furnishingstatus`: furnished=2, semi=1, unfurnished=0
            """)
        with col_r:
            st.markdown("""
            **3. Outlier Removal**
            - Metode **IQR** pada kolom `price`
            - Q1 − 1.5×IQR ≤ keep ≤ Q3 + 1.5×IQR
            
            **4. Normalization**
            - **Min-Max Scaling**: `(x − min) / (max − min)`
            """)

    st.subheader("Sample Data (Table 1)")
    st.dataframe(df_with.head(15), use_container_width=True, height=350)

    st.divider()

    # Figure 1
    st.subheader("Figure 1 · Pairwise Relationships")
    st.caption("Hubungan antara harga rumah dan parameter. Area vs Price menunjukkan korelasi positif terkuat.")
    fig1 = plot_figure1_pairwise(df_with)
    st.pyplot(fig1)

    st.divider()

    # Figure 2
    st.subheader("Figure 2 · Correlation Matrix")
    col_l, col_r = st.columns([2, 1])
    with col_r:
        st.markdown("""
        **Temuan Utama:**
        | Pasangan | r |
        |---|---|
        | price ↔ area | **0.54** |
        | price ↔ bathrooms | **0.52** |
        | price ↔ AC | **0.45** |
        | price ↔ stories | **0.42** |
        | price ↔ furnishing | **−0.30** |
        
        *Least relevant: hotwaterheating*
        """)
    with col_l:
        fig2 = plot_figure2_correlation(df_with)
        st.pyplot(fig2)

    st.divider()

    # Distribusi Harga
    st.subheader("Distribusi Harga")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_d1, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(df_with["price"] / 1e6, bins=40, color=C_BLUE, edgecolor="#0E1117", alpha=0.85)
        ax.set_xlabel("Price (Millions)")
        ax.set_ylabel("Frequency")
        ax.set_title("Dengan Outlier", color=C_BLUE)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig_d1)
    with col_b:
        fig_d2, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(df_no["price"] / 1e6, bins=40, color=C_GREEN, edgecolor="#0E1117", alpha=0.85)
        ax.set_xlabel("Price (Millions)")
        ax.set_ylabel("Frequency")
        ax.set_title("Tanpa Outlier", color=C_GREEN)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig_d2)

# ── TAB 2: UNIVARIATE ───────────────────────────────────────────
with tabs[1]:
    st.subheader("Univariate Linear Regression")
    st.latex(r"y = \beta_0 + \beta_1 \times \text{area} + \epsilon \quad \text{(Equation 2)}")

    c1, c2, c3 = st.columns(3)
    c1.metric("MSE", f"{mse_uni:,.0f}")
    c2.metric("R²", f"{r2_uni*100:.2f}%")
    c3.metric("Adjusted R²", f"{adj_r2_uni:.4f}")
    st.markdown(interpret_r2(r2_uni))

    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown("**Koefisien Model**")
        st.table(pd.DataFrame({
            "Parameter": ["Intercept (β₀)", "Area (β₁)"],
            "Nilai": [f"{beta_uni[0]:,.2f}", f"{beta_uni[1]:,.2f}"]
        }))
    with col_r:
        fig_uni = plot_actual_vs_predicted(y_uni, y_pred_uni,
                                           f"Univariate · Actual vs Predicted (R² = {r2_uni*100:.2f}%)",
                                           color=C_BLUE)
        st.pyplot(fig_uni)

    with st.container():
        st.markdown("##### 📊 Analisis Residual")
        fig_res_uni = plot_residuals(y_uni, y_pred_uni, "Residual Analysis — Univariate")
        st.pyplot(fig_res_uni)

    with st.container():
        st.markdown("##### ℹ️ Interpretasi (Section 3)")
        st.markdown(f"""
        - R² = **{r2_uni*100:.2f}%** — Paper: **~34.58%**
        - Hanya ~{r2_uni*100:.1f}% variasi harga dijelaskan oleh area saja.
        - *"only 34.58% of the variation in house prices can be explained by area variation alone"*
        - Perlu model multivariate untuk akurasi lebih baik.
        """)


# ── TAB 3: MULTIVARIATE WITH OUTLIERS ────────────────────────────
with tabs[2]:
    st.subheader("Multivariate — Dengan Outlier")


    st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon \quad \text{(Equation 3)}")

    c1, c2, c3 = st.columns(3)
    c1.metric("MSE", f"{mse_multi:,.0f}")
    c2.metric("R²", f"{r2_multi*100:.2f}%", delta=f"+{(r2_multi - r2_uni)*100:.2f}% vs Uni")
    c3.metric("Adjusted R²", f"{adj_r2_multi:.4f}")
    st.markdown(interpret_r2(r2_multi))

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Koefisien Model**")
        st.dataframe(pd.DataFrame({
            "Fitur": ["Intercept"] + feat_cols,
            "β": [f"{b:,.2f}" for b in beta_multi],
        }), use_container_width=True, height=300)
    with col_r:
        beta_abs = np.abs(beta_multi[1:])
        imp_norm = beta_abs / beta_abs.sum()
        fig_imp = plot_feature_importance(imp_norm, feat_cols,
                                          "Feature Importance (|β| normalized)", "Blues")
        st.pyplot(fig_imp)

    st.subheader("Actual vs Predicted")
    fig_m = plot_actual_vs_predicted(y_multi, y_pred_multi,
                                     f"Multivariate + Outliers · R² = {r2_multi*100:.2f}%",
                                     color=C_GREEN)
    st.pyplot(fig_m)

    with st.container():
        st.markdown("##### 📊 Analisis Residual")
        fig_rm = plot_residuals(y_multi, y_pred_multi, "Residual — Multivariate + Outliers")
        st.pyplot(fig_rm)

    with st.container():
        st.markdown("##### ℹ️ Interpretasi (Section 3)")
        st.markdown(f"""
        - R² = **{r2_multi*100:.2f}%** — Paper: **~61.97%**
        - *"61.97% of the variance is explained by area, bathrooms, bedrooms, and stories"*
        - Peningkatan signifikan: {r2_uni*100:.2f}% → {r2_multi*100:.2f}%
        """)


# ── TAB 4: MULTIVARIATE WITHOUT OUTLIERS ─────────────────────────
with tabs[3]:
    st.subheader("Multivariate — Tanpa Outlier")


    st.caption("Outlier pada `price` dihapus via IQR. Section 2.2: *\"We discard outliers...\"*")

    c1, c2, c3 = st.columns(3)
    c1.metric("MSE", f"{mse_no:,.0f}")
    c2.metric("R²", f"{r2_no*100:.2f}%",
              delta=f"{(r2_no - r2_multi)*100:.2f}% vs +Outlier", delta_color="inverse")
    c3.metric("Adjusted R²", f"{adj_r2_no:.4f}")
    st.markdown(interpret_r2(r2_no))

    st.warning(f"""
    ⚠️ **Outlier Paradox:** R² turun dari **{r2_multi*100:.2f}%** → **{r2_no*100:.2f}%** setelah outlier dihapus.  
    Paper: *"predictions are much better on the graph, but scores drop from 61.97% to 54.97%"*  
    Penyebab: loss of information, model sensitivity, overfitting pada outlier.
    """)

    st.subheader("Actual vs Predicted")
    fig_no = plot_actual_vs_predicted(y_no, y_pred_no,
                                      f"Multivariate − Outliers · R² = {r2_no*100:.2f}%",
                                      color=C_ORANGE)
    st.pyplot(fig_no)

    col_l, col_r = st.columns(2)
    with col_l:
        with st.container():
            st.markdown("##### 📊 Analisis Residual")
            fig_rn = plot_residuals(y_no, y_pred_no, "Residual — Multivariate − Outliers")
            st.pyplot(fig_rn)
    with col_r:
        with st.container():
            st.markdown("##### 📋 Koefisien Model")
            st.dataframe(pd.DataFrame({
                "Fitur": ["Intercept"] + feat_cols_no,
                "β": [f"{b:,.2f}" for b in beta_no],
            }), use_container_width=True)

    with st.container():
        st.markdown("##### ℹ️ Interpretasi (Section 4)")
        st.markdown(f"""
        - R² turun karena: (1) Loss of information, (2) Model sensitivity, (3) Overfitting
        - Prediksi secara visual **lebih baik** meskipun skor turun
        - Data: {len(df_with)} → {len(df_no)} ({len(df_with) - len(df_no)} outlier dihapus)
        """)


# ── TAB 5: XGBOOST ──────────────────────────────────────────────
with tabs[4]:
    st.subheader("🌲 XGBoost Regression")


    st.caption("Bonus: Paper Section 4 menyebutkan *\"XGBoost is superior to other models\"* [6, 16]")

    c1, c2, c3 = st.columns(3)
    c1.metric("MSE", f"{mse_xgb:,.0f}",
              delta=f"{(mse_xgb - mse_multi)/mse_multi*100:.1f}% vs OLS", delta_color="inverse")
    c2.metric("R²", f"{r2_xgb*100:.2f}%",
              delta=f"+{(r2_xgb - r2_multi)*100:.2f}% vs OLS")
    c3.metric("Adjusted R²", f"{adj_r2_xgb:.4f}")
    st.markdown(interpret_r2(r2_xgb))

    col_l, col_r = st.columns(2)
    with col_l:
        xgb_imp = model_xgb.feature_importances_
        fig_xgb_imp = plot_feature_importance(xgb_imp, feat_cols_xgb,
                                               "XGBoost Feature Importance", "Greens")
        st.pyplot(fig_xgb_imp)
    with col_r:
        fig_xgb = plot_actual_vs_predicted(y_xgb, y_pred_xgb,
                                            f"XGBoost · R² = {r2_xgb*100:.2f}%",
                                            color=C_GREEN)
        st.pyplot(fig_xgb)

    with st.container():
        st.markdown("##### 📊 Analisis Residual")
        fig_rxgb = plot_residuals(y_xgb, y_pred_xgb, "Residual — XGBoost")
        st.pyplot(fig_rxgb)

    with st.container():
        st.markdown("##### ℹ️ Interpretasi")
        st.markdown(f"""
        - R² = **{r2_xgb*100:.2f}%** vs Multivariate OLS = {r2_multi*100:.2f}%
        - XGBoost menangkap hubungan **non-linear** dan **interaksi antar fitur**
        - ⚠️ Hasil pada seluruh dataset — R² tinggi bisa indikasi *overfitting*
        """)


# ── TAB 6: MODEL COMPARISON ─────────────────────────────────────
with tabs[5]:
    st.subheader("⚖️ Perbandingan Semua Model")

    comp_df = pd.DataFrame({
        "Model": [
            "📈 Univariate (area)",
            "📉 Multivariate + Outlier",
            "📉 Multivariate − Outlier",
            "🌲 XGBoost"
        ],
        "R² Paper": ["~34.58%", "~61.97%", "~54.97%", "—"],
        "R² Ours": [
            f"{r2_uni*100:.2f}%",
            f"{r2_multi*100:.2f}%",
            f"{r2_no*100:.2f}%",
            f"{r2_xgb*100:.2f}%"
        ],
        "MSE": [f"{mse_uni:,.0f}", f"{mse_multi:,.0f}", f"{mse_no:,.0f}", f"{mse_xgb:,.0f}"],
        "Adj R²": [f"{adj_r2_uni:.4f}", f"{adj_r2_multi:.4f}", f"{adj_r2_no:.4f}", f"{adj_r2_xgb:.4f}"],
        "N Data": [len(df_with), len(df_with), len(df_no), len(df_with)]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.divider()

    # Figure 3
    st.subheader("Figure 3 · Predicted vs Actual")
    fig3 = plot_figure3_comparison(
        y_uni, y_pred_uni, r2_uni,
        y_multi, y_pred_multi, r2_multi,
        y_no, y_pred_no, r2_no,
        y_xgb, y_pred_xgb, r2_xgb
    )

    st.pyplot(fig3)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🔍 Key Factors")
        st.markdown("""
        | Rank | Factor | r |
        |---|---|---|
        | 1 | Area | 0.54 |
        | 2 | Bathrooms | 0.52 |
        | 3 | Air Conditioning | 0.45 |
        | 4 | Stories | 0.42 |
        | 5 | Parking | — |
        """)
    with col_b:
        st.markdown("#### 📊 Conclusions")
        st.markdown(f"""
        - **Univariate** ({r2_uni*100:.2f}%): Area saja tidak cukup
        - **Multi + Outlier** ({r2_multi*100:.2f}%): Signifikan lebih baik
        - **Multi − Outlier** ({r2_no*100:.2f}%): Skor turun

        - **XGBoost** ({r2_xgb*100:.2f}%): Superior untuk non-linear
        """)

    with st.container():
        st.markdown("##### 📚 Limitasi & Future Work (Section 4)")
        st.markdown("""
        **Limitasi:** Asumsi linearitas, sensitif outlier, homoscedasticity, normalitas error  
        **Future:** Polynomial/log regression, robust regression, Random Forest, XGBoost, deep learning, real-time forecasting
        """)


# ── TAB 7: PREDIKSI MANUAL ──────────────────────────────────────
with tabs[6]:
    st.subheader("🔮 Prediksi Harga Rumah (Simulasi Semua Model)")

    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📐 Spesifikasi Fisik")
            area = st.number_input("Luas Area (sq ft)", min_value=1000, max_value=20000, value=5000, step=100)
            bedrooms = st.slider("Kamar Tidur", 1, 6, 3)
            bathrooms = st.slider("Kamar Mandi", 1, 4, 2)
            stories = st.slider("Lantai", 1, 4, 2)
            parking = st.slider("Parkir", 0, 3, 1)
            furnishing = st.radio("Furnitur", ["furnished", "semi-furnished", "unfurnished"], horizontal=True)

        with col2:
            st.markdown("##### 🏗️ Fasilitas")
            mainroad = st.radio("Akses Jalan Utama?", ["Ya", "Tidak"], horizontal=True)
            guestroom = st.radio("Kamar Tamu?", ["Ya", "Tidak"], horizontal=True)
            basement = st.radio("Basement?", ["Ya", "Tidak"], horizontal=True)
            hotwater = st.radio("Hot Water?", ["Ya", "Tidak"], horizontal=True)
            aircon = st.radio("AC?", ["Ya", "Tidak"], horizontal=True)
            prefarea = st.radio("Preferred Area?", ["Ya", "Tidak"], horizontal=True)
            
        submit_btn = st.form_submit_button("🔮 Mulai Prediksi", type="primary", use_container_width=True)

    if submit_btn:
        furnish_map_input = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
        yn = lambda x: 1 if x == "Ya" else 0

        input_raw = np.array([[
            area, bedrooms, bathrooms, stories,
            yn(mainroad), yn(guestroom), yn(basement),
            yn(hotwater), yn(aircon),
            parking, yn(prefarea), furnish_map_input[furnishing]
        ]], dtype=float)

        # --- Predict Model 0: Univariate ---
        input_raw_uni = np.array([[area]], dtype=float)
        mins_pred_uni, maxs_pred_uni = mins_uni, maxs_uni
        rng_pred_uni = maxs_pred_uni - mins_pred_uni
        rng_pred_uni[rng_pred_uni == 0] = 1
        input_scaled_uni = (input_raw_uni - mins_pred_uni) / rng_pred_uni
        
        beta_pred_uni = beta_uni
        input_with_intercept_uni = np.hstack([np.ones((1, 1)), input_scaled_uni])
        price_pred_uni = float((input_with_intercept_uni @ beta_pred_uni)[0])
        rmse_uni_val = np.sqrt(mse_uni)

        # --- Predict Model 1: Multivariate + Outlier ---
        mins_pred_multi, maxs_pred_multi = mins_multi, maxs_multi
        rng_pred_multi = maxs_pred_multi - mins_pred_multi
        rng_pred_multi[rng_pred_multi == 0] = 1
        input_scaled_multi = (input_raw - mins_pred_multi) / rng_pred_multi
        
        beta_pred_multi = beta_multi
        input_with_intercept_multi = np.hstack([np.ones((1, 1)), input_scaled_multi])
        price_pred_multi = float((input_with_intercept_multi @ beta_pred_multi)[0])
        rmse_multi_val = np.sqrt(mse_multi)

        # --- Predict Model 2: Multivariate − Outlier ---
        mins_pred_no, maxs_pred_no = mins_no, maxs_no
        rng_pred_no = maxs_pred_no - mins_pred_no
        rng_pred_no[rng_pred_no == 0] = 1
        input_scaled_no = (input_raw - mins_pred_no) / rng_pred_no
        
        beta_pred_no = beta_no
        input_with_intercept_no = np.hstack([np.ones((1, 1)), input_scaled_no])
        price_pred_no = float((input_with_intercept_no @ beta_pred_no)[0])
        rmse_no_val = np.sqrt(mse_no)

        # --- Predict Model 3: XGBoost ---
        mins_pred_xgb, maxs_pred_xgb = mins_xgb, maxs_xgb
        rng_pred_xgb = maxs_pred_xgb - mins_pred_xgb
        rng_pred_xgb[rng_pred_xgb == 0] = 1
        input_scaled_xgb = (input_raw - mins_pred_xgb) / rng_pred_xgb
        
        price_pred_xgb = float(model_xgb.predict(input_scaled_xgb)[0])
        rmse_xgb_val = np.sqrt(mse_xgb)

        st.divider()
        st.markdown("### 💰 Estimasi Harga dari Keempat Model")

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        avg_price = df_with["price"].mean()

        # Model 0 Card: Univariate
        with res_col1:
            st.markdown("##### 📈 Univariate (Area)")
            if price_pred_uni > 0:
                st.success(f"**{price_pred_uni:,.0f}**  \n({price_pred_uni/1e6:.3f} Juta)")
                pct = (price_pred_uni - avg_price) / avg_price * 100
                direction = "di atas" if pct > 0 else "di bawah"
                st.info(f"📊 **{abs(pct):.1f}%** {direction} rata-rata")
                st.metric("Model R²", f"{r2_uni*100:.2f}%")
                st.metric("RMSE", f"{rmse_uni_val:,.0f}")
                st.caption(f"Rentang: **{max(0, price_pred_uni - rmse_uni_val):,.0f}** — **{price_pred_uni + rmse_uni_val:,.0f}**")
            else:
                st.warning("⚠️ Estimasi negatif")

        # Model 1 Card: Multivariate + Outlier
        with res_col2:
            st.markdown("##### 📉 Multivariate + Outlier")
            if price_pred_multi > 0:
                st.success(f"**{price_pred_multi:,.0f}**  \n({price_pred_multi/1e6:.3f} Juta)")
                pct = (price_pred_multi - avg_price) / avg_price * 100
                direction = "di atas" if pct > 0 else "di bawah"
                st.info(f"📊 **{abs(pct):.1f}%** {direction} rata-rata")
                st.metric("Model R²", f"{r2_multi*100:.2f}%")
                st.metric("RMSE", f"{rmse_multi_val:,.0f}")
                st.caption(f"Rentang: **{max(0, price_pred_multi - rmse_multi_val):,.0f}** — **{price_pred_multi + rmse_multi_val:,.0f}**")
            else:
                st.warning("⚠️ Estimasi negatif")

        # Model 2 Card: Multivariate − Outlier
        with res_col3:
            st.markdown("##### 📉 Multivariate − Outlier")
            if price_pred_no > 0:
                st.success(f"**{price_pred_no:,.0f}**  \n({price_pred_no/1e6:.3f} Juta)")
                pct = (price_pred_no - avg_price) / avg_price * 100
                direction = "di atas" if pct > 0 else "di bawah"
                st.info(f"📊 **{abs(pct):.1f}%** {direction} rata-rata")
                st.metric("Model R²", f"{r2_no*100:.2f}%")
                st.metric("RMSE", f"{rmse_no_val:,.0f}")
                st.caption(f"Rentang: **{max(0, price_pred_no - rmse_no_val):,.0f}** — **{price_pred_no + rmse_no_val:,.0f}**")
            else:
                st.warning("⚠️ Estimasi negatif")

        # Model 3 Card: XGBoost
        with res_col4:
            st.markdown("##### 🌲 XGBoost")
            if price_pred_xgb > 0:
                st.success(f"**{price_pred_xgb:,.0f}**  \n({price_pred_xgb/1e6:.3f} Juta)")
                pct = (price_pred_xgb - avg_price) / avg_price * 100
                direction = "di atas" if pct > 0 else "di bawah"
                st.info(f"📊 **{abs(pct):.1f}%** {direction} rata-rata")
                st.metric("Model R²", f"{r2_xgb*100:.2f}%")
                st.metric("RMSE", f"{rmse_xgb_val:,.0f}")
                st.caption(f"Rentang: **{max(0, price_pred_xgb - rmse_xgb_val):,.0f}** — **{price_pred_xgb + rmse_xgb_val:,.0f}**")
            else:
                st.warning("⚠️ Estimasi negatif")

    st.divider()

    with st.container():
        st.markdown("##### ℹ️ Catatan Model")
        st.markdown("""
        - OLS dari scratch (tanpa sklearn) · Min-Max normalization · IQR outlier removal  
        - Metrik: MSE (Eq. 4), R² (Eq. 5), Adjusted R²  
        - Paper: *Predicting House Prices with a Linear Regression Model* (Liming Yan, 2024)
        """)

