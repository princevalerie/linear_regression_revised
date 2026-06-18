"""
ask_ai_tab.py — ChatNVIDIA RAG Chatbot (LangChain)
Model: moonshotai/kimi-k2.6 via NVIDIA AI Endpoints
"""

import os, glob, textwrap
import streamlit as st

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser
    LLM_READY = True
except ImportError:
    LLM_READY = False

try:
    from PyPDF2 import PdfReader
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None

def _pdf_text(path, max_pages=30):
    if not PDF_OK:
        return None
    try:
        reader = PdfReader(path)
        out = []
        for i, p in enumerate(reader.pages[:max_pages]):
            t = p.extract_text()
            if t:
                out.append(f"[Page {i+1}]\n{t}")
        return "\n\n".join(out) if out else None
    except Exception:
        return None

def capture_chart(fig, label=""):
    """Dipanggil dari main.py — simpan label chart untuk konteks AI."""
    if "ai_charts" not in st.session_state:
        st.session_state.ai_charts = []
    if len(st.session_state.ai_charts) >= 20:
        st.session_state.ai_charts.pop(0)
    st.session_state.ai_charts.append(label or f"Chart {len(st.session_state.ai_charts)+1}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RAG SYSTEM PROMPT (cached)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=300)
def _build_system_prompt():
    parts = [textwrap.dedent("""\
    Kamu adalah AI assistant expert untuk proyek "House Price Prediction — Linear Regression from Scratch".
    Kamu memiliki akses PENUH ke dataset, source code, dokumentasi, dan paper.

    KEMAMPUAN:
    • Analisis dataset Housing.csv (545 baris, 13 kolom harga rumah)
    • Jelaskan implementasi OLS, MSE, R² dari source code
    • Bandingkan 4 model: Univariate, Multivariate ± Outlier, XGBoost
    • Debug code Python/Streamlit
    • Jawab tentang paper Liming Yan 2024

    ATURAN:
    • Jawab dalam Bahasa Indonesia kecuali diminta Bahasa Inggris
    • Gunakan markdown · LaTeX dalam $...$ jika relevan
    • Ringkas, informatif, akurat""")]

    csv = _read("Housing.csv")
    if csv:
        lines = csv.strip().split("\n")
        parts.append(f"---\n## [DATA] Housing.csv ({len(lines)-1} rows)\n```csv\n{chr(10).join(lines[:31])}\n```")

    for fname in ["main.py", "ask_ai_tab.py"]:
        code = _read(fname)
        if code:
            s = code[:15000] + ("\n…[truncated]" if len(code) > 15000 else "")
            parts.append(f"---\n## [CODE] {fname}\n```python\n{s}\n```")

    readme = _read("README.md")
    if readme:
        parts.append(f"---\n## [DOC] README.md\n{readme}")

    seen = set()
    for p in (glob.glob("*.pdf") + glob.glob("**/*.pdf", recursive=False))[:2]:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        txt = _pdf_text(p)
        if txt:
            txt = txt[:12000] + ("\n…[truncated]" if len(txt) > 12000 else "")
            parts.append(f"---\n## [PAPER] {os.path.basename(p)}\n{txt}")

    return "\n\n".join(parts)


def _build_metrics_context():
    """Build metrics block from session_state — dinamis, tidak di-cache."""
    m = st.session_state.get("ai_metrics")
    if not m:
        return ""

    def _pct(v): return f"{v*100:.2f}%"
    def _num(v): return f"{v:,.0f}"
    def _f4(v):  return f"{v:.4f}"

    lines = [
        "---",
        "## [METRICS] Hasil Evaluasi Model (Real-time dari Aplikasi)",
        "",
        "### Dataset",
        f"| Info | Nilai |",
        f"|---|---|",
        f"| Total data asli | {m['n_total']} baris |",
        f"| Data dengan outlier | {m['n_with_outliers']} baris |",
        f"| Data tanpa outlier | {m['n_without_outliers']} baris |",
        f"| Outlier dihapus | {m['n_outliers_removed']} baris |",
        f"| Jumlah fitur | {m['n_features']} fitur |",
        f"| Fitur | {', '.join(m['feature_cols'])} |",
        f"| Price mean | {_num(m['price_mean'])} |",
        f"| Price median | {_num(m['price_median'])} |",
        f"| Price std | {_num(m['price_std'])} |",
        f"| Price min | {_num(m['price_min'])} |",
        f"| Price max | {_num(m['price_max'])} |",
        f"| Area mean | {_num(m['area_mean'])} sq ft |",
        f"| Area median | {_num(m['area_median'])} sq ft |",
        "",
        "### Evaluasi Keempat Model",
        f"| Model | MSE | RMSE | R² |",
        f"|---|---|---|---|",
        f"| Univariate (area only) | {_num(m['uni_mse'])} | {_num(m['uni_rmse'])} | {_pct(m['uni_r2'])} |",
        f"| Multivariate + Outlier | {_num(m['multi_mse'])} | {_num(m['multi_rmse'])} | {_pct(m['multi_r2'])} |",
        f"| Multivariate - Outlier | {_num(m['no_mse'])} | {_num(m['no_rmse'])} | {_pct(m['no_r2'])} |",
        f"| XGBoost | {_num(m['xgb_mse'])} | {_num(m['xgb_rmse'])} | {_pct(m['xgb_r2'])} |",
        "",
        "### Koefisien OLS — Univariate",
        f"- β₀ (intercept): {m['uni_beta'][0]:,.2f}",
        f"- β₁ (area):      {m['uni_beta'][1]:,.2f}",
        "",
        "### Koefisien OLS — Multivariate + Outlier",
        "| Fitur | β (koefisien) | Importance (|β| norm) |",
        "|---|---|---|",
    ]
    for feat in ["intercept"] + m["feature_cols"]:
        beta_val = m["multi_beta"].get(feat, 0)
        imp_val  = m["multi_importance"].get(feat, "-")
        imp_str  = f"{imp_val:.4f}" if isinstance(imp_val, float) else "-"
        lines.append(f"| {feat} | {beta_val:,.2f} | {imp_str} |")

    lines += [
        "",
        "### Koefisien OLS — Multivariate - Outlier",
        "| Fitur | β (koefisien) | Importance (|β| norm) |",
        "|---|---|---|",
    ]
    for feat in ["intercept"] + m["feature_cols"]:
        beta_val = m["no_beta"].get(feat, 0)
        imp_val  = m["no_importance"].get(feat, "-")
        imp_str  = f"{imp_val:.4f}" if isinstance(imp_val, float) else "-"
        lines.append(f"| {feat} | {beta_val:,.2f} | {imp_str} |")

    lines += [
        "",
        "### XGBoost Feature Importance",
        "| Fitur | Importance Score |",
        "|---|---|",
    ]
    xgb_imp_sorted = sorted(m["xgb_importance"].items(), key=lambda x: x[1], reverse=True)
    for feat, val in xgb_imp_sorted:
        lines.append(f"| {feat} | {val:.4f} |")

    lines += [
        "",
        "### Korelasi Price vs Fitur (Pearson r)",
        "| Fitur | r (korelasi dengan price) |",
        "|---|---|",
    ]
    corr_sorted = sorted(m["correlations"].items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, val in corr_sorted:
        lines.append(f"| {feat} | {val:.4f} |")

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource
def _llm(api_key):
    return ChatNVIDIA(
        model="moonshotai/kimi-k2.6",
        api_key=api_key,
        temperature=1,
        top_p=1,
        max_tokens=16384,
        streaming=True,   # ← token-by-token streaming
    )

def _to_lc(system_prompt, history, user_input):
    msgs = [SystemMessage(content=system_prompt)]
    for m in history:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        msgs.append(cls(content=m["content"]))
    msgs.append(HumanMessage(content=user_input))
    return msgs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  QUICK QUESTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUICK_QS = [
    "Jelaskan hasil R² keempat model",
    "Bagaimana OLS diimplementasikan?",
    "Kenapa R² turun tanpa outlier?",
    "Bandingkan XGBoost vs OLS",
    "Apa kesimpulan paper?",
    "Korelasi price vs area",
    "Jelaskan preprocessing data",
    "Feature importance mana tertinggi?",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CSS = """
<style>
/* ── Scrollable chat area ───────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"]:has(
    > div > div > div > div[data-testid="stChatMessage"]
) {
    background: rgba(8, 10, 18, 0.35);
    border: 1px solid rgba(55, 65, 81, 0.25);
    border-radius: 14px;
}

/* ── Message rows ───────────────────────────── */
[data-testid="stChatMessage"] {
    padding: 10px 16px !important;
    margin: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    border-bottom: 1px solid rgba(55, 65, 81, 0.12) !important;
}
[data-testid="stChatMessage"]:last-child {
    border-bottom: none !important;
}

/* ── Chat input bar — sticky bottom look ────── */
[data-testid="stChatInput"] {
    border-top: 1px solid rgba(55, 65, 81, 0.2);
    padding-top: 6px;
}
[data-testid="stChatInput"] textarea {
    border-radius: 22px !important;
    font-size: 0.9em !important;
}

/* ── Welcome ────────────────────────────────── */
.ai-hi {
    text-align: center;
    padding: 48px 20px 20px;
    opacity: 0.85;
}
.ai-hi h3 {
    margin: 0 0 6px;
    font-size: 1.3em;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.ai-hi p {
    color: #64748b;
    font-size: 0.85em;
    margin: 2px 0;
}
.ctx-row {
    display: flex; justify-content: center;
    gap: 8px; margin-top: 12px; flex-wrap: wrap;
}
.ctx-tag {
    font-size: 0.7em; padding: 2px 9px;
    border-radius: 9px; color: #94a3b8;
    background: rgba(100,116,139,0.08);
    display: inline-flex; align-items: center; gap: 4px;
}
.ctx-tag .d { width:5px; height:5px; border-radius:50%; display:inline-block; }
.dg { background:#34d399; }
.dy { background:#fbbf24; }

/* ── Quick-question chips ───────────────────── */
.qq-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 6px 4px 2px;
    justify-content: center;
}
.qq-chip {
    font-size: 0.7em;
    padding: 4px 12px;
    border-radius: 14px;
    border: 1px solid rgba(139,92,246,0.18);
    background: rgba(139,92,246,0.05);
    color: #a5b4c8;
    cursor: pointer;
    transition: all 0.15s ease;
    text-decoration: none;
    white-space: nowrap;
}
.qq-chip:hover {
    background: rgba(139,92,246,0.14);
    border-color: rgba(139,92,246,0.35);
    color: #c4b5fd;
}
</style>
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RENDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_ask_ai_tab():
    if not LLM_READY:
        st.error("❌ `pip install langchain-nvidia-ai-endpoints`")
        return

    st.markdown(_CSS, unsafe_allow_html=True)

    # ── API key ───────────────────────────────
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        st.markdown('<div class="ai-hi"><h3>🔑 API Key</h3>'
                     '<p>Set <code>NVIDIA_API_KEY</code> di <code>.env</code></p></div>',
                     unsafe_allow_html=True)
        k = st.text_input("Key", type="password", placeholder="nvapi-...",
                          label_visibility="collapsed")
        api_key = k.strip() if k else ""
        if not api_key:
            return

    # ── State ─────────────────────────────────
    if "ask_ai_history" not in st.session_state:
        st.session_state.ask_ai_history = []
    history = st.session_state.ask_ai_history

    # ── Header ────────────────────────────────
    h1, h2 = st.columns([9, 1])
    with h1:
        n = sum(1 for m in history if m["role"] == "user")
        st.markdown(f"#### 🤖 Ask AI{'  ·  ' + str(n) + ' pesan' if n else ''}")
    with h2:
        if history and st.button("🗑️", help="Clear", use_container_width=True):
            st.session_state.ask_ai_history = []
            st.rerun()

    # ── Chat container (fixed height → input stays at bottom) ──
    chat_box = st.container(height=420)

    # ── Quick question chips (always visible, between chat & input) ──
    prefill = st.session_state.pop("_ai_prefill", None)

    # Render quick question row as clickable streamlit buttons
    qcols = st.columns(4)
    for i, q in enumerate(QUICK_QS[:4]):
        with qcols[i]:
            if st.button(q, key=f"qq_{i}", use_container_width=True):
                prefill = q

    # Second row of quick questions
    qcols2 = st.columns(4)
    for i, q in enumerate(QUICK_QS[4:8]):
        with qcols2[i]:
            if st.button(q, key=f"qq2_{i}", use_container_width=True):
                prefill = q

    # Small font override for the quick-q buttons
    st.markdown("""
    <style>
    /* Target only the quick-question button rows */
    div[data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) button {
        font-size: 0.68em !important;
        padding: 3px 6px !important;
        border-radius: 14px !important;
        border: 1px solid rgba(139,92,246,0.15) !important;
        background: rgba(139,92,246,0.04) !important;
        color: #94a3b8 !important;
        min-height: 0 !important;
        height: auto !important;
        line-height: 1.3 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(button[data-testid="stBaseButton-secondary"]) button:hover {
        background: rgba(139,92,246,0.12) !important;
        border-color: rgba(139,92,246,0.35) !important;
        color: #c4b5fd !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Chat input (pinned at bottom since page doesn't scroll) ──
    user_input = st.chat_input("Ketik pertanyaan...")
    if prefill and not user_input:
        user_input = prefill

    # ── Render inside chat container ──────────
    with chat_box:
        # Welcome (empty state)
        if not history and not user_input:
            has_csv   = os.path.exists("Housing.csv")
            has_code  = os.path.exists("main.py")
            pdfs      = glob.glob("*.pdf")
            charts    = st.session_state.get("ai_charts", [])
            has_metrics = "ai_metrics" in st.session_state

            st.markdown(f"""
            <div class="ai-hi">
                <h3>Ask AI Assistant</h3>
                <p>Tanya apapun — AI membaca semua konteks otomatis</p>
                <div class="ctx-row">
                    <span class="ctx-tag"><span class="d {'dg' if has_csv else 'dy'}"></span>Dataset</span>
                    <span class="ctx-tag"><span class="d {'dg' if has_code else 'dy'}"></span>Code</span>
                    <span class="ctx-tag"><span class="d {'dg' if pdfs else 'dy'}"></span>Paper</span>
                    <span class="ctx-tag"><span class="d {'dg' if charts else 'dy'}"></span>Charts ({len(charts)})</span>
                    <span class="ctx-tag"><span class="d {'dg' if has_metrics else 'dy'}"></span>Metrik & Evaluasi</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            return


        # Display history
        for msg in history:
            av = "👤" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=av):
                st.markdown(msg["content"])

        # New message
        if user_input:
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)

            with st.chat_message("assistant", avatar="🤖"):
                try:
                    sys_prompt = _build_system_prompt()

                    # Inject live metrics (not cached)
                    metrics_ctx = _build_metrics_context()
                    if metrics_ctx:
                        sys_prompt += "\n\n" + metrics_ctx

                    # Inject chart labels
                    charts = st.session_state.get("ai_charts", [])
                    if charts:
                        sys_prompt += "\n\n---\n## [CHARTS] Output Grafik Aplikasi\n" + "\n".join(f"- {c}" for c in charts)

                    chain = _llm(api_key) | StrOutputParser()
                    msgs = _to_lc(sys_prompt, history, user_input)
                    response_text = st.write_stream(chain.stream(msgs))

                except Exception as e:
                    err = str(e)
                    response_text = None
                    if "401" in err or "auth" in err.lower():
                        st.error("❌ API Key tidak valid")
                    elif "429" in err or "rate" in err.lower():
                        st.error("⏳ Rate limit — coba lagi")
                    else:
                        st.error(f"❌ {err}")

            if user_input:
                history.append({"role": "user", "content": user_input})
            if response_text:
                history.append({"role": "assistant", "content": response_text})
