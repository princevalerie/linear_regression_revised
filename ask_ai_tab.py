"""
ask_ai_tab.py
─────────────────────────────────────────────────────────────────────
Tab 8 · Ask AI — ChatNVIDIA (LangChain) RAG Chatbot
Menggunakan langchain-nvidia-ai-endpoints → moonshotai/kimi-k2.6

Konteks RAG (otomatis, ringan — text injection):
  1. Housing.csv     — dataset lengkap
  2. main.py         — source code aplikasi
  3. ask_ai_tab.py   — source code tab AI
  4. README.md       — dokumentasi
  5. *.pdf           — paper PDF (teks diekstrak via PyPDF2)

User hanya mengetik — tidak bisa upload file.
─────────────────────────────────────────────────────────────────────
"""

import os
import glob
import textwrap
import io
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ── LangChain NVIDIA ────────────────────────────────────────────
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

# ── PDF text extraction ────────────────────────────────────────
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ── dotenv ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None


def _extract_pdf_text(path, max_pages=30):
    """Ekstrak teks dari PDF menggunakan PyPDF2."""
    if not PDF_AVAILABLE:
        return None
    try:
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages[:max_pages]):
            text = page.extract_text()
            if text:
                pages.append(f"[Page {i+1}]\n{text}")
        return "\n\n".join(pages) if pages else None
    except Exception:
        return None


def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=90)
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────────
#  CHART CAPTURE — dipanggil dari main.py
# ──────────────────────────────────────────────

def capture_chart(fig, label=""):
    """Simpan label chart ke session state untuk deskripsi konteks AI."""
    if "ask_ai_chart_labels" not in st.session_state:
        st.session_state.ask_ai_chart_labels = []
    if len(st.session_state.ask_ai_chart_labels) >= 20:
        st.session_state.ask_ai_chart_labels.pop(0)
    st.session_state.ask_ai_chart_labels.append(label or f"Chart {len(st.session_state.ask_ai_chart_labels)+1}")


# ──────────────────────────────────────────────
#  BUILD RAG SYSTEM PROMPT (lightweight)
# ──────────────────────────────────────────────

@st.cache_data(ttl=300)
def _build_system_prompt():
    """Build system prompt dengan semua konteks RAG — di-cache 5 menit."""
    sections = []

    # ── Preamble ──
    sections.append(textwrap.dedent("""
    Kamu adalah AI assistant expert untuk proyek "House Price Prediction — Linear Regression from Scratch".
    Kamu memiliki akses ke SEMUA konteks berikut:

    KEMAMPUANMU:
    - Analisis dataset Housing.csv (545 baris, 13 kolom harga rumah)
    - Jelaskan implementasi matematis OLS, MSE, R², Adj R² dari source code
    - Bandingkan 4 model: Univariate, Multivariate +/- Outlier, XGBoost
    - Debug code Python/Streamlit
    - Jawab pertanyaan tentang paper Liming Yan 2024
    - Suggest perbaikan model atau kode
    - Analisis chart/grafik berdasarkan deskripsi

    ATURAN:
    - Jawab dalam Bahasa Indonesia kecuali diminta Bahasa Inggris
    - Gunakan format markdown
    - Sertakan rumus LaTeX jika relevan (wrap dengan $...$)
    - Jawab ringkas dan informatif
    - Jika tidak tahu, bilang tidak tahu — jangan mengarang
    """).strip())

    # ── Housing.csv ──
    csv_text = _read_text("Housing.csv")
    if csv_text:
        # Kirim header + 30 baris pertama + statistik
        lines = csv_text.strip().split("\n")
        sample = "\n".join(lines[:31])
        sections.append(f"---\n## [DATA] Housing.csv ({len(lines)-1} baris)\n```csv\n{sample}\n```\n... (total {len(lines)-1} baris)")

    # ── main.py ──
    code_text = _read_text("main.py")
    if code_text:
        # Kirim penuh jika < 15KB, truncate jika lebih
        if len(code_text) > 15000:
            snippet = code_text[:15000] + "\n...[truncated]"
        else:
            snippet = code_text
        sections.append(f"---\n## [CODE] main.py\n```python\n{snippet}\n```")

    # ── ask_ai_tab.py ──
    ai_code = _read_text("ask_ai_tab.py")
    if ai_code:
        if len(ai_code) > 8000:
            snippet = ai_code[:8000] + "\n...[truncated]"
        else:
            snippet = ai_code
        sections.append(f"---\n## [CODE] ask_ai_tab.py\n```python\n{snippet}\n```")

    # ── README.md ──
    readme = _read_text("README.md")
    if readme:
        sections.append(f"---\n## [DOC] README.md\n{readme}")

    # ── PDF papers ──
    pdf_paths = glob.glob("*.pdf") + glob.glob("**/*.pdf", recursive=False)
    seen = set()
    for pdf_path in pdf_paths[:2]:
        abs_path = os.path.abspath(pdf_path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        pdf_text = _extract_pdf_text(pdf_path)
        if pdf_text:
            # Limit PDF text
            if len(pdf_text) > 12000:
                pdf_text = pdf_text[:12000] + "\n...[truncated]"
            sections.append(f"---\n## [PAPER] {os.path.basename(pdf_path)}\n{pdf_text}")

    return "\n\n".join(sections)


# ──────────────────────────────────────────────
#  NVIDIA ChatNVIDIA CALL
# ──────────────────────────────────────────────

def _call_nvidia(api_key, user_question, history, system_prompt):
    """Call ChatNVIDIA via LangChain."""
    client = ChatNVIDIA(
        model="moonshotai/kimi-k2.6",
        api_key=api_key,
        temperature=1,
        top_p=1,
        max_tokens=16384,
    )

    # Build messages: system + history + current question
    lc_messages = [{"role": "system", "content": system_prompt}]

    for msg in history:
        lc_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    lc_messages.append({"role": "user", "content": user_question})

    response = client.invoke(lc_messages)

    # Extract content
    result = ""
    if hasattr(response, "additional_kwargs") and response.additional_kwargs:
        reasoning = response.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            result += f"<details><summary>💭 Reasoning</summary>\n\n{reasoning}\n\n</details>\n\n"
    result += response.content
    return result


# ──────────────────────────────────────────────
#  CUSTOM CSS
# ──────────────────────────────────────────────

def _inject_chat_css():
    st.markdown("""
    <style>
    /* ── Clean chatbot styling ─────────────────── */
    .stChatMessage {
        border-radius: 16px !important;
        margin-bottom: 8px !important;
        padding: 12px 16px !important;
    }

    /* User message */
    .stChatMessage[data-testid="stChatMessage-user"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        border: 1px solid rgba(96, 165, 250, 0.15) !important;
    }

    /* Assistant message */
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%) !important;
        border: 1px solid rgba(139, 92, 246, 0.15) !important;
    }

    /* Chat input styling */
    .stChatInput {
        border-radius: 24px !important;
    }
    .stChatInput > div {
        border-radius: 24px !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        background: rgba(15, 15, 26, 0.8) !important;
    }
    .stChatInput > div:focus-within {
        border-color: rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.1) !important;
    }

    /* Suggestion buttons */
    .suggestion-btn button {
        border-radius: 20px !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        background: rgba(139, 92, 246, 0.05) !important;
        font-size: 0.82em !important;
        padding: 6px 14px !important;
        transition: all 0.2s ease !important;
    }
    .suggestion-btn button:hover {
        background: rgba(139, 92, 246, 0.15) !important;
        border-color: rgba(139, 92, 246, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.08) 0%, rgba(96, 165, 250, 0.06) 100%);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 20px;
        padding: 28px 32px;
        margin: 16px 0 24px 0;
        text-align: center;
    }
    .welcome-card h2 {
        margin: 0 0 8px 0;
        font-size: 1.5em;
        background: linear-gradient(135deg, #8B5CF6, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .welcome-card p {
        color: #94a3b8;
        margin: 4px 0;
        font-size: 0.9em;
    }

    /* Status bar */
    .status-bar {
        display: flex;
        justify-content: center;
        gap: 16px;
        margin-top: 14px;
        flex-wrap: wrap;
    }
    .status-item {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.78em;
        color: #64748b;
        background: rgba(100, 116, 139, 0.08);
        padding: 4px 10px;
        border-radius: 12px;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot.green { background: #34D399; }
    .status-dot.yellow { background: #FBBF24; }
    .status-dot.red { background: #F87171; }

    /* Clear button */
    .clear-btn {
        position: fixed;
        bottom: 80px;
        right: 24px;
        z-index: 100;
    }
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  MAIN RENDER
# ──────────────────────────────────────────────

def render_ask_ai_tab():
    if not NVIDIA_AVAILABLE:
        st.error("❌ Package `langchain-nvidia-ai-endpoints` belum terinstall.")
        st.code("pip install langchain-nvidia-ai-endpoints", language="bash")
        return

    _inject_chat_css()

    # ── API Key (from .env or input) ──────────────────────────
    api_key = os.environ.get("NVIDIA_API_KEY", "")

    if not api_key:
        st.markdown("""
        <div class="welcome-card">
            <h2>🤖 Ask AI</h2>
            <p>Masukkan NVIDIA API Key untuk memulai</p>
        </div>
        """, unsafe_allow_html=True)

        api_key_input = st.text_input(
            "🔑 NVIDIA API Key",
            type="password",
            placeholder="nvapi-...",
            help="Dapatkan di https://build.nvidia.com — atau set NVIDIA_API_KEY di file .env",
            label_visibility="collapsed"
        )
        api_key = api_key_input.strip() if api_key_input else ""

        if not api_key:
            st.markdown("""
            <div style="text-align:center; color:#64748b; font-size:0.85em; margin-top:12px;">
                💡 Set <code>NVIDIA_API_KEY=nvapi-...</code> di file <code>.env</code> agar otomatis terbaca
            </div>
            """, unsafe_allow_html=True)
            return

    # ── Init session state ────────────────────────────────────
    if "ask_ai_history" not in st.session_state:
        st.session_state.ask_ai_history = []

    # ── Check context files ──────────────────────────────────
    ctx_files = {
        "Housing.csv": os.path.exists("Housing.csv"),
        "main.py": os.path.exists("main.py"),
        "README.md": os.path.exists("README.md"),
    }
    pdfs = glob.glob("*.pdf")
    n_charts = len(st.session_state.get("ask_ai_chart_labels", []))

    # ── Welcome screen (no history) ──────────────────────────
    if not st.session_state.ask_ai_history:
        # Welcome card
        ctx_count = sum(ctx_files.values()) + len(pdfs)
        st.markdown(f"""
        <div class="welcome-card">
            <h2>🤖 Ask AI Assistant</h2>
            <p>Tanya apapun tentang dataset, kode, model, atau paper.</p>
            <p>AI membaca <strong>{ctx_count} file konteks</strong> secara otomatis.</p>
            <div class="status-bar">
                <span class="status-item">
                    <span class="status-dot {'green' if ctx_files['Housing.csv'] else 'red'}"></span>
                    Dataset
                </span>
                <span class="status-item">
                    <span class="status-dot {'green' if ctx_files['main.py'] else 'red'}"></span>
                    Source Code
                </span>
                <span class="status-item">
                    <span class="status-dot {'green' if pdfs else 'yellow'}"></span>
                    Paper PDF {'✓' if pdfs else '—'}
                </span>
                <span class="status-item">
                    <span class="status-dot {'green' if n_charts > 0 else 'yellow'}"></span>
                    Charts ({n_charts})
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Suggestion buttons ────────────────────────────────
        suggestions = [
            ("📊", "Jelaskan hasil R² keempat model"),
            ("🧮", "Bagaimana OLS diimplementasikan di kode?"),
            ("📉", "Kenapa R² turun setelah outlier dihapus?"),
            ("🌲", "Bandingkan XGBoost vs OLS"),
            ("📄", "Apa kesimpulan paper Liming Yan 2024?"),
            ("🔍", "Analisis korelasi price vs fitur lainnya"),
        ]

        cols = st.columns(2)
        for i, (icon, text) in enumerate(suggestions):
            with cols[i % 2]:
                st.markdown('<div class="suggestion-btn">', unsafe_allow_html=True)
                if st.button(f"{icon}  {text}", key=f"sug_{i}", use_container_width=True):
                    st.session_state["ask_ai_prefill"] = text
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    # ── Chat history display ──────────────────────────────────
    for msg in st.session_state.ask_ai_history:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── Clear chat button (only if history exists) ────────────
    if st.session_state.ask_ai_history:
        col_spacer, col_clear = st.columns([5, 1])
        with col_clear:
            if st.button("🗑️ Clear", key="clear_chat", use_container_width=True):
                st.session_state.ask_ai_history = []
                st.rerun()

    # ── Handle prefill from suggestion buttons ────────────────
    prefill = st.session_state.pop("ask_ai_prefill", None)

    # ── Chat input ────────────────────────────────────────────
    user_input = st.chat_input("Ketik pertanyaan tentang dataset, kode, atau paper...")

    if prefill and not user_input:
        user_input = prefill

    if user_input:
        # Add user message
        st.session_state.ask_ai_history.append({
            "role": "user",
            "content": user_input
        })
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Call AI
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("💭 Thinking..."):
                try:
                    system_prompt = _build_system_prompt()

                    # Add chart labels to system prompt if available
                    chart_labels = st.session_state.get("ask_ai_chart_labels", [])
                    if chart_labels:
                        chart_info = "\n---\n## [CHARTS] Output Visual Aplikasi\nBerikut chart yang sudah di-generate:\n"
                        for i, lbl in enumerate(chart_labels):
                            chart_info += f"- Chart {i+1}: {lbl}\n"
                        system_prompt += "\n\n" + chart_info

                    response_text = _call_nvidia(
                        api_key=api_key,
                        user_question=user_input,
                        history=st.session_state.ask_ai_history[:-1],
                        system_prompt=system_prompt,
                    )
                    st.markdown(response_text)
                    st.session_state.ask_ai_history.append({
                        "role": "assistant",
                        "content": response_text
                    })

                except Exception as e:
                    err_msg = str(e)
                    if "401" in err_msg or "auth" in err_msg.lower() or "invalid" in err_msg.lower():
                        st.error("❌ **API Key tidak valid.** Cek NVIDIA API Key Anda.")
                    elif "429" in err_msg or "rate" in err_msg.lower():
                        st.error("⏳ **Rate limit.** Tunggu sebentar lalu coba lagi.")
                    else:
                        st.error(f"❌ **Error:** {err_msg}")
                    # Remove failed user message
                    st.session_state.ask_ai_history.pop()
