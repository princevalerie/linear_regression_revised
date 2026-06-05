"""
ask_ai_tab.py — ChatNVIDIA RAG Chatbot (LangChain)
Model: moonshotai/kimi-k2.6 via NVIDIA AI Endpoints
Lightweight RAG: semua konteks di-inject ke system prompt
"""

import os, glob, textwrap
import streamlit as st

# ── LangChain ──────────────────────────────────────────────────
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser
    LLM_READY = True
except ImportError:
    LLM_READY = False

# ── PDF ────────────────────────────────────────────────────────
try:
    from PyPDF2 import PdfReader
    PDF_OK = True
except ImportError:
    PDF_OK = False

# ── dotenv ─────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RAG SYSTEM PROMPT (cached, lightweight)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=300)
def _build_system_prompt():
    parts = [textwrap.dedent("""\
    Kamu adalah AI assistant expert untuk proyek "House Price Prediction — Linear Regression from Scratch".
    Kamu memiliki akses PENUH ke dataset, source code, dokumentasi, dan paper yang diberikan di bawah.

    KEMAMPUAN:
    • Analisis dataset Housing.csv (545 baris, 13 kolom harga rumah)
    • Jelaskan implementasi OLS, MSE, R², Adj R² dari source code
    • Bandingkan 4 model: Univariate, Multivariate ± Outlier, XGBoost
    • Debug code Python/Streamlit
    • Jawab tentang paper Liming Yan 2024

    ATURAN:
    • Jawab dalam Bahasa Indonesia kecuali diminta Bahasa Inggris
    • Gunakan markdown · LaTeX dalam $...$ jika relevan
    • Ringkas, informatif, akurat
    • Jangan mengarang — jika tidak tahu, bilang tidak tahu""")]

    # Dataset
    csv = _read("Housing.csv")
    if csv:
        lines = csv.strip().split("\n")
        parts.append(f"---\n## [DATA] Housing.csv ({len(lines)-1} rows)\n```csv\n{chr(10).join(lines[:31])}\n```")

    # Source code
    for fname in ["main.py", "ask_ai_tab.py"]:
        code = _read(fname)
        if code:
            s = code[:15000] + ("\n…[truncated]" if len(code) > 15000 else "")
            parts.append(f"---\n## [CODE] {fname}\n```python\n{s}\n```")

    # README
    readme = _read("README.md")
    if readme:
        parts.append(f"---\n## [DOC] README.md\n{readme}")

    # PDFs
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM (cached)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource
def _llm(api_key):
    return ChatNVIDIA(
        model="moonshotai/kimi-k2.6",
        api_key=api_key,
        temperature=1,
        top_p=1,
        max_tokens=16384,
    )


def _to_langchain_messages(system_prompt, history, user_input):
    """Convert chat history → LangChain message objects."""
    msgs = [SystemMessage(content=system_prompt)]
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=user_input))
    return msgs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CSS = """
<style>
/* ── Chat container scroll area ─────────────── */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div > div[data-testid="stChatMessage"]) {
    background: rgba(8, 10, 18, 0.4);
    border: 1px solid rgba(55, 65, 81, 0.3);
    border-radius: 16px;
}

/* ── Message bubbles ────────────────────────── */
[data-testid="stChatMessage"] {
    padding: 10px 16px !important;
    margin: 2px 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    border-bottom: 1px solid rgba(55, 65, 81, 0.15) !important;
}
[data-testid="stChatMessage"]:last-child {
    border-bottom: none !important;
}

/* ── Chat input ─────────────────────────────── */
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    font-size: 0.92em !important;
}

/* ── Welcome card ───────────────────────────── */
.ai-welcome {
    text-align: center;
    padding: 40px 20px 24px;
}
.ai-welcome h3 {
    font-size: 1.35em;
    margin-bottom: 4px;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.ai-welcome p {
    color: #64748b;
    font-size: 0.88em;
    margin: 2px 0;
}
.ctx-pills {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-top: 14px;
    flex-wrap: wrap;
}
.ctx-pill {
    font-size: 0.73em;
    padding: 3px 10px;
    border-radius: 10px;
    background: rgba(100,116,139,0.1);
    color: #94a3b8;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.ctx-pill .dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    display: inline-block;
}
.dot-g { background: #34d399; }
.dot-y { background: #fbbf24; }

/* ── Suggestion chips ───────────────────────── */
div[data-testid="stHorizontalBlock"] .stButton > button {
    border-radius: 18px !important;
    font-size: 0.8em !important;
    padding: 5px 14px !important;
    border: 1px solid rgba(139,92,246,0.2) !important;
    background: rgba(139,92,246,0.04) !important;
    transition: all 0.15s ease !important;
    white-space: nowrap !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: rgba(139,92,246,0.12) !important;
    border-color: rgba(139,92,246,0.4) !important;
}

/* ── Header row ─────────────────────────────── */
.chat-hdr {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
}
.chat-hdr h3 {
    margin: 0;
    font-size: 1.1em;
    color: #e2e8f0;
}
</style>
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN RENDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_ask_ai_tab():
    if not LLM_READY:
        st.error("❌ Install dulu: `pip install langchain-nvidia-ai-endpoints`")
        return

    st.markdown(_CSS, unsafe_allow_html=True)

    # ── API key ───────────────────────────────────────────────
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        st.markdown("""
        <div class="ai-welcome">
            <h3>🔑 API Key Required</h3>
            <p>Set <code>NVIDIA_API_KEY</code> di file <code>.env</code></p>
        </div>
        """, unsafe_allow_html=True)
        key_in = st.text_input("NVIDIA API Key", type="password",
                               placeholder="nvapi-...", label_visibility="collapsed")
        if key_in and key_in.strip():
            api_key = key_in.strip()
        else:
            return

    # ── Session state ─────────────────────────────────────────
    if "ask_ai_history" not in st.session_state:
        st.session_state.ask_ai_history = []

    history = st.session_state.ask_ai_history

    # ── Header ────────────────────────────────────────────────
    hdr_l, hdr_r = st.columns([8, 1])
    with hdr_l:
        count = len([m for m in history if m["role"] == "user"])
        label = f"🤖 Ask AI · {count} pesan" if count else "🤖 Ask AI"
        st.markdown(f"#### {label}")
    with hdr_r:
        if history and st.button("🗑️", help="Clear chat", use_container_width=True):
            st.session_state.ask_ai_history = []
            st.rerun()

    # ── Scrollable chat container ─────────────────────────────
    chat_box = st.container(height=520)

    # ── Prefill handling ──────────────────────────────────────
    prefill = st.session_state.pop("_ai_prefill", None)

    # ── Chat input (Streamlit pins this to bottom) ────────────
    user_input = st.chat_input("Ketik pertanyaan tentang dataset, kode, atau paper...")
    if prefill and not user_input:
        user_input = prefill

    # ── Render messages inside scroll container ───────────────
    with chat_box:
        # Welcome screen (empty history)
        if not history and not user_input:
            has_csv = os.path.exists("Housing.csv")
            has_code = os.path.exists("main.py")
            pdfs = glob.glob("*.pdf")
            charts = st.session_state.get("ai_charts", [])

            st.markdown(f"""
            <div class="ai-welcome">
                <h3>Ask AI Assistant</h3>
                <p>Tanya apapun — AI membaca dataset, code, dan paper otomatis</p>
                <div class="ctx-pills">
                    <span class="ctx-pill"><span class="dot {'dot-g' if has_csv else 'dot-y'}"></span> Dataset</span>
                    <span class="ctx-pill"><span class="dot {'dot-g' if has_code else 'dot-y'}"></span> Code</span>
                    <span class="ctx-pill"><span class="dot {'dot-g' if pdfs else 'dot-y'}"></span> Paper</span>
                    <span class="ctx-pill"><span class="dot {'dot-g' if charts else 'dot-y'}"></span> Charts ({len(charts)})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Suggestion chips
            suggestions = [
                "Jelaskan hasil R² keempat model",
                "Bagaimana OLS diimplementasikan?",
                "Kenapa R² turun tanpa outlier?",
                "Bandingkan XGBoost vs OLS",
                "Kesimpulan paper Liming Yan?",
                "Korelasi price vs fitur lain",
            ]
            cols = st.columns(3)
            for i, s in enumerate(suggestions):
                with cols[i % 3]:
                    if st.button(s, key=f"q{i}", use_container_width=True):
                        st.session_state["_ai_prefill"] = s
                        st.rerun()
            return

        # Display history
        for msg in history:
            av = "👤" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=av):
                st.markdown(msg["content"])

        # Handle new input
        if user_input:
            # Show user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)

            # AI response with streaming
            with st.chat_message("assistant", avatar="🤖"):
                try:
                    sys_prompt = _build_system_prompt()
                    # Append chart info
                    charts = st.session_state.get("ai_charts", [])
                    if charts:
                        sys_prompt += "\n\n---\n## [CHARTS]\n" + "\n".join(
                            f"- {c}" for c in charts
                        )

                    client = _llm(api_key)
                    chain = client | StrOutputParser()
                    msgs = _to_langchain_messages(sys_prompt, history, user_input)

                    # Stream response
                    response_text = st.write_stream(chain.stream(msgs))

                except Exception as e:
                    err = str(e)
                    if "401" in err or "auth" in err.lower():
                        response_text = None
                        st.error("❌ API Key tidak valid")
                    elif "429" in err or "rate" in err.lower():
                        response_text = None
                        st.error("⏳ Rate limit — tunggu sebentar")
                    else:
                        response_text = None
                        st.error(f"❌ {err}")

            # Save to history
            if user_input:
                history.append({"role": "user", "content": user_input})
            if response_text:
                history.append({"role": "assistant", "content": response_text})
