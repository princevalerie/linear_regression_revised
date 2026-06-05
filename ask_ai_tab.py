"""
ask_ai_tab.py
─────────────────────────────────────────────────────────────────────
Tab 8 · Ask AI — Gemini multimodal RAG
Menggunakan google-genai SDK v2 (google.genai)

Konteks yang di-load otomatis:
  1. Housing.csv          — dataset lengkap
  2. main.py             — source code aplikasi
  3. README.md           — dokumentasi
  4. *.pdf               — paper Liming Yan 2024 (jika ada di folder)
  5. output chart        — gambar yang di-render via session state
  6. File upload user    — gambar / PDF tambahan dari user
─────────────────────────────────────────────────────────────────────
"""

import os, io, base64, glob, textwrap
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None

def _read_bytes(path):
    try:
        with open(path, "rb") as f:
            return f.read()
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
    """Simpan figure ke session state agar bisa dibaca AI."""
    if "ask_ai_charts" not in st.session_state:
        st.session_state.ask_ai_charts = []
        st.session_state.ask_ai_chart_labels = []
    if len(st.session_state.ask_ai_charts) >= 12:
        st.session_state.ask_ai_charts.pop(0)
        st.session_state.ask_ai_chart_labels.pop(0)
    img_bytes = _fig_to_png_bytes(fig)
    st.session_state.ask_ai_charts.append(img_bytes)
    st.session_state.ask_ai_chart_labels.append(label or f"Chart {len(st.session_state.ask_ai_charts)}")


# ──────────────────────────────────────────────
#  BUILD RAG CONTEXT PARTS
# ──────────────────────────────────────────────

def _build_context_parts(extra_images=None, extra_pdf_bytes=None):
    parts = []

    preamble = textwrap.dedent("""
    Kamu adalah AI assistant expert untuk proyek "House Price Prediction — Linear Regression from Scratch".
    Kamu memiliki akses ke SEMUA konteks berikut yang disertakan dalam pesan ini:

    KONTEKS YANG TERSEDIA:
    - Dataset: Housing.csv (545 baris, 13 kolom harga rumah)
    - Source Code: main.py (implementasi OLS dari scratch, 4 model, 7 tab Streamlit)
    - README.md (dokumentasi proyek, pipeline, dependensi)
    - Paper PDF: "Predicting House Prices with a Linear Regression Model" — Liming Yan, 2024
    - Output chart dari aplikasi (gambar inline jika tersedia di session)
    - File yang di-upload user (jika ada)

    KEMAMPUANMU:
    - Analisis dataset, distribusi, statistik deskriptif, korelasi
    - Jelaskan implementasi matematis OLS, MSE, R², Adj R² dari kode
    - Bandingkan 4 model (Univariate, Multivariate +/- Outlier, XGBoost)
    - Debug code Python/Streamlit
    - Analisis gambar chart yang dikirimkan
    - Jawab pertanyaan tentang paper Liming Yan 2024 (dari PDF)
    - Suggest perbaikan model atau kode

    Jawab dalam Bahasa Indonesia kecuali user minta Bahasa Inggris.
    Gunakan format markdown. Sertakan rumus LaTeX jika relevan (wrap dengan $...$).
    Jika ada gambar chart, analisis dan deskripsikan apa yang terlihat.
    """).strip()
    parts.append(types.Part(text=preamble))

    # Housing.csv
    csv_text = _read_text("Housing.csv")
    if csv_text:
        parts.append(types.Part(text=f"\n\n---\n## [DATA] Housing.csv\n```\n{csv_text[:8000]}\n```"))

    # main.py
    code_text = _read_text("main.py")
    if code_text:
        snippet = code_text[:12000] + ("\n...[truncated]" if len(code_text) > 12000 else "")
        parts.append(types.Part(text=f"\n\n---\n## [CODE] main.py\n```python\n{snippet}\n```"))

    # README.md
    readme = _read_text("README.md")
    if readme:
        parts.append(types.Part(text=f"\n\n---\n## [DOC] README.md\n{readme}"))

    # PDF paper(s)
    pdf_paths = glob.glob("*.pdf") + glob.glob("**/*.pdf", recursive=False)
    for pdf_path in pdf_paths[:2]:
        pdf_bytes = _read_bytes(pdf_path)
        if pdf_bytes and len(pdf_bytes) < 20_000_000:
            parts.append(types.Part(
                inline_data=types.Blob(mime_type="application/pdf", data=pdf_bytes)
            ))
            parts.append(types.Part(
                text=f"\n\n---\n## [PAPER] {os.path.basename(pdf_path)} (lihat PDF di atas)"
            ))

    # Extra user PDF
    if extra_pdf_bytes:
        parts.append(types.Part(
            inline_data=types.Blob(mime_type="application/pdf", data=extra_pdf_bytes)
        ))
        parts.append(types.Part(text="\n\n---\n## [UPLOAD] PDF tambahan dari user"))

    # Chart dari session state
    charts = st.session_state.get("ask_ai_charts", [])
    labels = st.session_state.get("ask_ai_chart_labels", [])
    if charts:
        parts.append(types.Part(text=f"\n\n---\n## [CHARTS] Output Visual Aplikasi ({len(charts)} gambar)\nBerikut chart yang di-generate aplikasi:"))
        for i, (img_bytes, lbl) in enumerate(zip(charts[:6], labels[:6])):
            parts.append(types.Part(text=f"\n**Chart {i+1}: {lbl}**"))
            parts.append(types.Part(
                inline_data=types.Blob(mime_type="image/png", data=img_bytes)
            ))

    # Extra user images
    if extra_images:
        parts.append(types.Part(text=f"\n\n---\n## [UPLOAD] Gambar dari user ({len(extra_images)} file)"))
        for img_bytes in extra_images[:4]:
            parts.append(types.Part(
                inline_data=types.Blob(mime_type="image/png", data=img_bytes)
            ))

    return parts


# ──────────────────────────────────────────────
#  GEMINI API CALL
# ──────────────────────────────────────────────

def _call_gemini(api_key, user_question, history, context_parts, model_name="gemini-2.0-flash"):
    client = genai.Client(api_key=api_key)
    contents = []

    if not history:
        # First message: full context + question
        user_parts = context_parts + [
            types.Part(text=f"\n\n---\n## Pertanyaan\n{user_question}")
        ]
        contents = [types.Content(role="user", parts=user_parts)]
    else:
        # Multi-turn: inject context on first message only
        first_q = history[0]["content"]
        first_parts = context_parts + [
            types.Part(text=f"\n\n---\n## Pertanyaan (1)\n{first_q}")
        ]
        contents.append(types.Content(role="user", parts=first_parts))

        for msg in history[1:]:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])]
            ))
        # Current question
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=user_question)]
        ))

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            max_output_tokens=4096,
            temperature=0.7,
        )
    )
    return response.text


# ──────────────────────────────────────────────
#  FREE MODELS LIST
# ──────────────────────────────────────────────

FREE_MODELS = [
    ("gemini-2.0-flash",       "⚡ Gemini 2.0 Flash (Recommended)"),
    ("gemini-2.0-flash-lite",  "🪶 Gemini 2.0 Flash Lite (Fastest)"),
    ("gemini-1.5-flash",       "🔥 Gemini 1.5 Flash"),
    ("gemini-1.5-flash-8b",    "🔥 Gemini 1.5 Flash 8B"),
]


# ──────────────────────────────────────────────
#  MAIN RENDER
# ──────────────────────────────────────────────

def render_ask_ai_tab():
    if not GENAI_AVAILABLE:
        st.error("❌ Package `google-genai` belum terinstall. Jalankan: `pip install google-genai`")
        return

    st.markdown("""
## 🤖 Ask AI — Gemini Multimodal RAG

Tanya apapun tentang proyek ini: **dataset**, **kode**, **hasil model**, **paper**, atau **chart output**.  
AI membaca **semua konteks otomatis** — termasuk gambar output dari setiap tab aplikasi.
    """)

    # ── API Key & Model ────────────────────────────────────────
    col_key, col_model = st.columns([3, 2])
    with col_key:
        api_key_env = os.environ.get("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            "🔑 Gemini API Key",
            value=api_key_env,
            type="password",
            placeholder="AIza... (atau set GEMINI_API_KEY di .env)",
            help="Gratis di https://aistudio.google.com/apikey"
        )
        api_key = api_key_input.strip() if api_key_input.strip() else api_key_env

    with col_model:
        model_labels = [m[1] for m in FREE_MODELS]
        model_choice_label = st.selectbox("🧠 Model Gemini", model_labels)
        model_name = FREE_MODELS[model_labels.index(model_choice_label)][0]

    if not api_key:
        st.info("💡 Masukkan API Key untuk mulai. Set `GEMINI_API_KEY=AIza...` di file `.env` atau export di terminal.")
        st.code("# Cara set env:\nexport GEMINI_API_KEY=AIza...\n# atau taruh di .env file:\nGEMINI_API_KEY=AIza...", language="bash")
        return

    st.divider()

    # ── RAG Context Info ───────────────────────────────────────
    with st.expander("📂 Konteks RAG yang Di-load Otomatis", expanded=False):
        col_ctx1, col_ctx2 = st.columns(2)
        with col_ctx1:
            st.markdown("**📝 File Teks/Code:**")
            for fname in ["Housing.csv", "main.py", "README.md"]:
                ex = os.path.exists(fname)
                size = f" ({os.path.getsize(fname)/1024:.1f} KB)" if ex else ""
                st.markdown(f"- {'✅' if ex else '❌'} `{fname}`{size}")

            st.markdown("**📄 Paper PDF:**")
            pdfs = glob.glob("*.pdf")
            if pdfs:
                for p in pdfs:
                    st.markdown(f"- ✅ `{os.path.basename(p)}` ({os.path.getsize(p)/1024/1024:.2f} MB)")
            else:
                st.warning("⚠️ PDF tidak ditemukan — taruh `.pdf` di folder yang sama dengan `main.py`")

        with col_ctx2:
            st.markdown("**🖼️ Output Chart (dari session):**")
            n_charts = len(st.session_state.get("ask_ai_charts", []))
            if n_charts > 0:
                for lbl in st.session_state.get("ask_ai_chart_labels", []):
                    st.markdown(f"- 🖼️ {lbl}")
                if st.button("🗑️ Hapus semua chart", use_container_width=True):
                    st.session_state.ask_ai_charts = []
                    st.session_state.ask_ai_chart_labels = []
                    st.rerun()
            else:
                st.info("ℹ️ Belum ada chart. Buka tab lain (Univariate, Multi, XGBoost, Comparison) agar chart ter-capture.")

    st.divider()

    # ── File Upload ────────────────────────────────────────────
    up_col1, up_col2 = st.columns(2)
    with up_col1:
        uploaded_images = st.file_uploader(
            "📸 Upload Gambar Tambahan (opsional)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="Screenshot, chart lain, atau gambar yang ingin ditanyakan",
            key="ask_ai_img_upload"
        )
    with up_col2:
        uploaded_pdf = st.file_uploader(
            "📄 Upload PDF Tambahan (opsional)",
            type=["pdf"],
            help="Paper atau dokumen referensi tambahan",
            key="ask_ai_pdf_upload"
        )

    if uploaded_images:
        prev_cols = st.columns(min(len(uploaded_images), 4))
        for i, uf in enumerate(uploaded_images[:4]):
            with prev_cols[i]:
                st.image(uf, caption=uf.name, use_container_width=True)

    st.divider()

    # ── Chat History ───────────────────────────────────────────
    if "ask_ai_history" not in st.session_state:
        st.session_state.ask_ai_history = []

    for msg in st.session_state.ask_ai_history:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── Suggested Questions (only when empty) ─────────────────
    if not st.session_state.ask_ai_history:
        st.markdown("##### 💡 Contoh Pertanyaan Cepat")
        suggestions = [
            "Jelaskan hasil R² keempat model dan mana yang terbaik?",
            "Apa itu OLS dan bagaimana implementasinya di kode ini?",
            "Kenapa R² turun setelah outlier dihapus? Jelaskan matematis.",
            "Analisis feature importance XGBoost vs koefisien OLS",
            "Sesuai paper Liming Yan, apa limitasi linear regression?",
            "Analisis chart yang ada — apakah ada tanda overfitting?",
            "Berapa korelasi price vs area dari dataset ini?",
            "Jelaskan pipeline preprocessing data di kode ini",
        ]
        rows = [suggestions[i:i+2] for i in range(0, len(suggestions), 2)]
        for row in rows:
            cols = st.columns(2)
            for j, sug in enumerate(row):
                with cols[j]:
                    if st.button(f"💬 {sug}", key=f"sug_{''.join(c for c in sug if c.isalnum())[:20]}", use_container_width=True):
                        st.session_state["ask_ai_prefill"] = sug
                        st.rerun()

    # ── Handle prefill ─────────────────────────────────────────
    prefill = st.session_state.pop("ask_ai_prefill", None)

    # ── Chat Input ─────────────────────────────────────────────
    user_input = st.chat_input("Tanya tentang dataset, kode, model, chart, atau paper...")

    if prefill and not user_input:
        user_input = prefill

    if user_input:
        st.session_state.ask_ai_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        # Prepare uploads
        extra_images_bytes = []
        for uf in (uploaded_images or []):
            uf.seek(0)
            extra_images_bytes.append(uf.read())

        extra_pdf_bytes = None
        if uploaded_pdf:
            uploaded_pdf.seek(0)
            extra_pdf_bytes = uploaded_pdf.read()

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner(f"🔍 Membangun konteks RAG & mengirim ke {model_choice_label}..."):
                try:
                    ctx_parts = _build_context_parts(
                        extra_images=extra_images_bytes or None,
                        extra_pdf_bytes=extra_pdf_bytes
                    )
                    response_text = _call_gemini(
                        api_key=api_key,
                        user_question=user_input,
                        history=st.session_state.ask_ai_history[:-1],
                        context_parts=ctx_parts,
                        model_name=model_name
                    )
                    st.markdown(response_text)
                    st.session_state.ask_ai_history.append({
                        "role": "assistant", "content": response_text
                    })

                except Exception as e:
                    err_msg = str(e)
                    if "API_KEY_INVALID" in err_msg or "401" in err_msg:
                        st.error("❌ **API Key tidak valid.** Cek di [aistudio.google.com](https://aistudio.google.com/apikey)")
                    elif "quota" in err_msg.lower() or "429" in err_msg:
                        st.error("⏳ **Rate limit.** Tunggu sebentar atau ganti model.")
                    elif "SAFETY" in err_msg.upper():
                        st.warning("⚠️ **Blocked oleh safety filter.** Rephrase pertanyaan.")
                    elif "PDF" in err_msg.upper() or "mime" in err_msg.lower():
                        st.error(f"❌ **Error PDF/media:** {err_msg}")
                    else:
                        st.error(f"❌ **Error:** {err_msg}")
                        st.code(err_msg)
                    st.session_state.ask_ai_history.pop()

    # ── Footer Controls ────────────────────────────────────────
    if st.session_state.ask_ai_history:
        st.divider()
        f1, f2, f3 = st.columns([1, 1, 2])
        with f1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.ask_ai_history = []
                st.rerun()
        with f2:
            n = sum(1 for m in st.session_state.ask_ai_history if m["role"] == "user")
            st.markdown(f"<div style='padding-top:8px;color:gray;font-size:0.85em'>{n} pertanyaan</div>",
                        unsafe_allow_html=True)
        with f3:
            st.markdown(
                "<div style='padding-top:8px;color:gray;font-size:0.8em'>"
                "💡 Konteks lengkap (dataset+code+PDF+chart) di-inject tiap pertanyaan baru."
                "</div>",
                unsafe_allow_html=True
            )
