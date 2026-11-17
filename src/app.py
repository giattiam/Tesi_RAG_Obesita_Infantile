"""
app.py
Interfaccia web Streamlit per il sistema RAG sull'obesita infantile.
Layout ChatGPT-style con sidebar cronologia, pannello fonti laterale,
rilevamento automatico del profilo utente dal primo messaggio.
Avvio:
    streamlit run src/app.py
"""
import os
import sys
import uuid
import html
import json
from datetime import datetime
os.environ["TOKENIZERS_PARALLELISM"]       = "false"
os.environ["TRANSFORMERS_NO_TQDM"]         = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]         = "TRUE"
os.environ["OMP_NUM_THREADS"]              = "1"
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crea_vector_db import get_o_crea_retriever
from chatbot import AssistenteRAG, PROFILI, MODELLO_LLM
from markdown_it import MarkdownIt
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
st.set_page_config(
    page_title="Crescere Bene",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
/* === Sfondo principale === */
.stApp { background: #FAF8F4; }
.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}
/* === Sidebar === */
[data-testid="stSidebar"] {
    background: #F2EEE6;
    border-right: 1px solid #E7E1D5;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
/* Logo sidebar */
.brand-row {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.5rem 0.2rem 1rem 0.2rem;
    border-bottom: 1px solid #E7E1D5;
    margin-bottom: 1rem;
}
.brand-logo {
    width: 38px; height: 38px;
    border-radius: 12px;
    background: #4A8C83;
    color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Fraunces', serif;
    font-weight: 700; font-size: 0.92rem; letter-spacing: 0.02em;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(74,140,131,.4);
}
.brand-title {
    font-family: 'Fraunces', serif;
    font-size: 1rem; font-weight: 700; color: #2e2723; line-height: 1.1;
}
.brand-sub {
    font-size: 0.72rem; color: #9a8c7f; font-weight: 500;
}
/* "Nuova chat" button: protagonista */
[data-testid="stSidebar"] .stButton:first-of-type > button {
    background: #4A8C83 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(74,140,131,.4) !important;
}
[data-testid="stSidebar"] .stButton:first-of-type > button:hover {
    background: #3d7368 !important;
}
/* Bottoni "secondari" della cronologia: stile lista (spec HTML Crescere Bene) */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #6b615a !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    text-align: left !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 9px 12px !important;
    width: 100% !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(46,39,35,.06) !important;
    color: #2e2723 !important;
}
/* Sezione label sidebar */
.sb-label {
    font-size: 11px; font-weight: 700;
    color: #a99c8f; text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 22px 8px 8px 8px;
}
/* Search box */
[data-testid="stSidebar"] .stTextInput > div > div {
    background: #ffffff !important;
    border: 1px solid #E7E1D5 !important;
    border-radius: 999px !important;
}
[data-testid="stSidebar"] .stTextInput input {
    font-size: 0.82rem !important;
    padding: 0.45rem 0.8rem !important;
    color: #2e2723 !important;
}
/* User card in fondo alla sidebar */
.user-card {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.6rem 0.5rem;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #E7E1D5;
    margin-top: 1rem;
}
.user-avatar {
    width: 30px; height: 30px;
    background: #EAF1EF;
    color: #4A8C83;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.78rem;
    flex-shrink: 0;
}
.user-name { font-size: 0.85rem; font-weight: 700; color: #2e2723; line-height: 1.1; }
.user-role { font-size: 0.72rem; color: #9a8c7f; }
/* === Header conversazione === */
.conv-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #ECE5D6;
    margin-bottom: 1.5rem;
}
.conv-title {
    font-family: 'Fraunces', serif;
    font-size: 1rem; font-weight: 600; color: #2e2723;
}
/* === Welcome card === */
.welcome-wrap {
    text-align: center;
    padding: 5rem 1rem 1rem 1rem;
}
.welcome-title {
    font-family: 'Fraunces', serif;
    font-size: 2.1rem; font-weight: 400; color: #2a2a28;
    letter-spacing: 0.2px;
    margin-bottom: 0.8rem;
}
.welcome-sub {
    font-size: 0.95rem; color: #6b615a; line-height: 1.55;
    max-width: 640px; margin: 0 auto;
}
.welcome-disclaimer {
    display: inline-flex; align-items: center; gap: 0.4rem;
    margin-top: 2rem;
    padding: 0.45rem 1rem;
    background: #fcefd9;
    border: 1px solid #f0d49b;
    border-radius: 999px;
    font-size: 0.78rem; color: #8a6d2a;
}
/* === Bolle messaggi === */
.msg-row {
    display: flex;
    margin: 1rem auto;
    max-width: 760px;
}
.msg-user {
    justify-content: flex-end;
}
.msg-user-bubble {
    background: #efe9dd;
    color: #39342b;
    padding: 0.7rem 1rem;
    border: 1px solid #e7e0d0;
    border-radius: 18px 18px 6px 18px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.55;
}
.msg-assistant {
    justify-content: flex-start;
    gap: 0.7rem;
    align-items: flex-start;
}
.msg-avatar-ai {
    width: 32px; height: 32px;
    border-radius: 10px;
    background: #EAF1EF;
    color: #4A8C83;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Fraunces', serif;
    font-weight: 700; font-size: 0.72rem; letter-spacing: 0.02em;
    flex-shrink: 0;
}
.msg-assistant-bubble {
    background: #ffffff;
    color: #2e2723;
    padding: 0.85rem 1.1rem;
    border-radius: 6px 18px 18px 18px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.65;
    border: 1px solid #ECE5D6;
}
/* === Footer disclaimer === */
.bottom-disclaimer {
    text-align: center;
    font-size: 0.74rem;
    color: #a99c8f;
    padding: 0.5rem 0 0.2rem 0;
}
/* === Pannello fonti === */
.fonti-box {
    background: #ffffff;
    border: 1px solid #E7E1D5;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.7rem;
}
.fonti-title {
    font-size: 0.78rem; font-weight: 700; color: #2e2723;
    margin-bottom: 0.3rem;
}
.fonti-snippet {
    font-size: 0.8rem; color: #6b615a; line-height: 1.5;
}
.fonti-meta {
    font-size: 0.72rem; color: #a99c8f;
    margin-top: 0.35rem;
}
/* Nasconde solo gli elementi superflui. NON tocchiamo header/toolbar:
   il pulsante per riaprire la sidebar (stExpandSidebarButton in
   Streamlit 1.57) vive li' e sparirebbe. L'header di default e' gia'
   trasparente, quindi l'aspetto resta pulito. */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] button {
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 1000 !important;
}
/* Barra inferiore: rimuove lo sfondo bianco di default (blocco bianco) */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
    background: #FAF8F4 !important;
}
[data-testid="stBottom"] > div { background: transparent !important; }
/* Chat input */
[data-testid="stChatInput"] {
    border-radius: 16px !important;
    border: 1px solid #E7E1D5 !important;
    background: #ffffff !important;
    box-shadow: 0 8px 22px -12px rgba(46,39,35,.35) !important;
    max-width: 760px !important;
    margin: 0 auto !important;
}
[data-testid="stChatInput"] > div,
[data-testid="stChatInputTextArea"] {
    background: #ffffff !important;
}
[data-testid="stChatInputTextArea"] { color: #2e2723 !important; }
[data-testid="stChatInput"] button { color: #4A8C83 !important; }
/* Profilo selectbox sidebar */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #E7E1D5 !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
}
/* === Markdown dentro la bolla assistente === */
.msg-assistant-bubble p { margin: 0 0 0.6rem 0; }
.msg-assistant-bubble p:last-child { margin-bottom: 0; }
.msg-assistant-bubble ul, .msg-assistant-bubble ol {
    margin: 0.3rem 0 0.6rem 1.1rem; padding: 0;
}
.msg-assistant-bubble li { margin: 0.15rem 0; }
.msg-assistant-bubble h1, .msg-assistant-bubble h2,
.msg-assistant-bubble h3, .msg-assistant-bubble h4 {
    font-size: 0.98rem; font-weight: 700; margin: 0.5rem 0 0.3rem;
}
.msg-assistant-bubble code {
    background: #F2EEE6; padding: 0.1rem 0.3rem;
    border-radius: 4px; font-size: 0.85em;
}
.msg-assistant-bubble pre {
    background: #F2EEE6; padding: 0.6rem 0.8rem;
    border-radius: 10px; overflow-x: auto; font-size: 0.82em;
}
.msg-assistant-bubble strong { font-weight: 700; }
.msg-assistant-bubble a { color: #4A8C83; }
/* === Chip guardrail sopra la card === */
.guardrail-chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-size: 0.7rem; font-weight: 700;
    padding: 0.22rem 0.65rem; border-radius: 999px;
    white-space: nowrap; margin-bottom: 0.5rem;
}
/* === Nota profilo header (testo semplice, niente blocco) === */
.profilo-nota {
    margin-left: 0.6rem;
    font-size: 0.78rem; font-weight: 600; font-style: italic;
    color: #a99c8f;
}
/* === Riquadro messaggio assistente: SOLO le card delle risposte === */
[class*="st-key-msgcard_"] {
    background: #ffffff;
    border: 1px solid #ECE5D6 !important;
    border-radius: 18px;
    max-width: 760px;
    margin: 1rem auto;
    box-shadow: 0 8px 22px -16px rgba(46,39,35,.3);
}
.msg-card { display: flex; gap: 0.7rem; align-items: flex-start; }
.msg-card .msg-assistant-bubble {
    background: none; border: none; max-width: none;
    padding: 0; border-radius: 0;
}
/* Bottoni Copia/Rigenera dentro il riquadro */
.st-key-copia_btn button, .st-key-rigen_btn button {
    background: transparent !important;
    color: #9a8c7f !important;
    border: 1px solid #E7E1D5 !important;
    border-radius: 10px !important;
    font-size: 0.76rem !important;
    padding: 0.15rem 0.5rem !important;
    min-height: 0 !important;
}
.st-key-copia_btn button:hover, .st-key-rigen_btn button:hover {
    background: #EAF1EF !important; color: #4A8C83 !important;
    border-color: #bfe0cd !important;
}
/* === Riga chat in sidebar: menu ... + nome come unico riquadro === */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    gap: 0 !important;
    align-items: center;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]
    [data-testid="stPopover"] button {
    border: none !important;
    background: transparent !important;
    color: #9a8c7f !important;
    padding: 0.4rem 0.2rem !important;
    min-height: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]
    [data-testid="stPopover"] button:hover {
    color: #4A8C83 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource

def init_retriever():
    """Wrap su get_o_crea_retriever: combina cache Streamlit e singleton di processo."""
    return get_o_crea_retriever()

_MD = MarkdownIt("commonmark", {"html": False})
PERCORSO_CHATS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "logs", "conversazioni.json"
)

def _markdown_html(testo: str) -> str:
    """
    Converte il markdown della risposta LLM in HTML sicuro.
    MarkdownIt('commonmark') ha html=False: eventuali tag HTML nel testo
    vengono escapati, non eseguiti (protezione da injection).
    """
    return _MD.render(testo or "")

def _data_attivita(c: dict) -> datetime:
    """Data di riferimento per ordinamento/raggruppamento: ultima attivita."""
    return c.get("aggiornata_il") or c["creata_il"]

def _serializza_chats(chats: list) -> list:
    """Rende le chat serializzabili in JSON (datetime -> ISO string)."""
    fuori = []
    for c in chats:
        d = dict(c)
        d["creata_il"]    = c["creata_il"].isoformat()
        d["aggiornata_il"] = _data_attivita(c).isoformat()
        fuori.append(d)
    return fuori

def _carica_chats_da_disco() -> list:
    """Carica le conversazioni persistite. Lista vuota se assenti o corrotte."""
    if not os.path.exists(PERCORSO_CHATS):
        return []
    try:
        with open(PERCORSO_CHATS, encoding="utf-8") as f:
            dati = json.load(f)
        for c in dati:
            c["creata_il"] = datetime.fromisoformat(c["creata_il"])
            agg = c.get("aggiornata_il")
            c["aggiornata_il"] = (
                datetime.fromisoformat(agg) if agg else c["creata_il"]
            )
            c.setdefault("profilo", None)
            c.setdefault("fonti_ultima", [])
        return dati
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return []

def _salva_chats() -> None:
    """Persiste su disco le conversazioni della sessione corrente."""
    chats = st.session_state.get("chats", [])
    if not chats:
        return
    os.makedirs(os.path.dirname(PERCORSO_CHATS), exist_ok=True)
    try:
        with open(PERCORSO_CHATS, "w", encoding="utf-8") as f:
            json.dump(_serializza_chats(chats), f, ensure_ascii=False, indent=2)
    except OSError:
        pass

def init_session():
    if "chats" not in st.session_state:
        st.session_state.chats = _carica_chats_da_disco()
    if "chat_corrente" not in st.session_state:
        st.session_state.chat_corrente = None
    if "assistente" not in st.session_state:
        st.session_state.assistente = AssistenteRAG(retriever=init_retriever())
    if "mostra_fonti" not in st.session_state:
        st.session_state.mostra_fonti = False
    if "copia_attiva" not in st.session_state:
        st.session_state.copia_attiva = False
    if "vista" not in st.session_state:
        st.session_state.vista = "chat"

def nuova_chat():
    """Crea una nuova chat vuota e la rende corrente."""
    chat_id = str(uuid.uuid4())
    adesso = datetime.now()
    st.session_state.chats.insert(0, {
        "id": chat_id,
        "titolo": "Nuova conversazione",
        "messaggi": [],
        "fonti_ultima": [],
        "profilo": None,
        "creata_il": adesso,
        "aggiornata_il": adesso,
    })
    st.session_state.chat_corrente = chat_id
    st.session_state.assistente.cronologia       = []
    st.session_state.assistente.profilo_corrente = None

def chat_attuale() -> dict | None:
    if st.session_state.chat_corrente is None:
        return None
    for c in st.session_state.chats:
        if c["id"] == st.session_state.chat_corrente:
            return c
    return None

def elimina_chat(chat_id: str) -> None:
    st.session_state.chats = [
        c for c in st.session_state.chats if c["id"] != chat_id
    ]
    if st.session_state.chat_corrente == chat_id:
        st.session_state.chat_corrente = None
        st.session_state.assistente.cronologia       = []
        st.session_state.assistente.profilo_corrente = None
    _salva_chats()

def rinomina_chat(chat_id: str, nuovo_titolo: str) -> None:
    nuovo_titolo = (nuovo_titolo or "").strip()
    if not nuovo_titolo:
        return
    for c in st.session_state.chats:
        if c["id"] == chat_id:
            c["titolo"] = nuovo_titolo[:80]
            break
    _salva_chats()

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="brand-row">
            <div class="brand-logo">CB</div>
            <div>
                <div class="brand-title">Crescere Bene</div>
                <div class="brand-sub">Assistente RAG &middot; 0&ndash;10 anni</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("+ Nuova chat", key="btn_nuova_chat"):
            nuova_chat()
            st.session_state.vista = "chat"
            st.rerun()
        is_dash = st.session_state.get("vista") == "dashboard"
        if st.button(
            ("← Torna alla chat" if is_dash else "Dashboard sicurezza"),
            key="btn_dashboard",
            use_container_width=True,
        ):
            st.session_state.vista = "chat" if is_dash else "dashboard"
            st.rerun()
        recenti = sorted(
            st.session_state.chats, key=_data_attivita, reverse=True
        )
        if recenti:
            st.markdown('<div class="sb-label">Recenti</div>', unsafe_allow_html=True)
            for c in recenti:
                titolo    = c["titolo"]
                etichetta = titolo if len(titolo) <= 34 else titolo[:31] + "..."
                if st.button(
                    etichetta,
                    key=f"chat_{c['id']}",
                    use_container_width=True,
                ):
                    st.session_state.chat_corrente = c["id"]
                    _ricarica_cronologia_assistente(c)
                    st.rerun()

def _ricarica_cronologia_assistente(chat: dict) -> None:
    """Quando l'utente cambia chat, ripopola la memoria conversazionale del LLM."""
    from langchain_core.messages import HumanMessage, AIMessage
    assist = st.session_state.assistente
    assist.cronologia = []
    assist.profilo_corrente = chat.get("profilo")
    for m in chat["messaggi"]:
        if m["ruolo"] == "utente":
            assist.cronologia.append(HumanMessage(content=m["testo"]))
        else:
            assist.cronologia.append(AIMessage(content=m["testo"]))
    if len(assist.cronologia) > 10:
        assist.cronologia = assist.cronologia[-10:]

def render_header(titolo: str, profilo: str | None = None):
    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        nota = ""
        if profilo and profilo in PROFILI:
            nota = (
                f'<span class="profilo-nota">profilo rilevato: '
                f'{html.escape(PROFILI[profilo]["etichetta"])}</span>'
            )
        st.markdown(
            f'<div class="conv-header" style="border:none;padding:0;margin:0;">'
            f'<span class="conv-title">{html.escape(titolo)}</span>{nota}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        ch = chat_attuale()
        with st.popover("⋯", use_container_width=True):
            if ch is not None:
                nuovo = st.text_input(
                    "Rinomina conversazione",
                    value=ch["titolo"],
                    key=f"rename_hdr_{ch['id']}",
                )
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Salva", key=f"save_hdr_{ch['id']}"):
                        rinomina_chat(ch["id"], nuovo)
                        st.rerun()
                with b2:
                    if st.button("Elimina", key=f"del_hdr_{ch['id']}"):
                        elimina_chat(ch["id"])
                        st.rerun()
    with col3:
        label = "Fonti ▾" if not st.session_state.mostra_fonti else "Fonti ▴"
        if st.button(label, key="btn_fonti"):
            st.session_state.mostra_fonti = not st.session_state.mostra_fonti
            st.rerun()
    st.markdown('<hr style="border:none;border-top:1px solid #ECE5D6;margin:0.4rem 0 1.2rem 0;" />',
                unsafe_allow_html=True)

def _carica_guardrail_log() -> list:
    """Legge tutti i file logs/guardrail_*.jsonl e ritorna gli eventi."""
    import glob
    eventi = []
    for path in sorted(glob.glob("logs/guardrail_*.jsonl")):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            eventi.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError:
            continue
    return eventi

def render_dashboard() -> None:
    """Dashboard di sicurezza: aggrega i trigger registrati su file."""
    st.markdown("## Dashboard di sicurezza")
    st.caption(
        "Aggregazione dei trigger registrati dal livello guardrail. "
        "I dati provengono dai file `logs/guardrail_*.jsonl`."
    )
    eventi = _carica_guardrail_log()
    if not eventi:
        st.info(
            "Nessun trigger registrato finora. I controlli vengono loggati "
            "automaticamente quando scattano."
        )
        return
    n_tot    = len(eventi)
    n_input  = sum(1 for e in eventi if e.get("tipo") == "input")
    n_output = sum(1 for e in eventi if e.get("tipo") == "output")
    c1, c2, c3 = st.columns(3)
    c1.metric("Trigger totali", n_tot)
    c2.metric("Su input",       n_input)
    c3.metric("Su output",      n_output)
    from collections import Counter
    contatori = Counter()
    for e in eventi:
        for t in e.get("decisione", {}).get("triggered", []):
            chiave = t.split(":")[0]
            contatori[chiave] += 1
    if contatori:
        st.markdown("### Distribuzione per categoria")
        import pandas as pd
        df = pd.DataFrame(
            sorted(contatori.items(), key=lambda kv: -kv[1]),
            columns=["categoria", "occorrenze"],
        )
        st.bar_chart(df, x="categoria", y="occorrenze")
    st.markdown("### Ultimi 20 eventi")
    eventi_recenti = sorted(eventi, key=lambda e: e.get("ts", ""), reverse=True)[:20]
    righe = []
    for e in eventi_recenti:
        triggered = e.get("decisione", {}).get("triggered", [])
        righe.append({
            "timestamp":  e.get("ts", "")[:19].replace("T", " "),
            "tipo":       e.get("tipo", ""),
            "trigger":    ", ".join(triggered),
            "estratto":   str(e.get("payload", ""))[:120],
        })
    if righe:
        import pandas as pd
        st.dataframe(pd.DataFrame(righe), use_container_width=True, hide_index=True)

_ETICHETTE_GUARDRAIL = {
    "prompt_injection":             ("Richiesta non consentita",      "#dc2626"),
    "richiesta_prescrizione":       ("Ridiretto al medico",           "#dc2626"),
    "richiesta_diagnosi_personale": ("Ridiretto al pediatra",         "#dc2626"),
    "off_topic":                    ("Fuori dominio",                 "#dc2626"),
    "output_diagnosi":              ("Disclaimer applicato",          "#d97706"),
    "output_prescrizione":          ("Disclaimer applicato",          "#d97706"),
    "missing_citation":             ("Citazione mancante",            "#d97706"),
}

def _classifica_guardrail(triggers: list) -> tuple[str, str] | None:
    """Restituisce (etichetta, colore) in base alla severita' massima dei trigger."""
    if not triggers:
        return None
    for t in triggers:
        if t in _ETICHETTE_GUARDRAIL:
            return _ETICHETTE_GUARDRAIL[t]
    for t in triggers:
        if t.startswith("grounding_low"):
            return ("Grounding basso", "#d97706")
        if t.startswith("pii:"):
            return ("Dato personale rimosso", "#2563eb")
    return ("Avviso di sicurezza", "#6b7280")

def render_messaggio(msg: dict, idx: int, is_last: bool = False):
    if msg["ruolo"] == "utente":
        testo = html.escape(msg["testo"]).replace("\n", "<br>")
        st.markdown(
            f'<div class="msg-row msg-user">'
            f'<div class="msg-user-bubble">{testo}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return
    corpo = _markdown_html(msg["testo"])
    triggers = msg.get("guardrail") or []
    classe = _classifica_guardrail(triggers)
    with st.container(border=True, key=f"msgcard_{idx}"):
        if classe is not None:
            etichetta, colore = classe
            st.markdown(
                f'<div class="guardrail-chip" style="background:{colore}1A;'
                f'color:{colore};border:1px solid {colore};">'
                f'&#x1F6E1; {html.escape(etichetta)}'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="msg-card">'
            f'<div class="msg-avatar-ai">CB</div>'
            f'<div class="msg-assistant-bubble">{corpo}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if triggers:
            with st.expander("Controlli di sicurezza"):
                st.markdown(
                    "Questo messaggio ha attivato i seguenti controlli del "
                    "livello guardrail:"
                )
                for t in triggers:
                    st.markdown(f"- `{t}`")
        if is_last:
            a1, a2, _ = st.columns([1, 1, 6])
            with a1:
                if st.button("Copia", key="copia_btn", use_container_width=True):
                    st.session_state.copia_attiva = not st.session_state.copia_attiva
                    st.rerun()
            with a2:
                if st.button("Rigenera", key="rigen_btn", use_container_width=True):
                    rigenera_ultima_risposta()
                    st.rerun()
            if st.session_state.copia_attiva:
                st.code(msg["testo"], language=None)

def render_pannello_fonti(chat: dict):
    fonti = chat.get("fonti_ultima") or []
    if not fonti:
        st.info("Nessuna fonte disponibile per questa conversazione.")
        return
    st.markdown('<div class="sb-label" style="padding-left:0;">Fonti consultate</div>',
                unsafe_allow_html=True)
    for i, f in enumerate(fonti, 1):
        titolo  = f.get("titolo", "Documento sconosciuto")
        pagina  = f.get("pagina", "")
        snippet = f.get("snippet", "")
        meta    = titolo + (f", pag. {pagina}" if pagina != "" else "")
        st.markdown(f"""
        <div class="fonti-box">
            <div class="fonti-title">[{i}] {meta}</div>
            <div class="fonti-snippet">{snippet}</div>
        </div>
        """, unsafe_allow_html=True)

def esegui_pipeline(domanda: str) -> dict:
    """
    Esegue la pipeline RAG completa tramite AssistenteRAG.genera() (metodo
    unico condiviso con la CLI) e adatta il risultato al formato atteso
    dalla UI: {risposta, fonti, query_riscritta, profilo, guardrail}.
    """
    assistente = st.session_state.assistente
    with st.status("Sto consultando le fonti…", expanded=False) as status:
        ris = assistente.genera(
            domanda,
            on_step=lambda label: status.update(label=label),
        )
        status.update(label="Risposta pronta", state="complete")
    fonti = [
        {
            "titolo":  d.metadata.get("titolo_documento", "Sconosciuto"),
            "pagina":  d.metadata.get("page", ""),
            "snippet": d.page_content[:280] + ("..." if len(d.page_content) > 280 else ""),
        }
        for d in ris["docs"]
    ]
    return {
        "risposta":        ris["risposta"],
        "fonti":           fonti,
        "query_riscritta": ris["query_riscritta"],
        "profilo":         ris["profilo"],
        "guardrail":       ris["guardrail"],
    }

def rigenera_ultima_risposta() -> None:
    """Rigenera la risposta all'ultima domanda della chat corrente."""
    from langchain_core.messages import HumanMessage, AIMessage
    chat = chat_attuale()
    if chat is None or not chat["messaggi"]:
        return
    if chat["messaggi"][-1]["ruolo"] == "assistente":
        chat["messaggi"].pop()
    if not chat["messaggi"] or chat["messaggi"][-1]["ruolo"] != "utente":
        return
    ultima_domanda = chat["messaggi"][-1]["testo"]
    assist = st.session_state.assistente
    assist.profilo_corrente = chat.get("profilo")
    assist.cronologia = []
    for m in chat["messaggi"][:-1]:
        ruolo = HumanMessage if m["ruolo"] == "utente" else AIMessage
        assist.cronologia.append(ruolo(content=m["testo"]))
    if len(assist.cronologia) > 10:
        assist.cronologia = assist.cronologia[-10:]
    try:
        risultato = esegui_pipeline(ultima_domanda)
        chat["messaggi"].append({
            "ruolo":           "assistente",
            "testo":           risultato["risposta"],
            "query_riscritta": risultato["query_riscritta"],
            "profilo":         risultato["profilo"],
            "guardrail":       risultato.get("guardrail", []),
        })
        chat["fonti_ultima"] = risultato["fonti"]
        chat["profilo"]      = risultato["profilo"]
    except Exception as e:
        chat["messaggi"].append({"ruolo": "assistente", "testo": f"Errore: {e}"})
    chat["aggiornata_il"] = datetime.now()
    _salva_chats()

def main():
    init_session()
    init_retriever()
    render_sidebar()
    if st.session_state.get("vista") == "dashboard":
        render_dashboard()
        _salva_chats()
        return
    chat = chat_attuale()
    if chat is None:
        st.markdown("""
        <div class="welcome-wrap">
            <div class="welcome-title">Ciao, come posso aiutarti?</div>
            <div class="welcome-sub">
                Assistente per la <strong>prevenzione dell'obesit&agrave; infantile</strong> (0&ndash;10 anni).<br>
                Risposte basate su linee guida e documenti scientifici, adattate al tuo profilo.
            </div>
            <div class="welcome-disclaimer">
                Strumento di supporto &middot; non sostituisce il parere medico
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.session_state.mostra_fonti:
            col_chat, col_fonti = st.columns([2, 1])
        else:
            col_chat, col_fonti = st.container(), None
        with col_chat:
            render_header(chat["titolo"], chat.get("profilo"))
            ultimo_idx = len(chat["messaggi"]) - 1
            for i, m in enumerate(chat["messaggi"]):
                render_messaggio(
                    m, i,
                    is_last=(i == ultimo_idx and m["ruolo"] == "assistente"),
                )
        if col_fonti is not None:
            with col_fonti:
                render_pannello_fonti(chat)
    if domanda := st.chat_input("Scrivi una domanda..."):
        if chat_attuale() is None:
            nuova_chat()
            chat = chat_attuale()
        chat["messaggi"].append({"ruolo": "utente", "testo": domanda})
        if chat["titolo"] == "Nuova conversazione":
            chat["titolo"] = domanda[:60] + ("..." if len(domanda) > 60 else "")
        try:
            risultato = esegui_pipeline(domanda)
            chat["messaggi"].append({
                "ruolo":           "assistente",
                "testo":           risultato["risposta"],
                "query_riscritta": risultato["query_riscritta"],
                "profilo":         risultato["profilo"],
                "guardrail":       risultato.get("guardrail", []),
            })
            chat["fonti_ultima"] = risultato["fonti"]
            chat["profilo"]      = risultato["profilo"]
        except Exception as e:
            chat["messaggi"].append({
                "ruolo": "assistente",
                "testo": f"Errore: {e}",
            })
        chat["aggiornata_il"] = datetime.now()
        _salva_chats()
        st.rerun()
    st.markdown(
        '<div class="bottom-disclaimer">Le risposte possono contenere imprecisioni '
        '&middot; verifica sempre con la fonte</div>',
        unsafe_allow_html=True,
    )
    _salva_chats()
if __name__ == "__main__":
    main()
