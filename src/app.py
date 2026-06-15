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
    page_title="RAG-Obesita",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
/* === Sfondo principale === */
.stApp { background: #ffffff; }
.main .block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}
/* === Sidebar === */
[data-testid="stSidebar"] {
    background: #f7f8fa;
    border-right: 1px solid #e7e9ee;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
/* Logo sidebar */
.brand-row {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.5rem 0.2rem 1rem 0.2rem;
    border-bottom: 1px solid #e7e9ee;
    margin-bottom: 1rem;
}
.brand-logo {
    width: 36px; height: 36px;
    border-radius: 10px;
    background: #2151ff;
    color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 1rem;
    flex-shrink: 0;
}
.brand-title {
    font-size: 0.92rem; font-weight: 700; color: #0f172a; line-height: 1.1;
}
.brand-sub {
    font-size: 0.72rem; color: #6b7280; font-weight: 500;
}
/* "Nuova chat" button: protagonista */
[data-testid="stSidebar"] .stButton:first-of-type > button {
    background: #2151ff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.88rem !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton:first-of-type > button:hover {
    background: #1a3fd0 !important;
}
/* Bottoni "secondari" della cronologia: stile lista */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #1f2937 !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    text-align: left !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 0.6rem !important;
    width: 100% !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #ececf0 !important;
}
/* Sezione label sidebar */
.sb-label {
    font-size: 0.7rem; font-weight: 600;
    color: #6b7280; text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.6rem 0.4rem 0.3rem 0.4rem;
}
/* Search box */
[data-testid="stSidebar"] .stTextInput > div > div {
    background: #ffffff !important;
    border: 1px solid #e0e2e8 !important;
    border-radius: 999px !important;
}
[data-testid="stSidebar"] .stTextInput input {
    font-size: 0.82rem !important;
    padding: 0.45rem 0.8rem !important;
}
/* User card in fondo alla sidebar */
.user-card {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.6rem 0.5rem;
    border-radius: 10px;
    background: #ffffff;
    border: 1px solid #e0e2e8;
    margin-top: 1rem;
}
.user-avatar {
    width: 30px; height: 30px;
    background: #e5e7eb;
    color: #374151;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 600; font-size: 0.78rem;
    flex-shrink: 0;
}
.user-name { font-size: 0.85rem; font-weight: 600; color: #0f172a; line-height: 1.1; }
.user-role { font-size: 0.72rem; color: #6b7280; }
/* === Header conversazione === */
.conv-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #eef0f3;
    margin-bottom: 1.5rem;
}
.conv-title { font-size: 0.95rem; font-weight: 600; color: #0f172a; }
/* === Welcome card === */
.welcome-wrap {
    text-align: center;
    padding: 5rem 1rem 1rem 1rem;
}
.welcome-title {
    font-size: 2rem; font-weight: 700; color: #0f172a;
    margin-bottom: 0.8rem;
}
.welcome-sub {
    font-size: 0.95rem; color: #6b7280; line-height: 1.55;
    max-width: 640px; margin: 0 auto;
}
.welcome-disclaimer {
    display: inline-flex; align-items: center; gap: 0.4rem;
    margin-top: 2rem;
    padding: 0.45rem 1rem;
    background: #f7f8fa;
    border: 1px dashed #cbd5e1;
    border-radius: 999px;
    font-size: 0.78rem; color: #64748b;
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
    background: #2151ff;
    color: #ffffff;
    padding: 0.7rem 1rem;
    border-radius: 16px 16px 4px 16px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 2px 8px rgba(33,81,255,0.18);
}
.msg-assistant {
    justify-content: flex-start;
    gap: 0.7rem;
    align-items: flex-start;
}
.msg-avatar-ai {
    width: 32px; height: 32px;
    border-radius: 8px;
    background: #2151ff;
    color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.72rem;
    flex-shrink: 0;
}
.msg-assistant-bubble {
    background: #f7f8fa;
    color: #0f172a;
    padding: 0.85rem 1.1rem;
    border-radius: 4px 16px 16px 16px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.65;
    border: 1px solid #eef0f3;
}
/* === Footer disclaimer === */
.bottom-disclaimer {
    text-align: center;
    font-size: 0.74rem;
    color: #94a3b8;
    padding: 0.5rem 0 0.2rem 0;
}
/* === Pannello fonti === */
.fonti-box {
    background: #f7f8fa;
    border: 1px solid #e0e2e8;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.7rem;
}
.fonti-title {
    font-size: 0.78rem; font-weight: 700; color: #0f172a;
    margin-bottom: 0.3rem;
}
.fonti-snippet {
    font-size: 0.8rem; color: #475569; line-height: 1.5;
}
.fonti-meta {
    font-size: 0.72rem; color: #94a3b8;
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
/* Chat input */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1px solid #e0e2e8 !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04) !important;
    max-width: 760px !important;
    margin: 0 auto !important;
}
/* Profilo selectbox sidebar */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #e0e2e8 !important;
    border-radius: 8px !important;
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
    background: #eef0f3; padding: 0.1rem 0.3rem;
    border-radius: 4px; font-size: 0.85em;
}
.msg-assistant-bubble pre {
    background: #eef0f3; padding: 0.6rem 0.8rem;
    border-radius: 8px; overflow-x: auto; font-size: 0.82em;
}
.msg-assistant-bubble strong { font-weight: 700; }
.msg-assistant-bubble a { color: #2151ff; }
/* === Chip guardrail sopra la card === */
.guardrail-chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-size: 0.7rem; font-weight: 600;
    padding: 0.22rem 0.65rem; border-radius: 999px;
    white-space: nowrap; margin-bottom: 0.5rem;
}
/* === Chip profilo header === */
.profilo-chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: #eef2ff; color: #2151ff;
    font-size: 0.72rem; font-weight: 600;
    padding: 0.22rem 0.65rem; border-radius: 999px;
    white-space: nowrap;
}
/* === Riquadro messaggio assistente (container con bordo) === */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #f7f8fa;
    border: 1px solid #eef0f3 !important;
    border-radius: 16px;
    max-width: 760px;
    margin: 1rem auto;
}
.msg-card { display: flex; gap: 0.7rem; align-items: flex-start; }
.msg-card .msg-assistant-bubble {
    background: none; border: none; max-width: none;
    padding: 0; border-radius: 0;
}
/* Bottoni Copia/Rigenera dentro il riquadro */
.st-key-copia_btn button, .st-key-rigen_btn button {
    background: transparent !important;
    color: #6b7280 !important;
    border: 1px solid #e0e2e8 !important;
    border-radius: 8px !important;
    font-size: 0.76rem !important;
    padding: 0.15rem 0.5rem !important;
    min-height: 0 !important;
}
.st-key-copia_btn button:hover, .st-key-rigen_btn button:hover {
    background: #eef2ff !important; color: #2151ff !important;
    border-color: #c7d2fe !important;
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
    color: #6b7280 !important;
    padding: 0.4rem 0.2rem !important;
    min-height: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]
    [data-testid="stPopover"] button:hover {
    color: #2151ff !important;
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
    if "search_chats" not in st.session_state:
        st.session_state.search_chats = ""
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

def _applica_profilo_selectbox():
    """
    Callback della selectbox profilo. Scatta SOLO quando l'utente cambia
    manualmente il profilo (non a ogni rerun): aggiorna la fonte di verita
    (assistente.profilo_corrente) e la propaga alla chat corrente.
    """
    chiavi = [None] + list(PROFILI.keys())
    nuovo  = chiavi[st.session_state.sel_profilo]
    st.session_state.assistente.profilo_corrente = nuovo
    ch = chat_attuale()
    if ch is not None:
        ch["profilo"] = nuovo
        _salva_chats()

def raggruppa_chats(chats: list) -> dict:
    """
    Raggruppa le chat per fascia temporale in base all'ULTIMA ATTIVITA
    (non alla data di creazione), ordinate dalla piu' recente.
    """
    oggi   = datetime.now().date()
    gruppi = {
        "Oggi": [], "Ieri": [],
        "Ultimi 7 giorni": [], "Ultimi 30 giorni": [], "Piu' vecchie": [],
    }
    for c in sorted(chats, key=_data_attivita, reverse=True):
        delta = (oggi - _data_attivita(c).date()).days
        if delta <= 0:
            gruppi["Oggi"].append(c)
        elif delta == 1:
            gruppi["Ieri"].append(c)
        elif delta <= 7:
            gruppi["Ultimi 7 giorni"].append(c)
        elif delta <= 30:
            gruppi["Ultimi 30 giorni"].append(c)
        else:
            gruppi["Piu' vecchie"].append(c)
    return gruppi

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="brand-row">
            <div class="brand-logo">R</div>
            <div>
                <div class="brand-title">RAG-Obesita</div>
                <div class="brand-sub">v0.3 &middot; tesi</div>
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
        st.text_input(
            "cerca",
            placeholder="cerca chat...",
            label_visibility="collapsed",
            key="search_chats",
        )
        query   = (st.session_state.search_chats or "").strip().lower()
        gruppi  = raggruppa_chats(st.session_state.chats)
        for nome_gruppo, lista in gruppi.items():
            visibili = [c for c in lista if not query or query in c["titolo"].lower()]
            if not visibili:
                continue
            st.markdown(f'<div class="sb-label">{nome_gruppo}</div>', unsafe_allow_html=True)
            for c in visibili:
                titolo    = c["titolo"]
                etichetta = titolo if len(titolo) <= 30 else titolo[:27] + "..."
                col_menu, col_nome = st.columns([0.16, 0.84], gap="small")
                with col_menu:
                    with st.popover(":material/menu:", use_container_width=True):
                        nuovo = st.text_input(
                            "Rinomina conversazione",
                            value=c["titolo"],
                            key=f"rename_{c['id']}",
                        )
                        b1, b2 = st.columns(2)
                        with b1:
                            if st.button("Salva", key=f"save_{c['id']}"):
                                rinomina_chat(c["id"], nuovo)
                                st.rerun()
                        with b2:
                            if st.button("Elimina", key=f"del_{c['id']}"):
                                elimina_chat(c["id"])
                                st.rerun()
                with col_nome:
                    if st.button(
                        f"💬 {etichetta}",
                        key=f"chat_{c['id']}",
                        use_container_width=True,
                    ):
                        st.session_state.chat_corrente = c["id"]
                        _ricarica_cronologia_assistente(c)
                        st.rerun()
        st.markdown("<div style='flex:1; min-height:2rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sb-label">Profilo</div>', unsafe_allow_html=True)
        profilo_corr = st.session_state.assistente.profilo_corrente
        opzioni      = ["— rilevamento automatico —"] + [PROFILI[k]["etichetta"] for k in PROFILI]
        chiavi       = [None] + list(PROFILI.keys())
        desired_idx  = chiavi.index(profilo_corr) if profilo_corr in chiavi else 0
        st.session_state.sel_profilo = desired_idx
        st.selectbox(
            "profilo",
            options=range(len(opzioni)),
            format_func=lambda i: opzioni[i],
            label_visibility="collapsed",
            key="sel_profilo",
            on_change=_applica_profilo_selectbox,
        )
        nome_profilo = (
            PROFILI[profilo_corr]["etichetta"]
            if profilo_corr else "—"
        )
        iniziale = nome_profilo[0].upper() if nome_profilo != "—" else "?"
        st.markdown(f"""
        <div class="user-card">
            <div class="user-avatar">{iniziale}</div>
            <div>
                <div class="user-name">Utente</div>
                <div class="user-role">{nome_profilo}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
    col1, col2 = st.columns([6, 1])
    with col1:
        chip = ""
        if profilo and profilo in PROFILI:
            chip = (
                f'<span class="profilo-chip">&#9679; '
                f'{html.escape(PROFILI[profilo]["etichetta"])}</span>'
            )
        st.markdown(
            f'<div class="conv-header" style="border:none;padding:0;margin:0;">'
            f'<span class="conv-title">{html.escape(titolo)}</span>{chip}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        label = "Fonti ▾" if not st.session_state.mostra_fonti else "Fonti ▴"
        if st.button(label, key="btn_fonti"):
            st.session_state.mostra_fonti = not st.session_state.mostra_fonti
            st.rerun()
    st.markdown('<hr style="border:none;border-top:1px solid #eef0f3;margin:0.4rem 0 1.2rem 0;" />',
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
    with st.container(border=True):
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
            f'<div class="msg-avatar-ai">AI</div>'
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
    st.markdown('<div class="sb-label" style="padding-left:0;">Chunks recuperati</div>',
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
    Esegue rilevamento profilo (al primo messaggio) + pipeline RAG completa,
    racchiusa fra i guardrail di input e di output.
    Restituisce {risposta, fonti, query_riscritta, profilo, guardrail}.
    """
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage
    from chatbot import get_llm, POLICY_RAG
    from crea_vector_db import rerank_chunks, RETRIEVER_K
    from guardrails import filtra_input, filtra_output, verifica_grounding

    # --- INPUT GUARDRAIL ---
    dec_in = filtra_input(domanda)
    if not dec_in.consenti:
        return {
            "risposta":        dec_in.messaggio_utente,
            "fonti":           [],
            "query_riscritta": "",
            "profilo":         None,
            "guardrail":       list(dec_in.triggered),
        }
    domanda_pulita = dec_in.domanda_sanificata

    assistente = st.session_state.assistente
    with st.status("Elaborazione in corso...", expanded=False) as status:
        status.update(label="Identificazione del profilo...")
        system_prompt, temperatura, prof_eff = assistente._prepara_generazione(domanda_pulita)
        status.update(label="Analisi della domanda...")
        query_riscritta = assistente._riscrivi_query(domanda_pulita)
        status.update(label="Ricerca nei documenti...")
        candidati = assistente.retriever.invoke(query_riscritta)
        status.update(label="Selezione dei passaggi piu' pertinenti...")
        llm_rerank = get_llm(temperature=0.0, max_tokens=16)
        docs = rerank_chunks(candidati, query_riscritta, llm_rerank, k=RETRIEVER_K)
        status.update(label="Elaborazione della risposta...")
        contesto      = assistente._formatta_contesto(docs)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("system", POLICY_RAG),
            MessagesPlaceholder(variable_name="cronologia"),
            ("human", "{domanda}"),
        ])
        llm   = get_llm(temperature=temperatura)
        chain = prompt | llm | assistente.parser
        risposta = chain.invoke({
            "context":    contesto,
            "cronologia": assistente.cronologia,
            "domanda":    domanda_pulita,
        })
        # --- RUNTIME GROUNDING + OUTPUT GUARDRAIL ---
        status.update(label="Verifica grounding...")
        score_grounding = verifica_grounding(risposta, docs)
        dec_out = filtra_output(risposta, docs, score_grounding=score_grounding)
        risposta = dec_out.risposta
        assistente.cronologia.append(HumanMessage(content=domanda_pulita))
        assistente.cronologia.append(AIMessage(content=risposta))
        if len(assistente.cronologia) > 10:
            assistente.cronologia = assistente.cronologia[-10:]
        status.update(label="Risposta pronta", state="complete")
    fonti = [
        {
            "titolo":  d.metadata.get("titolo_documento", "Sconosciuto"),
            "pagina":  d.metadata.get("page", ""),
            "snippet": d.page_content[:280] + ("..." if len(d.page_content) > 280 else ""),
        }
        for d in docs
    ]
    return {
        "risposta":        risposta,
        "fonti":           fonti,
        "query_riscritta": query_riscritta,
        "profilo":         prof_eff,
        "guardrail":       list(dec_in.triggered) + list(dec_out.triggered),
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
            <div class="welcome-title">Come posso aiutarti oggi?</div>
            <div class="welcome-sub">
                Assistente conversazionale per il counseling sull'obesita infantile.<br>
                Risposte basate su linee guida e letteratura scientifica.
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
