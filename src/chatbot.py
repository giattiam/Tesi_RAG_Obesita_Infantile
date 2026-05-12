"""
chatbot.py

Interfaccia conversazionale per il sistema RAG sull'obesita infantile.
Gestisce il rilevamento del profilo utente, la riformulazione delle query,
il recupero contestuale dal vector store e la generazione delle risposte.
"""

import os
import sys
import json
import time
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from crea_vector_db import get_o_crea_vector_db, get_retriever, rerank_chunks, RETRIEVER_K

load_dotenv()

# ---------------------------------------------------------------------------
# Parametri globali
# ---------------------------------------------------------------------------

MODELLO_LLM  = "llama-3.3-70b-versatile"
TEMPERATURE  = 0.2
MAX_TOKENS   = 1024
# RETRIEVER_K importato da crea_vector_db
DEBUG_MODE   = False
CARTELLA_LOG = "./logs"

# ---------------------------------------------------------------------------
# Definizione dei profili utente
# Ogni profilo ha un system prompt calibrato sul destinatario.
# ---------------------------------------------------------------------------

PROFILI = {
    "genitore": {
        "etichetta": "Genitore",
        "emoji": "Genitore",
        "descrizione": "Genitore o tutore di un bambino",
        "temperature": 0.3,
        "prompt": (
            "Sei un assistente virtuale specializzato in nutrizione pediatrica e prevenzione "
            "dell'obesita infantile (fascia 0-10 anni). Stai parlando con un GENITORE.\n"
            "Usa un linguaggio semplice, caldo e rassicurante. Evita termini tecnici senza spiegarli. "
            "Fornisci consigli pratici applicabili nella vita quotidiana. "
            "Se il genitore sembra preoccupato, rassicura prima di rispondere nel merito.\n\n"
            "REGOLE:\n"
            "1. Rispondi SOLO usando le informazioni nel [CONTESTO] qui sotto.\n"
            "2. Se la risposta non e nel contesto, dillo e suggerisci di consultare il pediatra.\n"
            "3. Non fare diagnosi, non prescrivere terapie, non inventare dati.\n"
            "4. Usa elenchi puntati quando migliora la leggibilita.\n"
            "5. Cita il documento di riferimento alla fine della risposta.\n\n"
            "[CONTESTO]\n{context}\n[FINE CONTESTO]"
        ),
    },
    "insegnante": {
        "etichetta": "Insegnante",
        "emoji": "Insegnante",
        "descrizione": "Insegnante o educatore scolastico",
        "temperature": 0.3,
        "prompt": (
            "Sei un assistente virtuale specializzato in nutrizione pediatrica e prevenzione "
            "dell'obesita infantile. Stai parlando con un INSEGNANTE o EDUCATORE.\n"
            "Focalizzati su strategie educative applicabili in classe, attivita didattiche "
            "sull'alimentazione sana e su come coinvolgere bambini nella fascia 6-10 anni. "
            "Suggerisci approcci ludici e inclusivi. Usa un tono professionale ma accessibile.\n\n"
            "REGOLE:\n"
            "1. Rispondi SOLO usando le informazioni nel [CONTESTO] qui sotto.\n"
            "2. Se la risposta non e nel contesto, dillo chiaramente.\n"
            "3. Non fare diagnosi, non prescrivere terapie, non inventare dati.\n"
            "4. Collega le risposte al contesto scolastico quando possibile.\n"
            "5. Cita il documento di riferimento alla fine della risposta.\n\n"
            "[CONTESTO]\n{context}\n[FINE CONTESTO]"
        ),
    },
    "pediatra": {
        "etichetta": "Pediatra",
        "emoji": "Pediatra",
        "descrizione": "Medico pediatra o medico di base",
        "temperature": 0.0,
        "prompt": (
            "Sei un assistente virtuale specializzato in nutrizione pediatrica e prevenzione "
            "dell'obesita infantile. Stai parlando con un PEDIATRA o MEDICO.\n"
            "Usa terminologia clinica appropriata. Puoi fare riferimento a percentili, BMI, "
            "curve di crescita, linee guida nazionali e internazionali. "
            "Fornisci informazioni precise con riferimenti alle fonti documentali.\n\n"
            "REGOLE:\n"
            "1. Rispondi SOLO usando le informazioni nel [CONTESTO] qui sotto.\n"
            "2. Se la risposta non e nel contesto, segnalalo esplicitamente.\n"
            "3. Non inventare dati clinici o riferimenti bibliografici non presenti nel contesto.\n"
            "4. Struttura la risposta in modo clinicamente rigoroso.\n"
            "5. Cita sempre il documento di riferimento con precisione.\n\n"
            "[CONTESTO]\n{context}\n[FINE CONTESTO]"
        ),
    },
    "nutrizionista": {
        "etichetta": "Nutrizionista",
        "emoji": "Nutrizionista",
        "descrizione": "Nutrizionista o dietista",
        "temperature": 0.1,
        "prompt": (
            "Sei un assistente virtuale specializzato in nutrizione pediatrica e prevenzione "
            "dell'obesita infantile. Stai parlando con un NUTRIZIONISTA o DIETISTA.\n"
            "Puoi entrare nel dettaglio di macronutrienti, micronutrienti, apporti giornalieri "
            "raccomandati (LARN), indici glicemici, densita calorica e composizione degli alimenti. "
            "Usa un linguaggio tecnico-scientifico adeguato alla professione.\n\n"
            "REGOLE:\n"
            "1. Rispondi SOLO usando le informazioni nel [CONTESTO] qui sotto.\n"
            "2. Se la risposta non e nel contesto, segnalalo senza integrare con conoscenze esterne.\n"
            "3. Non inventare valori nutrizionali o raccomandazioni non presenti nel contesto.\n"
            "4. Fornisci risposte quantitative quando i dati sono disponibili nel contesto.\n"
            "5. Cita il documento di riferimento con precisione.\n\n"
            "[CONTESTO]\n{context}\n[FINE CONTESTO]"
        ),
    },
    "ricercatore": {
        "etichetta": "Ricercatore",
        "emoji": "Ricercatore",
        "descrizione": "Ricercatore, studente universitario o accademico",
        "temperature": 0.0,
        "prompt": (
            "Sei un assistente virtuale specializzato in nutrizione pediatrica e prevenzione "
            "dell'obesita infantile. Stai parlando con un RICERCATORE o STUDENTE UNIVERSITARIO.\n"
            "Privilegia precisione, citazione delle fonti, dati epidemiologici e riferimenti "
            "metodologici. Evidenzia i limiti dei dati quando presenti. "
            "Usa un registro accademico rigoroso.\n\n"
            "REGOLE:\n"
            "1. Rispondi SOLO usando le informazioni nel [CONTESTO] qui sotto.\n"
            "2. Segnala esplicitamente quando il contesto e insufficiente per una risposta completa.\n"
            "3. Non inventare riferimenti bibliografici o dati statistici.\n"
            "4. Distingui tra dati certi e inferenze quando necessario.\n"
            "5. Cita sempre il documento di riferimento con titolo preciso.\n\n"
            "[CONTESTO]\n{context}\n[FINE CONTESTO]"
        ),
    },
}

# ---------------------------------------------------------------------------
# Prompt di classificazione del profilo
# ---------------------------------------------------------------------------

PROMPT_CLASSIFICAZIONE = """Analizza il seguente messaggio e classifica chi lo ha scritto.
Scegli UNO dei seguenti profili in base a tono, vocabolario e contenuto:
- genitore: linguaggio familiare, riferimenti a "mio figlio/figlia", preoccupazioni quotidiane
- insegnante: menziona scuola, classe, alunni, attivita didattiche
- pediatra: terminologia medica, riferimenti a pazienti, diagnosi, percentili
- nutrizionista: termini come macronutrienti, LARN, dieta, piano alimentare
- ricercatore: linguaggio accademico, riferimenti a studi, dati, ricerca, tesi

Messaggio: "{messaggio}"

Rispondi con UNA SOLA PAROLA tra: genitore, insegnante, pediatra, nutrizionista, ricercatore"""

# ---------------------------------------------------------------------------
# Prompt di riformulazione della query per il retrieval semantico
# ---------------------------------------------------------------------------

PROMPT_QUERY_REWRITING = """Sei un esperto di information retrieval applicato alla nutrizione pediatrica.
Riscrivi la domanda seguente come query ottimizzata per la ricerca semantica in un database
di documenti scientifici italiani su alimentazione e obesita infantile (0-10 anni).
La query deve essere specifica, usare termini tecnici pertinenti e catturare l'intento reale.
Tieni conto che l'utente e un: {profilo}

Domanda originale: "{domanda}"

Rispondi SOLO con la query riscritta, senza spiegazioni o testo aggiuntivo."""


# ---------------------------------------------------------------------------
# Logging delle sessioni su file JSON
# ---------------------------------------------------------------------------

class SessionLogger:
    """
    Registra ogni interazione in un file JSON dedicato alla sessione.
    I log vengono salvati in CARTELLA_LOG con timestamp nel nome file.
    Struttura di ogni log: profilo rilevato, query originale, query riscritta,
    chunks recuperati (fonte, pagina, anteprima testo), risposta, latenza.
    """

    def __init__(self):
        os.makedirs(CARTELLA_LOG, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(CARTELLA_LOG, f"sessione_{timestamp}.json")
        self.sessione = {
            "timestamp_avvio": datetime.now().isoformat(),
            "modello_llm": MODELLO_LLM,
            "modello_retriever": f"gemini-embedding-001 k={RETRIEVER_K}",
            "profilo_rilevato": None,
            "interazioni": [],
        }

    def log_interazione(self, profilo, query_originale, query_riscritta,
                        chunks, risposta, latenza_ms):
        self.sessione["profilo_rilevato"] = profilo
        self.sessione["interazioni"].append({
            "timestamp": datetime.now().isoformat(),
            "profilo": profilo,
            "query_originale": query_originale,
            "query_riscritta": query_riscritta,
            "chunks_recuperati": [
                {
                    "fonte": c.metadata.get("titolo_documento", "?"),
                    "pagina": c.metadata.get("page", "?"),
                    "testo_preview": c.page_content[:150],
                }
                for c in chunks
            ],
            "risposta": risposta,
            "latenza_ms": latenza_ms,
        })
        self._salva()

    def _salva(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.sessione, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Classe principale del sistema RAG
# ---------------------------------------------------------------------------

class AssistenteRAG:
    """
    Pipeline RAG con rilevamento profilo, query rewriting e memoria conversazionale.

    Flusso per ogni domanda:
    1. Rilevamento profilo (solo al primo messaggio)
    2. Riformulazione della query tramite LLM
    3. Retrieval dal vector store con la query ottimizzata
    4. Generazione della risposta con system prompt specifico per profilo
    5. Aggiornamento della memoria conversazionale
    6. Logging su file JSON
    """

    def __init__(self):
        print("Inizializzazione sistema RAG...")

        self.llm = None   # inizializzato per profilo in rispondi()
        self.parser = StrOutputParser()

        db = get_o_crea_vector_db()
        self.retriever = get_retriever(db, k=RETRIEVER_K)

        self.profilo_corrente: str | None = None
        self.cronologia: list = []
        self.logger = SessionLogger()

        print("Sistema pronto.\n")

    def _rileva_profilo(self, messaggio: str) -> str:
        """
        Invia il primo messaggio al LLM con un prompt di classificazione
        e restituisce il profilo rilevato. Fallback su 'genitore'.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("human", PROMPT_CLASSIFICAZIONE.format(messaggio=messaggio))
        ])
        llm_temp  = ChatGroq(model=MODELLO_LLM, temperature=0.0, max_tokens=50)
        chain     = prompt | llm_temp | self.parser
        risultato = chain.invoke({}).strip().lower()
        for profilo in PROFILI:
            if profilo in risultato:
                return profilo
        return "genitore"

    def _riscrivi_query(self, domanda: str) -> str:
        """
        Riformula la domanda dell'utente in una query ottimizzata per la
        similarity search, tenendo conto del profilo corrente.
        """
        profilo_label = PROFILI[self.profilo_corrente]["etichetta"]
        prompt = ChatPromptTemplate.from_messages([
            ("human", PROMPT_QUERY_REWRITING.format(
                profilo=profilo_label,
                domanda=domanda,
            ))
        ])
        llm_temp       = ChatGroq(model=MODELLO_LLM, temperature=0.0, max_tokens=200)
        chain          = prompt | llm_temp | self.parser
        query_riscritta = chain.invoke({}).strip()
        return query_riscritta if query_riscritta else domanda

    def _formatta_contesto(self, docs: list) -> str:
        """Costruisce il blocco di contesto da passare al prompt di generazione."""
        sezioni = []
        for i, doc in enumerate(docs, 1):
            titolo = doc.metadata.get("titolo_documento", "Documento sconosciuto")
            pagina = doc.metadata.get("page", "")
            intestazione = (
                    f"--- Fonte {i}: {titolo}"
                    + (f", pag. {pagina}" if pagina != "" else "")
                    + " ---"
            )
            sezioni.append(f"{intestazione}\n{doc.page_content}")
        return "\n\n".join(sezioni)

    def _stampa_fonti(self, docs: list, query_riscritta: str) -> None:
        titoli = sorted(set(
            doc.metadata.get("titolo_documento", "Sconosciuto") for doc in docs
        ))
        print(f"\nFonti: {', '.join(titoli)}")
        if DEBUG_MODE:
            print(f"[DEBUG] Query riscritta: {query_riscritta!r}")
            for i, doc in enumerate(docs, 1):
                print(f"  [{i}] {doc.metadata.get('titolo_documento', '?')} "
                      f"- {doc.page_content[:120]}...")

    def rispondi(self, domanda: str) -> str:
        """
        Esegue la pipeline completa e restituisce la risposta testuale.
        Registra l'interazione nel log di sessione.
        """
        t_start = time.time()

        # Rilevamento profilo al primo messaggio
        if self.profilo_corrente is None:
            self.profilo_corrente = self._rileva_profilo(domanda)
            p = PROFILI[self.profilo_corrente]
            print(f"\nProfilo rilevato: {p['etichetta']}")
            print("(Digita 'profilo' per cambiarlo manualmente)\n")

        # Riformulazione query
        query_riscritta = self._riscrivi_query(domanda)

        # Retrieval
        docs     = self.retriever.invoke(query_riscritta)
        contesto = self._formatta_contesto(docs)

        # Generazione con temperatura specifica per profilo
        temperatura    = PROFILI[self.profilo_corrente]["temperature"]
        system_prompt  = PROFILI[self.profilo_corrente]["prompt"]
        llm_profilo    = ChatGroq(
            model=MODELLO_LLM,
            temperature=temperatura,
            max_tokens=MAX_TOKENS,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="cronologia"),
            ("human", "{domanda}"),
        ])
        chain    = prompt | llm_profilo | self.parser
        risposta = chain.invoke({
            "context":    contesto,
            "cronologia": self.cronologia,
            "domanda":    domanda,
        })

        # Aggiornamento memoria (finestra scorrevole: ultimi 10 messaggi = 5 turni)
        self.cronologia.append(HumanMessage(content=domanda))
        self.cronologia.append(AIMessage(content=risposta))
        if len(self.cronologia) > 10:
            self.cronologia = self.cronologia[-10:]

        # Log
        latenza_ms = int((time.time() - t_start) * 1000)
        self.logger.log_interazione(
            profilo=self.profilo_corrente,
            query_originale=domanda,
            query_riscritta=query_riscritta,
            chunks=docs,
            risposta=risposta,
            latenza_ms=latenza_ms,
        )

        self._stampa_fonti(docs, query_riscritta)
        return risposta

    def reset_memoria(self) -> None:
        self.cronologia = []
        print("Memoria conversazionale azzerata.")

    def cambia_profilo(self) -> None:
        """Menu interattivo per la selezione manuale del profilo."""
        print("\nProfili disponibili:")
        voci = list(PROFILI.items())
        for i, (chiave, dati) in enumerate(voci, 1):
            print(f"  {i}. {dati['etichetta']} - {dati['descrizione']}")
        scelta = input("Numero: ").strip()
        try:
            idx = int(scelta) - 1
            if 0 <= idx < len(voci):
                self.profilo_corrente = voci[idx][0]
                self.cronologia = []
                p = PROFILI[self.profilo_corrente]
                print(f"Profilo impostato: {p['etichetta']} (memoria azzerata)\n")
            else:
                print("Numero fuori range.")
        except ValueError:
            print("Input non valido.")


# ---------------------------------------------------------------------------
# Loop conversazionale a riga di comando
# ---------------------------------------------------------------------------

COMANDI_USCITA  = {"esci", "quit", "exit", "stop", "q"}
COMANDI_RESET   = {"reset", "nuova", "ricomincia"}
COMANDI_DEBUG   = {"debug"}
COMANDI_PROFILO = {"profilo", "profile", "cambia profilo"}


def chat_loop(assistente: AssistenteRAG) -> None:
    print("=" * 62)
    print("  Assistente Nutrizionale Pediatrico - RAG")
    print("-" * 62)
    print("  Comandi: reset | debug | profilo | esci")
    print("-" * 62)
    print("  Il profilo viene rilevato automaticamente dal primo messaggio.")
    print("=" * 62)

    global DEBUG_MODE

    while True:
        try:
            user_input = input("\nUtente: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nArrivederci.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in COMANDI_USCITA:
            print("Arrivederci.")
            break
        if cmd in COMANDI_RESET:
            assistente.reset_memoria()
            continue
        if cmd in COMANDI_DEBUG:
            DEBUG_MODE = not DEBUG_MODE
            print(f"Modalita debug: {'attiva' if DEBUG_MODE else 'disattiva'}.")
            continue
        if cmd in COMANDI_PROFILO:
            assistente.cambia_profilo()
            continue

        print("\nRicerca in corso...", end="", flush=True)

        try:
            risposta = assistente.rispondi(user_input)
            print("\r" + " " * 30 + "\r", end="")
            print(f"\nRisposta:\n{risposta}")
        except Exception as e:
            errore = str(e)
            if "429" in errore or "RESOURCE_EXHAUSTED" in errore:
                match  = re.search(r"retryDelay.*?(\d+)s", errore)
                attesa = int(match.group(1)) + 2 if match else 60
                print(f"\nQuota API raggiunta. Nuovo tentativo tra {attesa} secondi...")
                time.sleep(attesa)
                try:
                    risposta = assistente.rispondi(user_input)
                    print("\r" + " " * 30 + "\r", end="")
                    print(f"\nRisposta:\n{risposta}")
                except Exception as e2:
                    print(f"\nQuota giornaliera esaurita: {e2}")
                    print("Attivare la fatturazione su https://aistudio.google.com oppure riprovare domani.")
            else:
                print(f"\nErrore: {e}")
                print("Riprovare o digitare 'reset'.")


def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        print("Errore: GOOGLE_API_KEY non trovata nel file .env")
        sys.exit(1)
    try:
        assistente = AssistenteRAG()
        chat_loop(assistente)
    except FileNotFoundError as e:
        print(f"\nDatabase non trovato: {e}")
        print("Eseguire prima: python src/crea_vector_db.py")
        sys.exit(1)
    except Exception as e:
        print(f"\nErrore critico: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()