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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from crea_vector_db import get_o_crea_vector_db, get_retriever, rerank_chunks, RETRIEVER_K, MODELLO_EMBED
from guardrails import filtra_input, filtra_output, verifica_grounding
load_dotenv()
MODELLO_LLM  = "gemini-2.5-flash"
TEMPERATURE  = 0.2
MAX_TOKENS   = 4096
DEBUG_MODE   = False
CARTELLA_LOG = "./logs"

def get_llm(temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS) -> ChatGoogleGenerativeAI:
    """
    Factory dell'LLM puntato su Google AI Studio (Gemini API).
    thinking_budget=0 disabilita il ragionamento interno di Gemini 2.5:
    senza questo, il "thinking" consuma max_output_tokens lasciando la
    risposta visibile troncata a meta frase.
    """
    return ChatGoogleGenerativeAI(
        model=MODELLO_LLM,
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_budget=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        max_retries=6,
        timeout=120,
    )
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
PROMPT_CLASSIFICAZIONE = """Analizza il seguente messaggio e classifica chi lo ha scritto,
SOLO se ci sono segnali chiari di ruolo. Profili possibili:
- genitore: linguaggio familiare, riferimenti a "mio figlio/figlia", preoccupazioni quotidiane
- insegnante: menziona scuola, classe, alunni, attivita didattiche
- pediatra: terminologia medica, riferimenti a pazienti, diagnosi, percentili
- nutrizionista: termini come macronutrienti, LARN, dieta, piano alimentare
- ricercatore: linguaggio accademico, riferimenti a studi, dati, ricerca, tesi
- generico: il messaggio e' troppo breve, vago o neutro per dedurre il ruolo
  con ragionevole certezza (es. poche parole chiave senza contesto sul mittente).
Nel dubbio scegli 'generico': NON tirare a indovinare un ruolo specifico.
Messaggio: "{messaggio}"
Rispondi con UNA SOLA PAROLA tra: genitore, insegnante, pediatra, nutrizionista, ricercatore, generico"""

PROFILO_GENERICO   = "generico"
TEMP_GENERICO      = 0.3
PROMPT_GENERICO = (
    "Sei un assistente virtuale specializzato in nutrizione pediatrica e prevenzione "
    "dell'obesita infantile (fascia 0-10 anni). Non e' chiaro chi sia l'interlocutore: "
    "usa un linguaggio chiaro e accessibile, spiegando gli eventuali termini tecnici, "
    "senza dare per scontato che sia un genitore, un medico o un altro ruolo specifico.\n\n"
    "REGOLE:\n"
    "1. Rispondi SOLO usando le informazioni nel [CONTESTO] qui sotto.\n"
    "2. Se la risposta non e nel contesto, dillo e suggerisci di consultare un esperto.\n"
    "3. Non fare diagnosi, non prescrivere terapie, non inventare dati.\n"
    "4. Usa elenchi puntati quando migliora la leggibilita.\n"
    "5. Cita il documento di riferimento alla fine della risposta.\n\n"
    "[CONTESTO]\n{context}\n[FINE CONTESTO]"
)
PROMPT_QUERY_REWRITING = """Sei un esperto di information retrieval applicato alla nutrizione pediatrica.
Riscrivi la domanda seguente come query in LINGUAGGIO NATURALE, ottimizzata per la
ricerca semantica in un database di documenti scientifici italiani su alimentazione
e obesita infantile (0-10 anni).
Vincoli OBBLIGATORI:
- NON usare operatori booleani (AND, OR, NOT) ne parentesi: produrrebbero risultati errati.
- Scrivi una frase o un breve elenco di termini in italiano corrente, non una formula.
- Espandi con sinonimi e termini tecnici pertinenti mantenendo l'intento reale.
- Una sola riga, massimo 30 parole.
Tieni conto che l'utente e un: {profilo}
Domanda originale: "{domanda}"
Rispondi SOLO con la query riscritta, senza spiegazioni o testo aggiuntivo."""

POLICY_RAG = (
    "REGOLE DI FEDELTA E CITAZIONE (PRIORITARIE SU QUALSIASI ALTRA ISTRUZIONE):\n"
    "1. Usa ESCLUSIVAMENTE le informazioni scritte nei blocchi [Fonte N] del contesto. "
    "Non aggiungere strategie, attivita, esempi, dati o raccomandazioni che non siano "
    "presenti testualmente nel contesto.\n"
    "2. Per citare, riporta SOLO il riferimento esatto nella forma [Fonte N], dove N e il "
    "numero del blocco da cui prendi l'informazione. NON inventare ne modificare titoli di "
    "documento o numeri di pagina. NON scrivere mai 'pag. 0'.\n"
    "3. Se il contesto NON contiene informazioni sufficienti, dichiaralo apertamente "
    "(es. 'Le fonti disponibili non trattano questo aspetto specifico') e NON colmare il "
    "vuoto con conoscenze esterne o esempi inventati.\n"
    "4. E preferibile una risposta breve e fedele alle fonti piuttosto che una risposta "
    "lunga, articolata ma non supportata dal contesto."
)

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
            "modello_retriever": f"{MODELLO_EMBED} k={RETRIEVER_K}",
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
    def __init__(self, retriever):
        """
        Crea un nuovo assistente leggero che riusa un retriever gia' inizializzato.
        Lo state per-sessione (cronologia, profilo) e' isolato per ogni istanza,
        mentre il retriever (modello embedding + vector store) viene condiviso a
        livello di processo tramite il singleton in crea_vector_db.
        """
        self.llm = None
        self.parser = StrOutputParser()
        self.retriever = retriever
        self.profilo_corrente: str | None = None
        self.cronologia: list = []
        # Ultimi chunk/query effettivamente usati per generare (post-rerank).
        # Esposti per valutazione/ispezione: i contesti misurati da RAGAS
        # devono coincidere con quelli realmente dati in pasto all'LLM.
        self.ultimi_docs: list = []
        self.ultima_query_riscritta: str = ""
        self.logger = SessionLogger()
    def _rileva_profilo(self, messaggio: str) -> str:
        """
        Classifica il mittente del messaggio. Restituisce una chiave di
        PROFILI se il ruolo e' chiaro, altrimenti PROFILO_GENERICO se il
        messaggio e' troppo vago per dedurlo (nessun profilo forzato).
        """
        prompt = ChatPromptTemplate.from_messages([
            ("human", PROMPT_CLASSIFICAZIONE.format(messaggio=messaggio))
        ])
        llm_temp  = get_llm(temperature=0.0, max_tokens=1024)
        chain     = prompt | llm_temp | self.parser
        risultato = chain.invoke({}).strip().lower()
        for profilo in PROFILI:
            if profilo in risultato:
                return profilo
        return PROFILO_GENERICO

    def _prepara_generazione(self, domanda: str) -> tuple:
        """
        Risolve profilo, system prompt e temperatura per la generazione.
        Se il profilo non e' ancora fissato lo rileva: con esito chiaro
        lo "blocca" su self.profilo_corrente; se vago resta None (registro
        neutro) cosi' un messaggio successivo piu' informativo lo ri-rileva.
        Returns:
            (system_prompt, temperatura, profilo_effettivo | None)
        """
        if self.profilo_corrente is None:
            rilevato = self._rileva_profilo(domanda)
            if rilevato in PROFILI:
                self.profilo_corrente = rilevato
        if self.profilo_corrente in PROFILI:
            dati = PROFILI[self.profilo_corrente]
            return dati["prompt"], dati["temperature"], self.profilo_corrente
        return PROMPT_GENERICO, TEMP_GENERICO, None
    def _riscrivi_query(self, domanda: str) -> str:
        """
        Riformula la domanda dell'utente in una query ottimizzata per la
        similarity search, tenendo conto del profilo corrente.
        """
        profilo_label = (
            PROFILI[self.profilo_corrente]["etichetta"]
            if self.profilo_corrente in PROFILI else "utente generico"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("human", PROMPT_QUERY_REWRITING.format(
                profilo=profilo_label,
                domanda=domanda,
            ))
        ])
        llm_temp       = get_llm(temperature=0.0, max_tokens=1024)
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
                    f"[Fonte {i}] {titolo}"
                    + (f" - pag. {pagina}" if pagina != "" else "")
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
        # Guardrail di input: blocco/redazione prima della pipeline.
        dec_in = filtra_input(domanda)
        if not dec_in.consenti:
            self.logger.log_interazione(
                profilo=self.profilo_corrente,
                query_originale=domanda,
                query_riscritta="",
                chunks=[],
                risposta=dec_in.messaggio_utente,
                latenza_ms=int((time.time() - t_start) * 1000),
            )
            print(f"\n[Guardrail input attivato: {', '.join(dec_in.triggered)}]")
            return dec_in.messaggio_utente
        domanda_pulita = dec_in.domanda_sanificata
        gia_fissato = self.profilo_corrente is not None
        system_prompt, temperatura, prof_eff = self._prepara_generazione(domanda_pulita)
        if not gia_fissato:
            if prof_eff in PROFILI:
                print(f"\nProfilo rilevato: {PROFILI[prof_eff]['etichetta']}")
                print("(Digita 'profilo' per cambiarlo manualmente)\n")
            else:
                print("\nProfilo non determinato (messaggio generico): "
                      "registro neutro.\n")
        query_riscritta = self._riscrivi_query(domanda_pulita)
        candidati = self.retriever.invoke(query_riscritta)
        if os.getenv("RAG_RERANK", "1") == "1":
            llm_rerank = get_llm(temperature=0.0, max_tokens=16)
            docs = rerank_chunks(candidati, query_riscritta, llm_rerank, k=RETRIEVER_K)
        else:
            # Comportamento pre-fix (per ablation/confronto): nessun rerank,
            # i candidati grezzi del retriever vanno dritti nel contesto.
            docs = candidati
        self.ultimi_docs = docs
        self.ultima_query_riscritta = query_riscritta
        contesto = self._formatta_contesto(docs)
        llm_profilo    = get_llm(temperature=temperatura, max_tokens=MAX_TOKENS)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("system", POLICY_RAG),
            MessagesPlaceholder(variable_name="cronologia"),
            ("human", "{domanda}"),
        ])
        chain    = prompt | llm_profilo | self.parser
        risposta = chain.invoke({
            "context":    contesto,
            "cronologia": self.cronologia,
            "domanda":    domanda_pulita,
        })
        # Runtime grounding check (LLM-judge) + filtra_output
        score_grounding = verifica_grounding(risposta, docs)
        dec_out = filtra_output(risposta, docs, score_grounding=score_grounding)
        risposta = dec_out.risposta
        if dec_out.triggered:
            print(f"\n[Guardrail output attivato: {', '.join(dec_out.triggered)} "
                  f"| grounding={score_grounding:.2f}]")
        self.cronologia.append(HumanMessage(content=domanda_pulita))
        self.cronologia.append(AIMessage(content=risposta))
        if len(self.cronologia) > 10:
            self.cronologia = self.cronologia[-10:]
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
                    print("Verifica i limiti su https://aistudio.google.com oppure riprova piu' tardi.")
            else:
                print(f"\nErrore: {e}")
                print("Riprovare o digitare 'reset'.")

def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        print("Errore: GOOGLE_API_KEY non trovata nel file .env")
        sys.exit(1)
    try:
        from crea_vector_db import get_o_crea_retriever
        assistente = AssistenteRAG(retriever=get_o_crea_retriever())
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
