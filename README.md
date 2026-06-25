# Assistente RAG per la prevenzione dell'obesità infantile

Agente conversazionale in lingua italiana basato su **Retrieval-Augmented
Generation (RAG)** per il supporto informativo sulla prevenzione dell'obesità
infantile (fascia 0–10 anni), rivolto a più categorie di stakeholder e fondato
su linee guida e documenti scientifici italiani.

La pipeline RAG è integrata in un'architettura di sicurezza (**guardrail**) a
difesa in profondità che previene la generazione di contenuti clinicamente
inappropriati.

> Progetto di tesi. Strumento di supporto informativo: **non sostituisce il
> parere medico**.

---

## Caratteristiche principali

- **Retrieval ibrido** — similarity search vettoriale (ChromaDB) combinata con
  ricerca per parole chiave BM25, con normalizzazione e fusione pesata dei
  punteggi.
- **Chunking semantico** — i documenti sono divisi dove cambia il contenuto
  (`SemanticChunker`), con intestazione contestuale del documento anteposta a
  ogni chunk per migliorare il recupero dei documenti meno rappresentati.
- **Adattamento al profilo** — rilevamento automatico (o selezione manuale) di
  5 profili di interlocutore (genitore, insegnante, pediatra, nutrizionista,
  ricercatore): registro, lessico e contenuto vengono adattati.
- **Query rewriting** tramite LLM per ottimizzare il recupero.
- **Citazione delle fonti** — le risposte citano i passaggi recuperati nella
  forma `[Fonte N]` e si fondano esclusivamente sul contesto.
- **Architettura di guardrail**:
  - *input rails*: prompt injection/jailbreak, richieste di
    prescrizione/dosaggio, richieste di diagnosi personalizzata, redazione di
    dati personali (PII), enforcement del dominio;
  - *grounding check* a tempo di esecuzione (LLM-judge, politica fail-open);
  - *output rails*: rilevamento di formulazioni diagnostiche/prescrittive,
    enforcement della citazione, disclaimer su basso ancoraggio;
  - **audit log** JSONL e **dashboard** di sicurezza in-app.
- **Doppia interfaccia** — CLI testuale e applicazione web Streamlit (con
  persistenza delle conversazioni, pannello fonti e dashboard).
- **Valutazione multi-prospettica** — qualità delle risposte (RAGAS),
  robustezza dei guardrail (red-teaming), adattamento al profilo.

## Stack tecnologico

| Componente      | Tecnologia |
|-----------------|------------|
| LLM             | Google Gemini 2.5 Flash (Google AI Studio) |
| Embeddings      | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (locale, CPU) |
| Vector store    | ChromaDB |
| Ricerca keyword | BM25 (`rank-bm25`) |
| Orchestrazione  | LangChain |
| UI web          | Streamlit |
| Valutazione     | RAGAS |

## Risultati della valutazione

Valori ottenuti sul corpus e sui dataset di valutazione inclusi nel repository
(vedi i report in `logs/`).

| Aspetto valutato | Metrica | Valore |
|------------------|---------|--------|
| Fedeltà alle fonti (RAGAS) | Faithfulness | **0,934** |
| Pertinenza (RAGAS) | Answer Relevancy | 0,764 |
| Precisione del contesto (RAGAS) | Context Precision | 0,427 |
| Robustezza guardrail (red-team) | Attack Block Rate | 78,8% |
| Robustezza guardrail (red-team) | False Positive Rate | 8,8% |
| Difesa in profondità | Bypass neutralizzati a valle | 11/11 |
| Adattamento al profilo | Rubrica di appropriatezza (1–5) | 4,15 |

---

## Requisiti

- **Python 3.12** (testato su Windows 11)
- Una chiave API gratuita di [Google AI Studio](https://aistudio.google.com/app/apikey)

## Installazione

```bash
# 1. Ambiente virtuale
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2. Dipendenze runtime (chatbot CLI + web + creazione indice)
pip install -r requirements.txt

# 3. Chiave API
copy .env.example .env          # Windows  (cp su Linux/macOS)
# poi aprire .env e inserire la propria GOOGLE_API_KEY
```

### Dipendenze opzionali: valutazione RAGAS

`ragas` richiede `datasets >= 4`, che a sua volta richiede `pyarrow >= 21`; su
Windows però `pyarrow >= 24` causa un crash e l'applicazione è pinnata a
`pyarrow == 19.0.1`. I due vincoli non sono risolvibili insieme da `pip`,
quindi le dipendenze di valutazione si installano a parte (vedi
`requirements-eval.txt`):

```bash
pip install ragas==0.4.3 datasets==4.8.5 pandas==3.0.3
pip install --no-deps "pyarrow==19.0.1"   # ri-forza la pin Windows-safe
```

---

## Utilizzo

### 1. Creare il vector store (solo la prima volta)

Indicizza i documenti presenti in `data/raw/`. Se `chroma_db/` esiste già,
viene caricato senza ri-indicizzare.

```bash
python src/crea_vector_db.py
```

### 2. Avviare l'assistente

```bash
# Interfaccia da riga di comando
python src/chatbot.py

# Interfaccia web (consigliata)
streamlit run src/app.py
```

### 3. Valutazioni

```bash
# Qualità delle risposte (RAGAS)
python src/valutazione_ragas.py                  # intero dataset
python src/valutazione_ragas.py --per-profilo 3  # 3 domande per profilo
python src/valutazione_ragas.py --profilo genitore

# Robustezza dei guardrail (red-teaming) — non consuma token API
python src/valutazione_redteam.py

# Adattamento al profilo (classificazione cieca + rubrica)
python src/valutazione_profilo.py
python src/valutazione_profilo.py --domande 2

# Confronto A/B di configurazioni del retrieval
python src/valutazione_confronto.py --per-profilo 1
```

Tutti i report vengono salvati in `logs/` come file JSON e CSV.

### 4. Test

```bash
pytest
```

---

## Struttura del progetto

```
.
├── data/
│   ├── raw/                          # documenti sorgente (PDF: linee guida, OKkio, ...)
│   └── eval/
│       ├── dataset_valutazione.json  # dataset RAGAS (40 domande, 8 per profilo)
│       ├── redteam_dataset.json      # attacchi + domande legittime (red-team)
│       └── profilo_dataset.json      # domande neutre cross-profilo
├── src/
│   ├── document_loader.py            # caricamento PDF + chunking semantico
│   ├── crea_vector_db.py             # ChromaDB + BM25 + retriever ibrido
│   ├── chatbot.py                    # pipeline RAG + interfaccia CLI
│   ├── app.py                        # interfaccia web Streamlit + dashboard
│   ├── guardrails.py                 # livello di sicurezza (input/output/grounding)
│   ├── valutazione_ragas.py          # valutazione qualità (RAGAS)
│   ├── valutazione_redteam.py        # valutazione robustezza guardrail
│   ├── valutazione_profilo.py        # valutazione adattamento al profilo
│   └── valutazione_confronto.py      # confronto A/B configurazioni retrieval
├── tests/
│   ├── test_smoke.py                 # smoke test della pipeline
│   └── test_guardrails.py            # test del livello di sicurezza
├── tesi/                             # sorgenti LaTeX della tesi
├── chroma_db/                        # vector store persistito (generato)
├── logs/                             # report di valutazione e audit (generato)
├── requirements.txt                  # dipendenze runtime
├── requirements-eval.txt             # dipendenze per la valutazione RAGAS
├── .env.example
└── README.md
```

## Architettura della pipeline

```
Utente
  │
  ▼
filtra_input (guardrails)        ── blocco / redazione PII / log
  │  injection · prescrizione · diagnosi · PII · enforcement dominio
  ▼
Rilevamento profilo  →  Query rewriting  →  Retrieval ibrido (Chroma + BM25)
  │
  ▼
Generazione (Gemini 2.5 Flash + POLICY_RAG + system prompt del profilo)
  │
  ▼
verifica_grounding (LLM-judge runtime, fail-open)
  │
  ▼
filtra_output (guardrails)       ── disclaimer / citation enforcement / log
  │
  ▼
Risposta  →  UI (chip di sicurezza, pannello fonti, dashboard)
```

## Note

- Il modello di embedding (~500 MB) viene scaricato al primo avvio e caricato
  una sola volta per processo.
- Il pin `pyarrow==19.0.1` è obbligatorio: le versioni `>= 24` causano un crash
  su Windows (access violation `0xc0000005`).
- I documenti in `data/raw/` devono mantenere l'estensione `.pdf` (o `.txt` /
  `.md`): i file senza estensione vengono ignorati dall'indicizzazione.
- Le interazioni e gli interventi dei guardrail sono registrati in `logs/` per
  finalità di audit; le conversazioni dell'interfaccia web sono persistite in
  `logs/conversazioni.json`.

## Licenza e ambito

Progetto sviluppato a scopo di tesi e ricerca. I contenuti generati hanno
finalità esclusivamente informative e non costituiscono parere medico,
diagnosi o prescrizione.
