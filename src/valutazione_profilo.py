"""
valutazione_profilo.py
Valutazione dell'adattamento delle risposte al profilo dello stakeholder.

Il claim centrale del sistema e' che la stessa domanda, posta da profili
diversi, produce risposte adattate per registro, lessico e contenuto.
Questo script lo misura con due metodi indipendenti:

  1. BLIND PROFILE CLASSIFICATION
     Per ogni (domanda, profilo) si genera la risposta forzando il profilo.
     Un giudice LLM riceve SOLO il testo della risposta (senza sapere il
     profilo target) e deve indovinare a quale stakeholder e' rivolta.
     Se l'adattamento e' reale, la classificazione e' accurata.
     -> accuracy + matrice di confusione tra profili.

  2. RUBRICA DI APPROPRIATEZZA
     Un giudice LLM, conoscendo il profilo target, valuta su scala 1-5
     quanto la risposta sia adeguata a quel profilo per registro e lessico.
     -> punteggio medio per profilo.

Uso:
    python src/valutazione_profilo.py
    python src/valutazione_profilo.py --domande 2
"""
import os
import sys
import json
import csv
import argparse
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crea_vector_db import get_o_crea_retriever
from chatbot import AssistenteRAG, PROFILI, POLICY_RAG, get_llm

load_dotenv()

PERCORSO_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "eval", "profilo_dataset.json"
)
CARTELLA_LOG = "./logs"
PROFILI_KEYS = list(PROFILI.keys())

PROMPT_CLASSIFICAZIONE_CIECA = """Leggi la seguente risposta di un assistente
su nutrizione e prevenzione dell'obesita' infantile. La risposta e' stata
adattata a UNO specifico interlocutore. In base a registro, lessico, livello
tecnico e tipo di contenuto, indica a quale dei seguenti interlocutori e'
rivolta:
- genitore: linguaggio semplice, rassicurante, consigli pratici quotidiani
- insegnante: contesto scolastico, attivita' didattiche, gestione in classe
- pediatra: terminologia clinica, percentili, BMI, criteri diagnostici
- nutrizionista: macronutrienti, LARN, indici, dettaglio quantitativo
- ricercatore: registro accademico, dati epidemiologici, riferimenti, metodo

Rispondi con UNA SOLA PAROLA tra: genitore, insegnante, pediatra, nutrizionista, ricercatore.

RISPOSTA DA CLASSIFICARE:
{risposta}

Interlocutore (una sola parola):"""

PROMPT_RUBRICA = """Sei un valutatore esperto di comunicazione sanitaria.
Valuta quanto la seguente risposta sia ADEGUATA all'interlocutore target
indicato, considerando registro linguistico, lessico, livello tecnico e
tipo di contenuto.

Interlocutore target: {profilo} ({descrizione})

Scala di valutazione (rispondi con UN SOLO numero intero da 1 a 5):
1 = per niente adeguata al target
2 = poco adeguata
3 = parzialmente adeguata
4 = adeguata
5 = perfettamente adeguata al target

RISPOSTA DA VALUTARE:
{risposta}

Punteggio (solo il numero da 1 a 5):"""


def carica_domande(percorso: str, limite: int | None = None) -> list:
    with open(percorso, encoding="utf-8") as f:
        domande = json.load(f)
    return domande[:limite] if limite else domande


def genera_risposta(assistente, retriever, domanda: str, profilo: str) -> str:
    """Genera la risposta forzando il profilo, senza guardrail/grounding
    (vogliamo misurare la generazione adattata, non la sicurezza)."""
    assistente.profilo_corrente = profilo
    assistente.cronologia = []
    query = assistente._riscrivi_query(domanda)
    docs = retriever.invoke(query)
    contesto = assistente._formatta_contesto(docs)
    system_prompt = PROFILI[profilo]["prompt"]
    temperatura = PROFILI[profilo]["temperature"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", POLICY_RAG),
        MessagesPlaceholder(variable_name="cronologia"),
        ("human", "{domanda}"),
    ])
    llm = get_llm(temperature=temperatura)
    chain = prompt | llm | assistente.parser
    return chain.invoke({"context": contesto, "cronologia": [], "domanda": domanda})


def classifica_cieca(risposta: str) -> str:
    llm = get_llm(temperature=0.0, max_tokens=256)
    out = llm.invoke(PROMPT_CLASSIFICAZIONE_CIECA.format(risposta=risposta[:2500]))
    testo = getattr(out, "content", str(out)).strip().lower()
    for p in PROFILI_KEYS:
        if p in testo:
            return p
    return "sconosciuto"


def valuta_rubrica(risposta: str, profilo: str) -> int:
    llm = get_llm(temperature=0.0, max_tokens=256)
    out = llm.invoke(PROMPT_RUBRICA.format(
        profilo=profilo,
        descrizione=PROFILI[profilo]["descrizione"],
        risposta=risposta[:2500],
    ))
    testo = getattr(out, "content", str(out)).strip()
    for ch in testo:
        if ch in "12345":
            return int(ch)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Valutazione adattamento al profilo.")
    parser.add_argument("--domande", type=int, default=None,
                        help="Limita il numero di domande.")
    args = parser.parse_args()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Errore: GOOGLE_API_KEY non trovata nel file .env")
        sys.exit(1)

    domande = carica_domande(os.path.abspath(PERCORSO_DATASET), args.domande)
    retriever = get_o_crea_retriever()
    assistente = AssistenteRAG(retriever=retriever)

    print("=" * 64)
    print("  VALUTAZIONE ADATTAMENTO AL PROFILO")
    print("=" * 64)
    print(f"  Domande: {len(domande)} | Profili: {len(PROFILI_KEYS)} | "
          f"Risposte totali: {len(domande) * len(PROFILI_KEYS)}\n")

    dettaglio = []
    for i, domanda in enumerate(domande, 1):
        print(f"[{i}/{len(domande)}] {domanda}")
        for profilo in PROFILI_KEYS:
            risposta = genera_risposta(assistente, retriever, domanda, profilo)
            predetto = classifica_cieca(risposta)
            voto = valuta_rubrica(risposta, profilo)
            ok = "OK" if predetto == profilo else "  "
            print(f"    {profilo:<14} -> predetto: {predetto:<14} {ok} | rubrica: {voto}/5")
            dettaglio.append({
                "domanda": domanda,
                "profilo_target": profilo,
                "profilo_predetto": predetto,
                "match": predetto == profilo,
                "rubrica": voto,
                "risposta": risposta,
            })

    # --- metriche ---
    n = len(dettaglio)
    corretti = sum(1 for d in dettaglio if d["match"])
    accuracy = corretti / n if n else 0
    rubrica_media = sum(d["rubrica"] for d in dettaglio) / n if n else 0

    confusion = defaultdict(lambda: defaultdict(int))
    for d in dettaglio:
        confusion[d["profilo_target"]][d["profilo_predetto"]] += 1

    rubrica_per_profilo = defaultdict(list)
    for d in dettaglio:
        rubrica_per_profilo[d["profilo_target"]].append(d["rubrica"])

    print("\n" + "=" * 64)
    print("  RISULTATI")
    print("=" * 64)
    print(f"  Accuracy classificazione cieca : {accuracy:.1%} ({corretti}/{n})")
    print(f"  Rubrica media (1-5)            : {rubrica_media:.2f}")
    print("\n  Rubrica per profilo:")
    for p in PROFILI_KEYS:
        voti = rubrica_per_profilo[p]
        media = sum(voti) / len(voti) if voti else 0
        print(f"    {p:<14}: {media:.2f}/5")

    print("\n  Matrice di confusione (riga=target, colonna=predetto):")
    header = "  " + " " * 14 + "".join(f"{p[:5]:>8}" for p in PROFILI_KEYS)
    print(header)
    for t in PROFILI_KEYS:
        riga = f"  {t:<14}" + "".join(f"{confusion[t][p]:>8}" for p in PROFILI_KEYS)
        print(riga)
    print("=" * 64)

    # --- salvataggio ---
    os.makedirs(CARTELLA_LOG, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    percorso_json = os.path.join(CARTELLA_LOG, f"profilo_{ts}.json")
    with open(percorso_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "rubrica_media": rubrica_media,
            "rubrica_per_profilo": {p: sum(v) / len(v) if v else 0
                                    for p, v in rubrica_per_profilo.items()},
            "confusion": {t: dict(confusion[t]) for t in PROFILI_KEYS},
            "dettaglio": dettaglio,
        }, f, ensure_ascii=False, indent=2)
    percorso_csv = os.path.join(CARTELLA_LOG, f"profilo_{ts}.csv")
    with open(percorso_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domanda", "profilo_target", "profilo_predetto", "match", "rubrica"])
        for d in dettaglio:
            w.writerow([d["domanda"], d["profilo_target"], d["profilo_predetto"],
                        d["match"], d["rubrica"]])
    print(f"\nReport salvato:\n  JSON: {percorso_json}\n  CSV : {percorso_csv}")


if __name__ == "__main__":
    main()
