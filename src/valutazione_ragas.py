"""
valutazione_ragas.py
Script di valutazione automatica del sistema RAG tramite la libreria RAGAS.
Carica il dataset di valutazione, esegue la pipeline RAG su ogni domanda
e calcola le metriche: faithfulness, answer_relevancy, context_precision.
I risultati vengono salvati in logs/ come file JSON e CSV.
Uso:
    python src/valutazione_ragas.py
    python src/valutazione_ragas.py --profilo genitore
    python src/valutazione_ragas.py --domande 10
"""
import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings
from crea_vector_db import get_o_crea_vector_db, get_retriever
from chatbot import AssistenteRAG, PROFILI, get_llm, MODELLO_LLM
load_dotenv()
PERCORSO_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "eval", "dataset_valutazione.json"
)
CARTELLA_LOG = "./logs"

def carica_dataset(percorso: str, filtro_profilo: str | None = None) -> list:
    with open(percorso, encoding="utf-8") as f:
        dataset = json.load(f)
    if filtro_profilo:
        dataset = [d for d in dataset if d["profilo"] == filtro_profilo]
    return dataset

def esegui_pipeline(campioni: list, max_domande: int | None = None) -> list:
    """
    Per ogni campione del dataset esegue la pipeline RAG completa:
    rilevamento profilo, query rewriting, retrieval, generazione.
    Restituisce una lista di dizionari con tutti i dati necessari a RAGAS.
    """
    if max_domande:
        campioni = campioni[:max_domande]
    db        = get_o_crea_vector_db()
    retriever = get_retriever(db)
    assistente = AssistenteRAG(retriever=retriever)
    risultati = []
    totale    = len(campioni)
    for i, campione in enumerate(campioni, 1):
        profilo      = campione["profilo"]
        domanda      = campione["domanda"]
        ground_truth = campione["ground_truth"]
        print(f"  [{i}/{totale}] Profilo: {profilo} — {domanda[:60]}...")
        try:
            assistente.profilo_corrente = profilo
            assistente.cronologia       = []
            query_riscritta = assistente._riscrivi_query(domanda)
            docs    = retriever.invoke(query_riscritta)
            contesti = [doc.page_content for doc in docs]
            risposta = assistente.rispondi(domanda)
            risultati.append({
                "profilo":         profilo,
                "user_input":      domanda,
                "query_riscritta": query_riscritta,
                "response":        risposta,
                "retrieved_contexts": contesti,
                "reference":       ground_truth,
            })
        except Exception as e:
            print(f"    Errore sul campione {i}: {e}")
            risultati.append({
                "profilo":            profilo,
                "user_input":         domanda,
                "query_riscritta":    "",
                "response":           "",
                "retrieved_contexts": [],
                "reference":          ground_truth,
                "errore":             str(e),
            })
    return risultati

def calcola_metriche(risultati: list) -> dict:
    """
    Costruisce l'EvaluationDataset RAGAS e calcola le metriche.
    Utilizza il modello LLM e il modello di embedding configurati.
    """
    campioni_validi = [r for r in risultati if r.get("response") and not r.get("errore")]
    if not campioni_validi:
        print("Nessun campione valido per la valutazione.")
        return {}
    print(f"\nCalcolo metriche RAGAS su {len(campioni_validi)} campioni...")
    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        for r in campioni_validi
    ]
    dataset = EvaluationDataset(samples=samples)
    llm        = get_llm(temperature=0.0, max_tokens=8192)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    run_config = RunConfig(
        timeout=180,
        max_retries=4,
        max_wait=30,
        max_workers=3,
    )
    risultato_ragas = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )
    return risultato_ragas

def salva_report(risultati_pipeline: list, metriche: dict) -> str:
    """
    Salva il report completo in JSON e CSV nella cartella logs/.
    Restituisce il percorso del file JSON.
    """
    os.makedirs(CARTELLA_LOG, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metriche_per_profilo = {}
    for r in risultati_pipeline:
        p = r["profilo"]
        if p not in metriche_per_profilo:
            metriche_per_profilo[p] = {"domande": 0, "errori": 0}
        metriche_per_profilo[p]["domande"] += 1
        if r.get("errore"):
            metriche_per_profilo[p]["errori"] += 1
    metriche_dict = {}
    if metriche:
        try:
            df = metriche.to_pandas()
            metriche_dict = {
                "faithfulness":     float(df["faithfulness"].mean()),
                "answer_relevancy": float(df["answer_relevancy"].mean()),
                "context_precision": float(df["context_precision"].mean()),
            }
        except Exception:
            metriche_dict = dict(metriche)
    report = {
        "timestamp":           datetime.now().isoformat(),
        "modello_llm":         MODELLO_LLM,
        "campioni_totali":     len(risultati_pipeline),
        "campioni_validi":     len([r for r in risultati_pipeline if not r.get("errore")]),
        "metriche_globali":    metriche_dict,
        "metriche_per_profilo": metriche_per_profilo,
        "dettaglio":           risultati_pipeline,
    }
    percorso_json = os.path.join(CARTELLA_LOG, f"ragas_{timestamp}.json")
    with open(percorso_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    percorso_csv = os.path.join(CARTELLA_LOG, f"ragas_{timestamp}.csv")
    with open(percorso_csv, "w", encoding="utf-8") as f:
        f.write("profilo,domanda,query_riscritta,risposta,faithfulness,answer_relevancy,context_precision\n")
        try:
            df = metriche.to_pandas() if metriche else None
        except Exception:
            df = None
        for i, r in enumerate(risultati_pipeline):
            faith = ans_rel = ctx_prec = ""
            if df is not None and i < len(df):
                faith    = df.iloc[i].get("faithfulness", "")
                ans_rel  = df.iloc[i].get("answer_relevancy", "")
                ctx_prec = df.iloc[i].get("context_precision", "")
            riga = '","'.join([
                r["profilo"],
                r["user_input"].replace('"', "'"),
                r.get("query_riscritta", "").replace('"', "'"),
                r.get("response", "").replace('"', "'")[:200],
                str(faith),
                str(ans_rel),
                str(ctx_prec),
            ])
            f.write(f'"{riga}"\n')
    print(f"\nReport salvato:")
    print(f"  JSON: {percorso_json}")
    print(f"  CSV : {percorso_csv}")
    return percorso_json

def stampa_riepilogo(metriche: dict) -> None:
    print("\n" + "=" * 50)
    print("  RISULTATI RAGAS")
    print("=" * 50)
    if not metriche:
        print("  Nessuna metrica disponibile.")
        return
    try:
        df = metriche.to_pandas()
        faith    = df["faithfulness"].mean()
        ans_rel  = df["answer_relevancy"].mean()
        ctx_prec = df["context_precision"].mean()
        print(f"  Faithfulness        : {faith:.3f}")
        print(f"  Answer Relevancy    : {ans_rel:.3f}")
        print(f"  Context Precision   : {ctx_prec:.3f}")
        print(f"  Media complessiva   : {(faith + ans_rel + ctx_prec) / 3:.3f}")
    except Exception as e:
        print(f"  Errore nel calcolo del riepilogo: {e}")
        print(f"  Metriche raw: {metriche}")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(
        description="Valutazione RAGAS del sistema RAG per l'obesita infantile."
    )
    parser.add_argument(
        "--profilo",
        choices=list(PROFILI.keys()),
        default=None,
        help="Filtra la valutazione su un profilo specifico.",
    )
    parser.add_argument(
        "--domande",
        type=int,
        default=None,
        help="Limita il numero di domande (taglia i primi N dal dataset).",
    )
    parser.add_argument(
        "--per-profilo",
        type=int,
        default=None,
        help="Prende N domande per ciascun profilo (totale = 5 * N). Coverage uniforme.",
    )
    args = parser.parse_args()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Errore: GOOGLE_API_KEY non trovata nel file .env")
        sys.exit(1)
    percorso_dataset = os.path.abspath(PERCORSO_DATASET)
    if not os.path.exists(percorso_dataset):
        print(f"Errore: dataset non trovato in {percorso_dataset}")
        print("Creare la cartella data/eval/ e inserire dataset_valutazione.json")
        sys.exit(1)
    print("=" * 50)
    print("  VALUTAZIONE RAGAS")
    print("=" * 50)
    if args.profilo:
        print(f"  Filtro profilo: {args.profilo}")
    if args.domande:
        print(f"  Limite domande: {args.domande}")
    print("\nCaricamento dataset...")
    campioni = carica_dataset(percorso_dataset, filtro_profilo=args.profilo)
    print(f"  Campioni caricati: {len(campioni)}")
    if args.per_profilo:
        bilanciati = []
        for prof in PROFILI:
            del_prof = [c for c in campioni if c["profilo"] == prof][:args.per_profilo]
            bilanciati.extend(del_prof)
        campioni = bilanciati
        print(f"  Campionamento bilanciato: {args.per_profilo}/profilo -> {len(campioni)} totali")
    print("\nEsecuzione pipeline RAG sul dataset...")
    risultati = esegui_pipeline(campioni, max_domande=args.domande)
    metriche = calcola_metriche(risultati)
    stampa_riepilogo(metriche)
    salva_report(risultati, metriche)
if __name__ == "__main__":
    main()
