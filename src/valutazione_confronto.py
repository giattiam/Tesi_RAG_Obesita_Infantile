"""
valutazione_confronto.py
Confronto RAGAS "prima vs dopo" delle due correzioni al retrieval:
  - reranking reintrodotto      (toggle: RAG_RERANK)
  - fix collisione doc_id       (toggle: RAG_DOCID_LEGACY)

Esegue la stessa pipeline due volte sullo stesso sottoinsieme bilanciato del
dataset, una in configurazione "prima" (buggata) e una "dopo" (corretta), e
stampa una tabella comparativa. Il modello di embedding viene caricato una
sola volta (in-process), cosi' l'unico costo extra sono le chiamate LLM.

Uso:
    python src/valutazione_confronto.py --per-profilo 1
    python src/valutazione_confronto.py --per-profilo 2
"""
import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

from chatbot import PROFILI
from valutazione_ragas import (
    carica_dataset, esegui_pipeline, calcola_metriche,
    PERCORSO_DATASET, CARTELLA_LOG,
)
load_dotenv()

CONFIG = {
    "prima": {"RAG_RERANK": "0", "RAG_DOCID_LEGACY": "1"},
    "dopo":  {"RAG_RERANK": "1", "RAG_DOCID_LEGACY": "0"},
}
METRICHE = ["faithfulness", "answer_relevancy", "context_precision"]


def _campioni_bilanciati(percorso: str, per_profilo: int) -> list:
    campioni = carica_dataset(percorso)
    fuori = []
    for prof in PROFILI:
        fuori.extend([c for c in campioni if c["profilo"] == prof][:per_profilo])
    return fuori


def _media_metriche(metriche) -> dict:
    if not metriche:
        return {m: None for m in METRICHE}
    try:
        df = metriche.to_pandas()
        return {m: float(df[m].mean()) for m in METRICHE}
    except Exception:
        return {m: None for m in METRICHE}


def _esegui_configurazione(nome: str, campioni: list) -> dict:
    print("\n" + "#" * 60)
    print(f"#  CONFIGURAZIONE: {nome.upper()}  ->  {CONFIG[nome]}")
    print("#" * 60)
    for chiave, valore in CONFIG[nome].items():
        os.environ[chiave] = valore
    risultati = esegui_pipeline(campioni)
    metriche  = calcola_metriche(risultati)
    return _media_metriche(metriche)


def _stampa_tabella(prima: dict, dopo: dict) -> None:
    print("\n" + "=" * 64)
    print("  CONFRONTO RAGAS  (prima = buggato | dopo = corretto)")
    print("=" * 64)
    print(f"  {'Metrica':<20} {'Prima':>10} {'Dopo':>10} {'Delta':>12}")
    print("  " + "-" * 56)
    for m in METRICHE:
        p, d = prima.get(m), dopo.get(m)
        if p is None or d is None:
            print(f"  {m:<20} {'n/d':>10} {'n/d':>10} {'n/d':>12}")
            continue
        delta = d - p
        segno = "+" if delta >= 0 else ""
        print(f"  {m:<20} {p:>10.3f} {d:>10.3f} {segno + format(delta, '.3f'):>12}")
    medie = {n: [v for v in dd.values() if v is not None]
             for n, dd in (("prima", prima), ("dopo", dopo))}
    if medie["prima"] and medie["dopo"]:
        mp = sum(medie["prima"]) / len(medie["prima"])
        md = sum(medie["dopo"]) / len(medie["dopo"])
        print("  " + "-" * 56)
        dlt = md - mp
        print(f"  {'MEDIA':<20} {mp:>10.3f} {md:>10.3f} "
              f"{('+' if dlt >= 0 else '') + format(dlt, '.3f'):>12}")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(description="Confronto RAGAS prima/dopo i fix retrieval.")
    parser.add_argument("--per-profilo", type=int, default=1,
                        help="Domande per profilo (totale = 5 * N). Default 1.")
    args = parser.parse_args()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Errore: GOOGLE_API_KEY non trovata nel file .env")
        sys.exit(1)
    percorso = os.path.abspath(PERCORSO_DATASET)
    campioni = _campioni_bilanciati(percorso, args.per_profilo)
    print(f"Campioni selezionati: {len(campioni)} "
          f"({args.per_profilo}/profilo su {len(PROFILI)} profili)")

    prima = _esegui_configurazione("prima", campioni)
    dopo  = _esegui_configurazione("dopo", campioni)
    _stampa_tabella(prima, dopo)

    os.makedirs(CARTELLA_LOG, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    percorso_out = os.path.join(CARTELLA_LOG, f"confronto_ragas_{ts}.json")
    with open(percorso_out, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "campioni": len(campioni),
            "per_profilo": args.per_profilo,
            "config": CONFIG,
            "prima": prima,
            "dopo": dopo,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nReport salvato: {percorso_out}")


if __name__ == "__main__":
    main()
