"""
valutazione_redteam.py
Valutazione adversariale del livello di guardrail (solo input rails).

Esegue filtra_input su un dataset di attacchi categorizzati e di domande
legittime, confronta l'esito (consenti/blocca) con quello atteso e calcola:
  - Attack Block Rate per categoria (% di attacchi correttamente bloccati)
  - False Positive Rate (% di domande legittime erroneamente bloccate)
  - confusion matrix complessiva.

I controlli di input sono pattern-based: la valutazione NON consuma token API.

Uso:
    python src/valutazione_redteam.py
"""
import os
import sys
import json
import csv
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guardrails import filtra_input

PERCORSO_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "eval", "redteam_dataset.json"
)
CARTELLA_LOG = "./logs"

# Categorie che rappresentano attacchi (atteso=block) vs benigne (atteso=pass).
CATEGORIE_ATTACCO = {
    "injection_diretta", "injection_parafrasata",
    "prescrizione_esplicita", "prescrizione_obliqua",
    "diagnosi_personale", "diagnosi_mascherata",
    "off_topic_netto", "off_topic_borderline",
}
CATEGORIE_BENIGNE = {"legittima_difficile", "pii"}


def carica_dataset(percorso: str) -> list:
    with open(percorso, encoding="utf-8") as f:
        return json.load(f)


def valuta(dataset: list) -> dict:
    """Esegue filtra_input su ogni entry e raccoglie gli esiti."""
    risultati = []
    for entry in dataset:
        dec = filtra_input(entry["testo"])
        bloccato = not dec.consenti
        atteso_block = entry["atteso"] == "block"
        corretto = bloccato == atteso_block
        risultati.append({
            "testo": entry["testo"],
            "categoria": entry["categoria"],
            "atteso": entry["atteso"],
            "esito": "block" if bloccato else "pass",
            "corretto": corretto,
            "triggered": list(dec.triggered),
        })
    return risultati


def aggrega(risultati: list) -> dict:
    """Calcola block rate / FP rate per categoria e metriche globali."""
    per_cat = defaultdict(lambda: {"tot": 0, "corretti": 0, "bloccati": 0})
    for r in risultati:
        c = per_cat[r["categoria"]]
        c["tot"] += 1
        c["corretti"] += int(r["corretto"])
        c["bloccati"] += int(r["esito"] == "block")

    # confusion matrix globale: il "positivo" e' definito dall'esito ATTESO
    # (atteso=block), non dalla categoria, perche' alcune entry borderline
    # sono volutamente legittime (atteso=pass) pur appartenendo a categorie
    # di attacco. Contare per categoria falserebbe i bypass/falsi positivi.
    tp = fp = tn = fn = 0
    for r in risultati:
        deve_bloccare = r["atteso"] == "block"
        bloccato = r["esito"] == "block"
        if deve_bloccare and bloccato:
            tp += 1
        elif deve_bloccare and not bloccato:
            fn += 1
        elif not deve_bloccare and bloccato:
            fp += 1
        else:
            tn += 1

    n_attacchi = tp + fn
    n_benigne = tn + fp
    return {
        "per_categoria": dict(per_cat),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "attack_block_rate": tp / n_attacchi if n_attacchi else 0.0,
        "false_positive_rate": fp / n_benigne if n_benigne else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def stampa_report(risultati: list, metriche: dict) -> None:
    print("=" * 64)
    print("  VALUTAZIONE RED-TEAM DEI GUARDRAIL (input rails)")
    print("=" * 64)
    print(f"\n{'CATEGORIA':<26}{'N':>4}{'BLOCCATI':>10}{'CORRETTI':>10}{'ACCURACY':>10}")
    print("-" * 64)
    for cat in sorted(metriche["per_categoria"]):
        c = metriche["per_categoria"][cat]
        acc = c["corretti"] / c["tot"] if c["tot"] else 0
        tag = "ATTACCO" if cat in CATEGORIE_ATTACCO else "benigna"
        print(f"{cat:<26}{c['tot']:>4}{c['bloccati']:>10}{c['corretti']:>10}{acc:>9.0%}  {tag}")

    cm = metriche["confusion"]
    print("\n" + "-" * 64)
    print("  CONFUSION MATRIX (positivo = attacco)")
    print("-" * 64)
    print(f"  Veri positivi  (attacco bloccato)  : {cm['tp']}")
    print(f"  Falsi negativi (attacco passato)   : {cm['fn']}  <- bypass")
    print(f"  Veri negativi  (benigna passata)   : {cm['tn']}")
    print(f"  Falsi positivi (benigna bloccata)  : {cm['fp']}  <- over-blocking")
    print("\n" + "-" * 64)
    print("  METRICHE GLOBALI")
    print("-" * 64)
    print(f"  Attack Block Rate (recall)   : {metriche['attack_block_rate']:.1%}")
    print(f"  False Positive Rate          : {metriche['false_positive_rate']:.1%}")
    print(f"  Precision                    : {metriche['precision']:.1%}")
    print("=" * 64)

    bypass = [r for r in risultati
              if r["atteso"] == "block" and r["esito"] == "pass"]
    if bypass:
        print(f"\n  BYPASS ({len(bypass)} attacchi non bloccati):")
        for r in bypass:
            print(f"    [{r['categoria']}] {r['testo'][:70]}")

    falsi_pos = [r for r in risultati
                 if r["atteso"] == "pass" and r["esito"] == "block"]
    if falsi_pos:
        print(f"\n  FALSI POSITIVI ({len(falsi_pos)} legittime bloccate):")
        for r in falsi_pos:
            print(f"    [{r['categoria']}] {r['testo'][:70]} -> {r['triggered']}")


def salva_report(risultati: list, metriche: dict) -> str:
    os.makedirs(CARTELLA_LOG, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    percorso_json = os.path.join(CARTELLA_LOG, f"redteam_{ts}.json")
    with open(percorso_json, "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": datetime.now().isoformat(),
             "metriche": metriche, "dettaglio": risultati},
            f, ensure_ascii=False, indent=2,
        )
    percorso_csv = os.path.join(CARTELLA_LOG, f"redteam_{ts}.csv")
    with open(percorso_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["categoria", "atteso", "esito", "corretto", "triggered", "testo"])
        for r in risultati:
            w.writerow([r["categoria"], r["atteso"], r["esito"],
                        r["corretto"], "|".join(r["triggered"]), r["testo"]])
    print(f"\nReport salvato:\n  JSON: {percorso_json}\n  CSV : {percorso_csv}")
    return percorso_json


def main():
    percorso = os.path.abspath(PERCORSO_DATASET)
    if not os.path.exists(percorso):
        print(f"Errore: dataset non trovato in {percorso}")
        sys.exit(1)
    dataset = carica_dataset(percorso)
    risultati = valuta(dataset)
    metriche = aggrega(risultati)
    stampa_report(risultati, metriche)
    salva_report(risultati, metriche)


if __name__ == "__main__":
    main()
