"""
guardrails.py
Livello di sicurezza separato dalla pipeline RAG.

Espone due funzioni pure (no side-effect sul resto del sistema):
  - filtra_input(domanda)             - eseguito PRIMA della pipeline RAG
  - filtra_output(risposta, chunks)   - eseguito DOPO la generazione

Controlli implementati:
  INPUT (blocco o redazione)
    1. Prompt injection / jailbreak / esfiltrazione del system prompt
    2. Richiesta di diagnosi personalizzata su un bambino specifico
    3. Richiesta di prescrizione, dosaggio o terapia farmacologica
    4. Rilevamento PII (codice fiscale, email, telefono, tessera sanitaria)
       -> redazione in-place, non bloccante
    5. Domanda fuori dominio (no keyword rilevanti, non conversazionale)
  OUTPUT (riscrittura con disclaimer)
    6. Frasi che formulano diagnosi/prescrizioni nella risposta
    7. Affermazioni cliniche con dati numerici prive di citazione [Fonte N]

Ogni intervento viene registrato su file JSONL in CARTELLA_LOG per audit.
"""
import os
import re
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict

CARTELLA_LOG = "./logs"

# === PATTERN DI INPUT ============================================

_INJECTION_PATTERNS = [
    r"\bignora(re)?\s+(le|l[ae])?\s*(istruzion|regole|policy|tutte)",
    r"\bignore\s+(previous|all|the|your)\s+(instructions|rules)",
    r"\bdimentic[ah]?\s+(le|tutte\s+le|ogni)\s+(istruzion|regole)",
    r"\bforget\s+(your|the|previous|all)\s+(instructions|rules)",
    r"\bagisci\s+come\s+(se|fossi)\b",
    r"\bact\s+as\s+(if|a|an)\b",
    r"\bfingi\s+di\s+essere\b",
    r"\bpretend\s+(to\s+be|you'?re|you\s+are)\b",
    r"\brivel[ai]?\s+(il|la|le|lo)\s*(system\s+prompt|prompt|istruzion)",
    r"\bshow\s+me\s+(your|the)\s+(prompt|instructions|system)",
    r"\b(stampa|mostra)\s+(il)?\s*system\s*prompt\b",
    r"\bjailbreak\b",
    r"\bDAN\s+mode\b",
    r"\bdeveloper\s+mode\b",
    r"<\s*/?\s*system\s*>",
    r"^\s*system\s*:\s*",
]

# Richiesta di valutazione clinica personalizzata su un bambino specifico.
# Volutamente stretto: e' un'azione richiesta dall'utente sul SUO bambino,
# non un riferimento generico a "diagnosi" o "diagnosticare" che fa parte
# del lessico clinico legittimo del corpus.
_PERSONAL_DIAGNOSIS_PATTERNS = [
    # "mio figlio + ha/presenta/e' + obesita/sovrappeso/..."
    r"\b(mio|il\s+mio|nostr[oa]|la\s+mia)\s+(figli[oa]|bambin[oa]|nipote|alunn[oa])"
    r"[^.?!\n]{0,200}\b(ha|presenta|soffre|e['’]\s|pesa|misura|alt[oa]\s)"
    r"[^.?!\n]{0,80}\b(obes|sovrappes|patolog|diabete|sindrom|ipertens|disturb)",
    # imperativo rivolto all'assistente ("diagnosticami", "diagnosticamelo")
    r"\bdiagnostic(ami|amelo|amela|aci)\b",
    # "mio figlio e' obeso/sovrappeso" diretto
    r"\b(mio|il\s+mio)\s+(figli[oa]|bambin[oa])\s+(e['’]\s+|risulta\s+|sembra\s+)"
    r"(obes|sovrappes)",
]

# Richiesta di prescrizione / dosaggio / terapia
_PRESCRIPTION_REQUEST_PATTERNS = [
    r"\b(che|quale|cosa|qual[ei])\s+(farmaco|farmaci|medicina|medicine|integrator[ei]|"
    r"cura|terapia)\b",
    r"\bdosagg(io|i)\b",
    r"\bdose\s+(giornaliera|consigliata|massima|minima|al\s+giorno|raccomandata)\b",
    r"\bquant[ie]\s+(mg|grammi|gocce|compress|unit|cucchia)",
    r"\bprescriv(i|imi|ere|ete|ermi)\b",
    r"\bsomministr(a|argli|are|ate|azione)\b",
    r"\b(devo|posso)\s+(dargli|darle|fargli\s+prendere|somministrar)",
    r"\b(deve|dovrebbe)\s+(prendere|assumere)\s+(il|la|lo|un|una|del|della)\b",
]

# PII (Personally Identifiable Information)
_PII_PATTERNS = {
    "codice_fiscale": re.compile(
        r"\b[A-Z]{6}\d{2}[A-EHLMPRST]\d{2}[A-Z]\d{3}[A-Z]\b", re.IGNORECASE
    ),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "telefono": re.compile(r"\b(?:\+39\s?)?3\d{2}[\s.\-]?\d{6,7}\b"),
    "tessera_sanitaria": re.compile(r"\b\d{20}\b"),
}

# Dominio: termini che, se presenti, qualificano la domanda come in-topic.
# Volutamente specifico: verbi generici come "consigli" sono esclusi perche'
# triggererebbero su "consigliami un film".
_DOMAIN_KEYWORDS = [
    # obesita / peso / crescita
    r"obesit", r"sovrappeso", r"\bpeso\b", r"altezza", r"\bBMI\b", r"percentil",
    r"crescita", r"caloric", r"energetic", r"\bkcal\b", r"calorie",
    # alimentazione
    r"alimentazion", r"nutrizion", r"dieta", r"merenda", r"colazion",
    r"pranzo", r"cena", r"spuntin", r"mensa", r"\bcibo\b", r"alimenti",
    r"\bpasto\b", r"\bpasti\b", r"\btavola\b",
    r"frutta", r"verdura", r"pasta", r"\bpane\b", r"zuccher", r"\bgrass",
    r"protei", r"carboidrat", r"lipid", r"\bfibra\b", r"vitamin", r"minerali",
    r"larn", r"indice\s+glicemico", r"yogurt", r"latte", r"pesce", r"legumi",
    r"cottura", r"ristoraz", r"\bapporto\b", r"porzion",
    # contesto bambini / scuola / famiglia
    r"bambin", r"figli", r"alunn", r"infanzia", r"pediatri",
    r"lattant", r"neonat", r"svezzament",
    r"scuol", r"scolast", r"\bclasse\b", r"\bfamig", r"genitor",
    r"insegnant", r"educator", r"didattic",
    # attivita / sport / schermo
    r"attivit", r"\bfisic", r"\bsport\b", r"movimento", r"sedentari",
    r"schermi", r"\btv\b", r"videogioch",
    # linee guida / sistema
    r"linee\s+guida", r"raccomandazion", r"\bsalute\b", r"prevenzion",
    r"\bsan[oae]\b", r"abitudin", r"stile\s+di\s+vita", r"okkio", r"siedp",
    r"ministero\s+della\s+salute",
]

# Aperture conversazionali brevi: accettate anche senza termini di dominio
_CONVERSATIONAL_OPENINGS = {
    "ciao", "salve", "buongiorno", "buonasera", "buon giorno", "buona sera",
    "hey", "grazie", "ok", "perfetto", "aiuto", "help",
    "come stai", "chi sei", "cosa puoi fare", "cosa fai", "presentati",
    "cosa sai fare",
}

# === MESSAGGI DI RIFIUTO =========================================

_MSG_INJECTION = (
    "Non posso rispondere a questa richiesta. Posso aiutarti con domande "
    "sulla nutrizione e sulla prevenzione dell'obesita' infantile basate "
    "sulle linee guida disponibili."
)

_MSG_DIAGNOSI = (
    "Non posso formulare diagnosi ne' fornire valutazioni cliniche "
    "personalizzate sul peso, l'altezza o i sintomi specifici di un bambino: "
    "per questo e' indispensabile rivolgersi al **pediatra di fiducia**. "
    "Posso invece aiutarti con informazioni generali sui criteri di "
    "sovrappeso/obesita' previsti dall'OMS, sugli stili di vita raccomandati "
    "e sulle strategie di prevenzione."
)

_MSG_PRESCRIZIONE = (
    "Non posso indicare farmaci, dosaggi o terapie: la prescrizione e' di "
    "competenza esclusiva del **medico**, che valuta il singolo caso. Posso "
    "pero' fornirti informazioni generali sulle indicazioni nutrizionali e "
    "sulle linee guida di prevenzione."
)

_MSG_OFF_TOPIC = (
    "Sono un assistente specializzato in **nutrizione pediatrica e "
    "prevenzione dell'obesita' infantile** (fascia 0-10 anni). La tua "
    "domanda sembra non rientrare in questo dominio. Se invece riguarda "
    "l'alimentazione o lo stile di vita di un bambino, riformulala "
    "specificando il contesto."
)

# === DATA STRUCTURES =============================================

@dataclass
class InputDecision:
    """Esito del filtro di input."""
    consenti: bool
    domanda_sanificata: str
    messaggio_utente: str | None = None
    triggered: list = field(default_factory=list)

@dataclass
class OutputDecision:
    """Esito del filtro di output."""
    risposta: str
    triggered: list = field(default_factory=list)

# === HELPERS =====================================================

def _hit(patterns, testo: str) -> bool:
    return any(re.search(p, testo, re.IGNORECASE) for p in patterns)

def _is_injection(testo: str) -> bool:
    return _hit(_INJECTION_PATTERNS, testo)

def _is_personal_diagnosis(testo: str) -> bool:
    return _hit(_PERSONAL_DIAGNOSIS_PATTERNS, testo)

def _is_prescription_request(testo: str) -> bool:
    return _hit(_PRESCRIPTION_REQUEST_PATTERNS, testo)

def _redact_pii(testo: str) -> tuple[str, list[str]]:
    """Sostituisce i pattern PII con placeholder e ritorna l'elenco dei tipi."""
    triggered = []
    sanificata = testo
    for nome, pat in _PII_PATTERNS.items():
        if pat.search(sanificata):
            triggered.append(nome)
            sanificata = pat.sub(f"[DATO_PERSONALE_RIMOSSO:{nome}]", sanificata)
    return sanificata, triggered

def _is_conversational(testo: str) -> bool:
    t = testo.lower().strip(" !?.,\"'\n\r\t")
    if not t:
        return True
    if t in _CONVERSATIONAL_OPENINGS:
        return True
    if len(t.split()) <= 4 and any(t.startswith(g) for g in _CONVERSATIONAL_OPENINGS):
        return True
    return False

def _is_off_topic(testo: str) -> bool:
    """True se la domanda non contiene NESSUN termine di dominio
    e non e' una semplice formula conversazionale."""
    if _is_conversational(testo):
        return False
    for kw in _DOMAIN_KEYWORDS:
        if re.search(kw, testo, re.IGNORECASE):
            return False
    return True

# === FILTRO INPUT ================================================

def filtra_input(domanda: str) -> InputDecision:
    """Applica i 5 controlli di input nell'ordine di severita'.
    Restituisce InputDecision; logga sempre i trigger su file."""
    triggered: list = []

    if _is_injection(domanda):
        triggered.append("prompt_injection")
        dec = InputDecision(False, domanda, _MSG_INJECTION, triggered)
        _log_trigger("input", dec, domanda)
        return dec

    if _is_prescription_request(domanda):
        triggered.append("richiesta_prescrizione")
        dec = InputDecision(False, domanda, _MSG_PRESCRIZIONE, triggered)
        _log_trigger("input", dec, domanda)
        return dec

    if _is_personal_diagnosis(domanda):
        triggered.append("richiesta_diagnosi_personale")
        dec = InputDecision(False, domanda, _MSG_DIAGNOSI, triggered)
        _log_trigger("input", dec, domanda)
        return dec

    sanificata, pii_hits = _redact_pii(domanda)
    triggered.extend(f"pii:{t}" for t in pii_hits)

    if _is_off_topic(sanificata):
        triggered.append("off_topic")
        dec = InputDecision(False, sanificata, _MSG_OFF_TOPIC, triggered)
        _log_trigger("input", dec, domanda)
        return dec

    dec = InputDecision(True, sanificata, None, triggered)
    if triggered:
        _log_trigger("input", dec, domanda)
    return dec

# === FILTRO OUTPUT ===============================================

# Frasi che formulano una diagnosi sul bambino dell'utente
_OUTPUT_DIAGNOSIS_PATTERNS = [
    r"\b(tuo|il\s+tuo|vostro|nostro)\s+(figli[oa]|bambin[oa])\s+"
    r"(ha|presenta|e['’]\s+|soffre\s+di)\s+"
    r"(l['’]?\s*)?(obesit|sovrappeso|patolog|diabete|sindrom)",
    r"\bdiagnosi\s+e['’]?\s+(di\s+)?(obesit|sovrappeso|diabete)",
    r"\bha\s+l['’]?\s*obesit",
    r"\brisulta\s+(obeso|sovrappeso|in\s+obesit)",
]

# Frasi che danno una prescrizione/posologia specifica
_OUTPUT_PRESCRIPTION_PATTERNS = [
    r"\bsomministr(a|are|argli|argli|argli)\s+\d",
    r"\bdagli\s+\d+\s*(mg|grammi|gocce|compress|cucchia|ml)",
    r"\bprescriv\w*\s+(il|la|un[oa]?)\s+",
    r"\bdosaggio\s+(consigliato|raccomandato|da\s+assumere)\s+e['’]?\s+\d",
    r"\bdevi\s+dargli\s+\d",
    r"\bcompresse?\s+da\s+\d+\s*mg",
]

# Affermazioni cliniche con numeri (richiedono citazione [Fonte N])
_CLINICAL_NUMBER_PATTERNS = [
    r"\b\d{1,2}\s*°?\s*(?:esim[oa]\s+)?percentile\b",
    r"\bBMI\s+(?:>=?|maggiore|superiore|inferiore|<=?)\s+(?:al\s+|del\s+|di\s+)?\d",
    r"\b\d{2,4}\s*(?:kcal|kilocalor)",
    r"\b\d{1,3}\s*(?:mg|grammi|g|ml)\s+(?:al\s+giorno|/die|giornalier)",
    r"\b\d{1,2}\s*(?:porzioni?|volt[ei])\s+(?:al\s+giorno|a\s+settimana)\b",
]

_DISCLAIMER_OUTPUT = (
    "\n\n_Nota: questa risposta ha scopo informativo e non sostituisce il "
    "parere del pediatra. Per valutazioni cliniche personalizzate o decisioni "
    "terapeutiche rivolgiti al medico._"
)

_DISCLAIMER_GROUNDING = (
    "\n\n**Avviso di verifica:** un controllo automatico indica che parte di "
    "questa risposta potrebbe non essere pienamente supportata dalle fonti "
    "recuperate. Consulta sempre il pediatra per decisioni rilevanti."
)

# Soglia di grounding sotto la quale la risposta viene marcata come dubbia.
SOGLIA_GROUNDING = 0.6

# Prompt per l'LLM-judge di grounding (runtime).
_PROMPT_GROUNDING = """Sei un valutatore esperto di sistemi RAG.
Ti vengono forniti un CONTESTO con i passaggi recuperati dai documenti di
riferimento e una RISPOSTA generata da un assistente sulla base di quel
contesto.

Valuta in che misura le affermazioni della RISPOSTA sono effettivamente
supportate dal CONTESTO. Ignora le formule di cortesia e i disclaimer.
Rispondi con UN SOLO numero decimale tra 0.0 e 1.0:
- 1.0 = tutte le affermazioni sono direttamente supportate dal contesto
- 0.7 = la maggior parte delle affermazioni e' supportata
- 0.5 = circa meta' delle affermazioni e' supportata
- 0.2 = poche affermazioni sono supportate
- 0.0 = nessuna affermazione e' supportata

[CONTESTO]
{contesto}
[FINE CONTESTO]

[RISPOSTA]
{risposta}
[FINE RISPOSTA]

Punteggio (solo il numero, niente altro):"""

def verifica_grounding(risposta: str, chunks: list) -> float:
    """
    Calcola un punteggio di grounding 0.0-1.0 via LLM-judge.

    Politica fail-open: se la chiamata fallisce o l'output non e' parsabile,
    restituisce 1.0 per non bloccare il flusso dell'utente.
    Restituisce immediatamente 1.0 se la risposta o i chunk sono vuoti.
    """
    if not risposta or not risposta.strip() or not chunks:
        return 1.0
    contesto = "\n\n".join(
        f"[Fonte {i}] {getattr(d, 'page_content', str(d))[:800]}"
        for i, d in enumerate(chunks[:8], 1)
    )
    prompt_testo = _PROMPT_GROUNDING.format(
        contesto=contesto, risposta=risposta[:2000]
    )
    try:
        # import locale: chatbot importa guardrails, quindi evitiamo il ciclo
        from chatbot import get_llm
        llm = get_llm(temperature=0.0, max_tokens=64)
        out = llm.invoke(prompt_testo)
        testo = getattr(out, "content", str(out)).strip()
        m = re.search(r"\b([01](?:\.\d+)?|0?\.\d+)\b", testo)
        if not m:
            return 1.0
        valore = float(m.group(1))
        return max(0.0, min(1.0, valore))
    except Exception:
        return 1.0

def filtra_output(
    risposta: str,
    chunks_usati: list | None = None,
    score_grounding: float | None = None,
) -> OutputDecision:
    """Applica i controlli post-generazione.

    Args:
        risposta: testo generato dal modello.
        chunks_usati: chunk recuperati dal retriever (riservato a estensioni).
        score_grounding: punteggio di grounding pre-calcolato via
            verifica_grounding(); se < SOGLIA_GROUNDING aggiunge un trigger.
    """
    triggered: list = []

    if _hit(_OUTPUT_DIAGNOSIS_PATTERNS, risposta):
        triggered.append("output_diagnosi")
    if _hit(_OUTPUT_PRESCRIPTION_PATTERNS, risposta):
        triggered.append("output_prescrizione")

    has_clinical_numbers = _hit(_CLINICAL_NUMBER_PATTERNS, risposta)
    has_citation = bool(re.search(r"\[Fonte\s+\d+\]", risposta))
    if has_clinical_numbers and not has_citation:
        triggered.append("missing_citation")

    grounding_basso = (
        score_grounding is not None and score_grounding < SOGLIA_GROUNDING
    )
    if grounding_basso:
        triggered.append(f"grounding_low:{score_grounding:.2f}")

    out = risposta
    if grounding_basso:
        out = out + _DISCLAIMER_GROUNDING
    if any(t in ("output_diagnosi", "output_prescrizione", "missing_citation")
           for t in triggered):
        out = out + _DISCLAIMER_OUTPUT

    dec = OutputDecision(out, triggered)
    if triggered:
        _log_trigger("output", dec, risposta[:300])
    return dec

# === LOGGING (AUDIT) =============================================

_LOG_FILE = None

def _log_path() -> str:
    global _LOG_FILE
    if _LOG_FILE is None:
        os.makedirs(CARTELLA_LOG, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _LOG_FILE = os.path.join(CARTELLA_LOG, f"guardrail_{ts}.jsonl")
    return _LOG_FILE

def _log_trigger(tipo: str, decisione, payload_extra) -> None:
    try:
        with open(_log_path(), "a", encoding="utf-8") as f:
            record = {
                "ts": datetime.now().isoformat(),
                "tipo": tipo,
                "decisione": asdict(decisione),
                "payload": payload_extra,
            }
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass
