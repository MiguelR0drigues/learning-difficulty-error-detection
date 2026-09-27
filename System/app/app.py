"""Pilot-ready web prototype for the thesis system (Ch.4 demonstrator).

Run from System/ (so module paths resolve):
    python app/app.py
Then open http://localhost:5000

- /aluno/<nome>   student view: one writing task + two math tasks per round
- /professor      teacher dashboard: per-student difficulty profiles

Design constraints (deliberate, documented for the thesis):
- All data stays in a local SQLite file (app/pilot.db) -- no cloud, no
  accounts; a real school pilot would add consent + pseudonymization on top.
- UI is Portuguese (deployment context); FREE writing tasks are answered in
  ENGLISH because the linguistic model is English-trained (Ch.5 limitation).
  The SCREENING mode, however, also ships a European Portuguese item bank
  (/rastreio/<nome>?lang=pt) -- closed-world scoring needs no corpus and no
  trained model, so the PT port only required PT items + a PT phonetic
  comparator (screening/pt_g2p.py). Free-text PT support remains future work.
- Detected labels only are stored alongside raw responses; the teacher view
  is formative ("perfil de erros"), never diagnostic.
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
import sys
import time
from collections import Counter
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "math"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "linguistic"))

from flask import Flask, redirect, request, render_template_string, url_for

from mal_rules import (detect_subtraction_error, detect_fraction_add_error,
                        generate_subtraction_problem, generate_fraction)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "screening"))
from item_bank import build_worksheet
from scorer import score_worksheet, flag_against_norm

DB = os.environ.get("PILOT_DB", os.path.join(os.path.dirname(__file__), "pilot.db"))
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "linguistic", "final_model")
LABEL_LIST = ["O", "B-PHONO", "I-PHONO", "B-ORTHO", "I-ORTHO", "B-SEG", "I-SEG"]
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL_PT = {"PHONO": "Fonológico", "ORTHO": "Ortográfico", "SEG": "Segmentação",
            "SFL": "Subtração: menor-do-maior", "BORROW_ZERO": "Empréstimo com zero",
            "BORROW_NO_DEC": "Empréstimo sem decremento", "FRAC_ADD": "Fração: soma independente",
            "UNKNOWN_ERROR": "Erro não classificado"}

WRITING_PROMPTS = [
    "Write two sentences about your family.",
    "Write two sentences about what you did yesterday.",
    "Write two sentences about your favourite food.",
    "Write two sentences about your school.",
    "Write two sentences about a place you would like to visit.",
]

app = Flask(__name__)
_model = None
_tokenizer = None


def db():
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS responses(
        id INTEGER PRIMARY KEY, student TEXT, ts REAL, kind TEXT,
        payload TEXT, labels TEXT)""")
    return conn


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return True
    if not os.path.isdir(MODEL_DIR):
        return False
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    _model.eval()
    return True


def detect_text_errors(text: str) -> list:
    """Returns [(span_text, TYPE), ...]."""
    import torch
    if not load_model():
        return []
    spans = []
    # Tokenize like the training corpora: punctuation as separate tokens.
    # (Stripping the final period caused systematic sentence-final SEG
    # false positives -- the model never saw unpunctuated sentences.)
    import re as _re
    sentences = [s for s in _re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    for sentence in sentences:
        tokens = _re.findall(r"[A-Za-z0-9'’-]+|[^\sA-Za-z0-9]", sentence)
        if not tokens:
            continue
        enc = _tokenizer(tokens, is_split_into_words=True, truncation=True,
                         max_length=64, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(_model(**enc).logits[0], dim=-1)
        # Confidence threshold: the model's real-data precision is ~0.61
        # (Ch.5), too trigger-happy for a teacher-facing tool. Only report
        # spans predicted with >= CONF confidence; empirically (see thesis)
        # 0.7 removes most clean-sentence false positives while keeping the
        # majority of true detections. Tunable per deployment.
        CONF = float(os.environ.get("DETECT_CONF", "0.55"))
        word_pred = {}
        for wid, p in zip(enc.word_ids(), probs):
            if wid is not None and wid not in word_pred:
                top = int(p.argmax())
                word_pred[wid] = ID2LABEL[top] if (top != 0 and float(p[top]) >= CONF) else "O"
        cur = None
        for i, tok in enumerate(tokens):
            tag = word_pred.get(i, "O")
            if tag.startswith("B-"):
                if cur:
                    spans.append(tuple(cur))
                cur = [tok, tag[2:]]
            elif tag.startswith("I-") and cur and cur[1] == tag[2:]:
                cur[0] += " " + tok
            else:
                if cur:
                    spans.append(tuple(cur))
                cur = None
        if cur:
            spans.append(tuple(cur))
    return spans


def new_round():
    rng = random.Random()
    a, b = generate_subtraction_problem(digits=rng.choice([2, 3]), force_borrow=True, rng=rng)
    n1, d1 = generate_fraction(rng=rng)
    n2, d2 = generate_fraction(rng=rng)
    return {"prompt": rng.choice(WRITING_PROMPTS), "sub_a": a, "sub_b": b,
            "f_n1": n1, "f_d1": d1, "f_n2": n2, "f_d2": d2}


STUDENT_HTML = """<!doctype html><html lang="pt"><head><meta charset="utf-8">
<title>Exercícios — {{ nome }}</title><style>
body{font-family:system-ui,sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem;color:#222}
.card{border:1px solid #ddd;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}
input[type=text],textarea{width:100%;padding:.5rem;font-size:1rem;border:1px solid #bbb;border-radius:6px}
button{background:#2563eb;color:#fff;border:0;border-radius:6px;padding:.6rem 1.4rem;font-size:1rem;cursor:pointer}
.fb{background:#f0fdf4;border-color:#86efac}.fb.err{background:#fef2f2;border-color:#fca5a5}
small{color:#666}</style></head><body>
<h2>Olá, {{ nome }} 👋</h2>
{% if feedback %}<div class="card fb {{ 'err' if feedback_err else '' }}">{{ feedback|safe }}</div>{% endif %}
<form method="post">
<div class="card"><b>1. Escrita</b> <small>(responde em inglês)</small><br><br>
{{ round.prompt }}<br><br>
<textarea name="texto" rows="3" required></textarea></div>
<div class="card"><b>2. Subtração</b><br><br>
{{ round.sub_a }} − {{ round.sub_b }} = <input type="text" name="sub" size="8" required style="width:8rem">
</div>
<div class="card"><b>3. Frações</b><br><br>
{{ round.f_n1 }}/{{ round.f_d1 }} + {{ round.f_n2 }}/{{ round.f_d2 }} =
<input type="text" name="frac" placeholder="ex: 5/6" required style="width:8rem">
</div>
{% for k,v in round.items() %}<input type="hidden" name="r_{{k}}" value="{{v}}">{% endfor %}
<button>Enviar respostas</button></form>
<p><small>As tuas respostas ajudam o professor a perceber onde precisas de apoio. Dados guardados apenas neste computador.</small></p>
</body></html>"""

TEACHER_HTML = """<!doctype html><html lang="pt"><head><meta charset="utf-8">
<title>Painel do professor</title><style>
body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#222}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.5rem .7rem;text-align:left}
th{background:#f8fafc}.bar{display:inline-block;height:.8rem;background:#2563eb;border-radius:3px;vertical-align:middle}
.tag{display:inline-block;background:#eff6ff;color:#1d4ed8;border-radius:4px;padding:.1rem .4rem;margin:.1rem;font-size:.85rem}
small{color:#666}</style></head><body>
<h2>Painel do professor</h2>
<p><small>Perfis de <b>padrões de erro</b> para apoio formativo — não são diagnósticos.
Novo aluno: abrir <code>/aluno/&lt;nome&gt;</code>.</small></p>
{% if not students %}<p>Ainda sem respostas.</p>{% endif %}
{% for s in students %}
<h3>{{ s.nome }} <small>({{ s.n }} respostas, {{ s.nerr }} erros)</small></h3>
<table><tr><th>Tipo de erro</th><th>Ocorrências</th><th style="width:45%">Proporção</th></tr>
{% for lab, c, pct in s.rows %}
<tr><td>{{ lab }}</td><td>{{ c }}</td>
<td><span class="bar" style="width:{{ pct*3 }}px"></span> {{ "%.0f"|format(pct) }}%</td></tr>
{% endfor %}</table>
{% if s.dominant %}<p>Padrão dominante: <span class="tag">{{ s.dominant }}</span>
{% if s.persistent %}<span class="tag" style="background:#fef3c7;color:#92400e">persistente em várias sessões</span>{% endif %}</p>{% endif %}
{% endfor %}
{% if screenings %}
<h3>Rastreios (fichas diagnósticas)</h3>
<table><tr><th>Aluno</th><th>ORTHO-trap</th><th>PHONO-transp.</th><th>SEG</th><th>Matemática</th><th>Sinal</th></tr>
{% for s in screenings %}
<tr><td>{{ s.nome }}</td><td>{{ s.ortho }}</td><td>{{ s.phono }}</td><td>{{ s.seg }}</td><td>{{ s.math }}</td>
<td>{% for f in s.flags %}<span class="tag" style="background:#fee2e2;color:#991b1b">{{ f.family }} z={{ f.z }}</span>{% endfor %}</td></tr>
{% endfor %}</table>
<p><small>Sinal = desvio &ge; 1.5 desvios-padrão acima da média da turma nessa família de itens.</small></p>
{% endif %}
</body></html>"""


@app.route("/")
def index():
    return redirect(url_for("teacher"))


@app.route("/aluno/<nome>", methods=["GET", "POST"])
def student(nome):
    feedback, feedback_err = None, False
    if request.method == "POST":
        rnd = {k[2:]: request.form[k] for k in request.form if k.startswith("r_")}
        labels, msgs = [], []
        # writing
        texto = request.form.get("texto", "").strip()
        spans = detect_text_errors(texto)
        for span, etype in spans:
            labels.append(etype)
            msgs.append(f"✏️ <b>{span}</b> — {LABEL_PT.get(etype, etype)}")
        # subtraction
        try:
            sub_ans = int(request.form.get("sub", "").strip())
            lab = detect_subtraction_error(int(rnd["sub_a"]), int(rnd["sub_b"]), sub_ans)
            if lab != "CORRECT":
                labels.append(lab)
                msgs.append(f"🔢 Subtração — {LABEL_PT.get(lab, lab)}")
        except ValueError:
            msgs.append("🔢 Subtração: resposta inválida")
        # fraction
        try:
            frac_ans = Fraction(request.form.get("frac", "").strip())
            lab = detect_fraction_add_error(int(rnd["f_n1"]), int(rnd["f_d1"]),
                                            int(rnd["f_n2"]), int(rnd["f_d2"]), frac_ans)
            if lab != "CORRECT":
                labels.append(lab)
                msgs.append(f"➗ Frações — {LABEL_PT.get(lab, lab)}")
        except (ValueError, ZeroDivisionError):
            msgs.append("➗ Frações: resposta inválida (usa a forma a/b)")
        conn = db()
        conn.execute("INSERT INTO responses(student, ts, kind, payload, labels) VALUES(?,?,?,?,?)",
                     (nome.lower(), time.time(), "round",
                      json.dumps({"texto": texto, "round": rnd}), json.dumps(labels)))
        conn.commit(); conn.close()
        feedback_err = bool(labels)
        feedback = ("Tudo certo, bom trabalho! 🎉" if not labels
                    else "Obrigado! Nota para o próximo round:<br>" + "<br>".join(msgs))
    return render_template_string(STUDENT_HTML, nome=nome, round=new_round(),
                                  feedback=feedback, feedback_err=feedback_err)


SCREENING_HTML = """<!doctype html><html lang="pt"><head><meta charset="utf-8">
<title>Rastreio — {{ nome }}</title><style>
body{font-family:system-ui,sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem;color:#222}
.card{border:1px solid #ddd;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}
input[type=text]{padding:.4rem;font-size:1rem;border:1px solid #bbb;border-radius:6px}
button{background:#7c3aed;color:#fff;border:0;border-radius:6px;padding:.6rem 1.4rem;font-size:1rem;cursor:pointer}
table{border-collapse:collapse}td{padding:.3rem .6rem}
.fb{background:#f5f3ff;border-color:#c4b5fd}small{color:#666}</style></head><body>
<h2>Ficha de rastreio — {{ nome }} <small>[{{ 'português' if ws.lang == 'pt' else 'inglês' }}]</small></h2>
{% if result %}<div class="card fb"><b>Resultado registado.</b><br>
{% for fam, rate in result.family_rates.items() %}{{ fam }}: {{ "%.0f"|format(rate*100) }}% erros<br>{% endfor %}
Matemática: {{ result.math_labels|join(", ") }}</div>{% endif %}
<p><small>O professor dita cada palavra; o aluno escreve. Depois as contas.</small></p>
<form method="post">
<div class="card"><b>Ditado</b> <small>(professor lê a palavra {{ ws.dictation|length }}x)</small><br><br>
<table>{% for it in ws.dictation %}
<tr><td>{{ loop.index }}.</td><td><input type="text" name="d{{ loop.index0 }}" required></td></tr>
{% endfor %}</table></div>
<div class="card"><b>Subtrações</b><br><br>
{% for s in ws.subtraction %}{{ s.a }} − {{ s.b }} = <input type="text" name="s{{ loop.index0 }}" style="width:8rem" required><br><br>{% endfor %}</div>
<div class="card"><b>Frações</b><br><br>
{% for f in ws.fractions %}{{ f.n1 }}/{{ f.d1 }} + {{ f.n2 }}/{{ f.d2 }} = <input type="text" name="f{{ loop.index0 }}" placeholder="a/b" style="width:8rem" required><br><br>{% endfor %}</div>
<input type="hidden" name="seed" value="{{ ws.seed }}">
<input type="hidden" name="lang" value="{{ ws.lang }}">
<button>Registar rastreio</button></form>
<p><small>Classificação determinística (alvo conhecido) — sem modelo neural, sem gap de domínio. Sinal formativo, não diagnóstico.</small></p>
</body></html>"""


@app.route("/rastreio/<nome>", methods=["GET", "POST"])
def screening(nome):
    result = None
    # Language of the linguistic items: ?lang=pt selects the European
    # Portuguese item bank + pt_g2p comparator (Ch.4 Portuguese port);
    # default remains "en" (the configuration evaluated in Ch.5). Math
    # items are language-independent.
    lang = (request.form.get("lang") or request.args.get("lang") or "en").lower()
    lang = lang if lang in ("en", "pt") else "en"
    seed = int(request.form.get("seed", 0)) if request.method == "POST" else 0
    ws = build_worksheet(seed=seed, lang=lang)
    if request.method == "POST":
        answers = {
            "dictation": [request.form.get(f"d{i}", "") for i in range(len(ws["dictation"]))],
            "subtraction": [request.form.get(f"s{i}", "") for i in range(len(ws["subtraction"]))],
            "fractions": [request.form.get(f"f{i}", "") for i in range(len(ws["fractions"]))],
        }
        result = score_worksheet(ws, answers)  # lang travels inside ws
        conn = db()
        conn.execute("INSERT INTO responses(student, ts, kind, payload, labels) VALUES(?,?,?,?,?)",
                     (nome.lower(), time.time(), "screening",
                      json.dumps({"answers": answers, "seed": seed, "lang": lang}),
                      json.dumps({"family_rates": result["family_rates"],
                                   "math_labels": result["math_labels"],
                                   "detail": result["dictation_detail"]})))
        conn.commit(); conn.close()
        class _R: pass
        r = _R(); r.family_rates = result["family_rates"]; r.math_labels = result["math_labels"]
        result = r
        ws = build_worksheet(seed=seed + 1, lang=lang)
    return render_template_string(SCREENING_HTML, nome=nome, ws=ws, result=result)


@app.route("/professor")
def teacher():
    conn = db()
    rows = conn.execute("SELECT student, ts, labels FROM responses ORDER BY student, ts").fetchall()
    conn.close()
    by_student, screenings = {}, {}
    for student_, ts, labels in rows:
        parsed = json.loads(labels)
        if isinstance(parsed, dict):  # screening record
            screenings.setdefault(student_, []).append(parsed)
            continue
        by_student.setdefault(student_, []).append((ts, parsed))
    students = []
    for nome, hist in sorted(by_student.items()):
        counts = Counter(l for _, labs in hist for l in labs)
        total = sum(counts.values())
        # persistence: dominant type present in >=2 distinct sessions (gap > 30 min)
        sessions, last_ts = [], None
        for ts, labs in hist:
            if last_ts is None or ts - last_ts > 1800:
                sessions.append([])
            sessions[-1].extend(labs); last_ts = ts
        dominant = counts.most_common(1)[0][0] if counts else None
        persistent = dominant and sum(1 for s in sessions if dominant in s) >= 2
        students.append({
            "nome": nome.title(), "n": len(hist), "nerr": total,
            "rows": [(LABEL_PT.get(l, l), c, 100 * c / total) for l, c in counts.most_common()],
            "dominant": LABEL_PT.get(dominant, dominant) if dominant else None,
            "persistent": persistent})
    # screening aggregation: latest screening per student + class norms
    import statistics as _st
    fam_all = {"ORTHO_TRAP": [], "PHONO_TRANSPARENT": [], "SEG_COMPOUND": []}
    latest = {}
    for st_, recs in screenings.items():
        latest[st_] = recs[-1]
        for fam, r in recs[-1].get("family_rates", {}).items():
            fam_all.setdefault(fam, []).append(r)
    class_mu = {f: (_st.mean(v) if v else 0.0) for f, v in fam_all.items()}
    class_sd = {f: (_st.pstdev(v) if len(v) > 1 else 0.15) for f, v in fam_all.items()}
    screening_rows = []
    for st_, rec in sorted(latest.items()):
        fr = rec.get("family_rates", {})
        flags = flag_against_norm(fr, class_mu, class_sd)
        screening_rows.append({
            "nome": st_.title(),
            "ortho": f"{fr.get('ORTHO_TRAP', 0)*100:.0f}%",
            "phono": f"{fr.get('PHONO_TRANSPARENT', 0)*100:.0f}%",
            "seg": f"{fr.get('SEG_COMPOUND', 0)*100:.0f}%",
            "math": ", ".join(l for l in rec.get("math_labels", []) if l != "CORRECT") or "ok",
            "flags": flags})
    return render_template_string(TEACHER_HTML, students=students, screenings=screening_rows)


if __name__ == "__main__":
    print("Aluno:     http://localhost:5000/aluno/<nome>")
    print("Professor: http://localhost:5000/professor")
    app.run(debug=False, port=5000)
