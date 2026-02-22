import streamlit as st
import streamlit.components.v1 as components
import hashlib, uuid, re, html as html_module, time, os, json, datetime
import PyPDF2

st.set_page_config(
    page_title="QuizGenius AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── ENV / CONFIG ─────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama3-8b-8192"

# ─── OPTIONAL HEAVY DEPS (graceful fallback if not installed) ─────────────────
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# ─── SESSION STATE DEFAULTS ───────────────────────────────────────────────────
_DEFAULTS = {
    "logged_in": False, "current_user": None, "auth_mode": "login",
    "users": {},          # in-memory user store
    "user_data": {},      # per-user score/bookmark data
    "questions": [], "test_questions": [], "has_generated": False,
    "has_test_generated": False, "quiz_key": 0, "pdf_text": "",
    "pdf_filename": "", "pdf_size": 0, "pdf_hash": "", "questions_pdf_hash": "",
    "user_answers": {}, "test_submitted": False, "current_page": "Home",
    "selected_difficulty": None, "vector_store_texts": [], "chunks": [],
    "test_started_at": None, "generation_done": False,
    "bookmarks": [], "wrong_answers": [], "score_history": [],
    "detected_difficulty": "Medium", "topics": [], "selected_topics": [],
    "q_type": "MCQ", "fc_idx": 0, "fc_filter": "All",
    "timed_mode": False, "per_q_time": 45,
    "preview_question": None, "show_preview": False,
    "groq_key_input": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─── HELPERS ──────────────────────────────────────────────────────────────────
S = lambda t: html_module.escape(str(t))

def go(page):
    st.session_state.current_page = page
    st.rerun()

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def get_groq_key():
    """Return API key from env first, then session input."""
    return GROQ_API_KEY or st.session_state.get("groq_key_input", "")

# ─── AUTH (in-memory) ─────────────────────────────────────────────────────────
def do_login(username, password):
    users = st.session_state.users
    if username not in users:
        return False, "User not found."
    if users[username]["pw"] != hash_pw(password):
        return False, "Incorrect password."
    st.session_state.logged_in = True
    st.session_state.current_user = username
    ud = st.session_state.user_data.get(username, {})
    st.session_state.score_history  = ud.get("score_history", [])
    st.session_state.bookmarks      = ud.get("bookmarks", [])
    st.session_state.wrong_answers  = ud.get("wrong_answers", [])
    return True, "ok"

def do_signup(username, password, display_name=""):
    if len(username) < 3:  return False, "Username must be at least 3 characters."
    if len(password) < 6:  return False, "Password must be at least 6 characters."
    if username in st.session_state.users:
        return False, "Username already taken."
    st.session_state.users[username] = {
        "pw": hash_pw(password),
        "name": display_name or username,
        "created": datetime.datetime.now().strftime("%Y-%m-%d")
    }
    return do_login(username, password)

def persist_user_data():
    u = st.session_state.current_user
    if not u or u == "__guest__": return
    st.session_state.user_data[u] = {
        "score_history": st.session_state.score_history,
        "bookmarks":     st.session_state.bookmarks,
        "wrong_answers": st.session_state.wrong_answers,
    }

# ─── PDF ──────────────────────────────────────────────────────────────────────
def get_pdf_text(f):
    try:
        reader = PyPDF2.PdfReader(f)
        text = "".join(p.extract_text() or "" for p in reader.pages)
        if len(text.strip()) < 100 and OCR_AVAILABLE:
            st.warning("Scanned PDF detected — running OCR...")
            f.seek(0)
            try:
                imgs = convert_from_bytes(f.read())
                return "".join(pytesseract.image_to_string(i) + "\n" for i in imgs)
            except Exception as e:
                st.error(f"OCR failed: {e}")
                return text
        return text
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return ""

def calc_max_q(wc):
    if wc < 500:   return 10
    elif wc < 1000: return 25
    elif wc < 2000: return 50
    elif wc < 5000: return 100
    return 150

def detect_difficulty(text):
    words = text.split()
    if not words: return "Medium"
    avg_wl  = sum(len(w) for w in words) / len(words)
    sents   = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    cpx     = sum(1 for w in words if len(w) > 9) / len(words) * 100
    score   = avg_wl * 1.5 + (len(words) / max(len(sents), 1)) * 0.15 + cpx * 0.4
    return "Easy" if score < 12 else "Hard" if score > 20 else "Medium"

def extract_topics(text):
    topics = []
    for line in text.split('\n'):
        line = line.strip()
        if not (3 < len(line) < 90): continue
        if re.match(r'^(chapter|section|unit|topic|part|module)\s+\d+', line, re.IGNORECASE):
            topics.append(line)
        elif re.match(r'^\d+[\.\)]\s+[A-Z]', line):
            topics.append(line)
        elif line.isupper() and 1 < len(line.split()) <= 8:
            topics.append(line)
    return list(dict.fromkeys(topics))[:25]

def clear_pdf_state():
    for k, v in [
        ("pdf_text",""),("pdf_filename",""),("pdf_size",0),("pdf_hash",""),
        ("questions_pdf_hash",""),("questions",[]),("test_questions",[]),
        ("has_generated",False),("has_test_generated",False),("generation_done",False),
        ("vector_store_texts",[]),("chunks",[]),("selected_difficulty",None),
        ("user_answers",{}),("test_submitted",False),("preview_question",None),
        ("show_preview",False),("topics",[]),("selected_topics",[]),
    ]:
        st.session_state[k] = v

# ─── LLM / GENERATION ─────────────────────────────────────────────────────────
def call_groq(prompt: str) -> str:
    key = get_groq_key()
    if not key:
        raise ValueError("No Groq API key found. Please enter it in the sidebar.")
    if not GROQ_AVAILABLE:
        raise ImportError("groq package not installed. Run: pip install groq")
    client = Groq(api_key=key)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    return resp.choices[0].message.content

def simple_similarity_search(query: str, texts: list, k: int = 6) -> list:
    """Lightweight keyword similarity — no heavy embeddings needed."""
    query_words = set(query.lower().split())
    scored = []
    for t in texts:
        words = set(t.lower().split())
        score = len(query_words & words) / max(len(query_words), 1)
        scored.append((score, t))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:k]]

def build_prompt(ctx, q_type, diff="Medium"):
    lvl = {"Easy": "BASIC recall", "Medium": "COMPREHENSION", "Hard": "ANALYSIS"}.get(diff, "COMPREHENSION")
    if q_type == "TF":
        return (f"Create ONE True/False question ({lvl}) based on the context below.\n"
                f"Output ONLY this exact format:\n"
                f"Question: [statement]\nAnswer: True\n\nContext:\n{ctx[:1500]}")
    elif q_type == "FIB":
        return (f"Create ONE fill-in-the-blank MCQ ({lvl}). Use ___ for the blank.\n"
                f"Output ONLY this exact format:\n"
                f"Question: [sentence with ___]\nA) [correct answer]\nB) [wrong]\nC) [wrong]\nD) [wrong]\n"
                f"Correct Answer: A\n\nContext:\n{ctx[:1500]}")
    return (f"Generate ONE multiple choice question ({lvl}) based on the context.\n"
            f"Output ONLY this exact format:\n"
            f"Question: [question text]\nA) [option]\nB) [option]\nC) [option]\nD) [option]\n"
            f"Correct Answer: [A/B/C/D]\n\nContext:\n{ctx[:1500]}")

def clean_opt(opt):
    raw = opt.strip()
    txt = raw[2:].strip() if len(raw) > 2 else raw
    for stop in ("Context:", "Correct Answer:", "Question:"):
        if stop.lower() in txt.lower():
            txt = txt[:txt.lower().index(stop.lower())].strip()
    return txt.splitlines()[0].strip() if txt else "--"

def parse_mcq(raw):
    question = ""; options = {}; correct = "A"
    for line in raw.splitlines():
        line = line.strip()
        if not line: continue
        if line.lower().startswith("question:"):
            question = line[line.index(":") + 1:].strip()
        elif re.match(r'^[A-D]\)', line):
            letter = line[0]; txt = line[2:].strip()
            for stop in ("Context:", "Correct Answer:", "Question:"):
                if stop.lower() in txt.lower():
                    txt = txt[:txt.lower().index(stop.lower())].strip()
            if txt: options[letter] = f"{letter}) {txt.splitlines()[0].strip()}"
        elif re.match(r'^Correct Answer\s*:', line, re.IGNORECASE):
            m = re.search(r'[A-D]', line)
            if m: correct = m.group(0)
    return question or "Parsing error", [options.get(l, f"{l}) --") for l in ("A","B","C","D")], correct

def parse_tf(raw):
    question = ""; correct = "True"
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("question:"):
            question = line[line.index(":") + 1:].strip()
        elif re.match(r'^answer\s*:', line, re.IGNORECASE):
            correct = "True" if "true" in line.lower() else "False"
    return question or "Parsing error", ["A) True", "B) False"], "A" if correct == "True" else "B"

def parse_fib(raw):
    question = ""; options = {}; correct = "A"
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("question:"):
            question = line[line.index(":") + 1:].strip()
        elif re.match(r'^[A-D]\)', line):
            letter = line[0]; txt = line[2:].strip().splitlines()[0].strip()
            if txt: options[letter] = f"{letter}) {txt}"
        elif re.match(r'^correct\s*answer\s*:', line, re.IGNORECASE):
            m = re.search(r'[A-D]', line)
            if m: correct = m.group(0)
    return question or "Parsing error", [options.get(l, f"{l}) --") for l in ("A","B","C","D")], correct

def llm_generate(ctx, q_type, diff="Medium"):
    raw = call_groq(build_prompt(ctx, q_type, diff))
    if q_type == "TF":   q, opts, c = parse_tf(raw)
    elif q_type == "FIB": q, opts, c = parse_fib(raw)
    else:                  q, opts, c = parse_mcq(raw)
    return {"question": q, "options": opts, "correct": c, "context": ctx[:400], "type": q_type, "difficulty": diff}

def save_score(diff, correct, total, pdf_name):
    pct = round(correct / max(total, 1) * 100, 1)
    st.session_state.score_history.append({
        "date":  datetime.datetime.now().strftime("%b %d %H:%M"),
        "diff":  diff, "score": correct, "total": total, "pct": pct, "pdf": pdf_name
    })
    st.session_state.wrong_answers = [
        q for i, q in enumerate(st.session_state.test_questions)
        if st.session_state.user_answers.get(i) != q["correct"]
    ]
    persist_user_data()

# ─── EXPORT ───────────────────────────────────────────────────────────────────
def export_html(questions, title):
    rows = ""
    for i, q in enumerate(questions):
        opts = "".join(
            f'<li style="padding:.45rem .875rem;border-radius:7px;margin:.25rem 0;'
            f'background:{"#ecfdf5" if o[0]==q["correct"] else "#f8fafc"};'
            f'border-left:3px solid {"#10b981" if o[0]==q["correct"] else "transparent"};'
            f'color:{"#064e3b" if o[0]==q["correct"] else "#64748b"};font-size:.875rem;">'
            f'{S(o)}</li>'
            for o in q["options"] if o.strip()
        )
        rows += (
            f'<div style="margin-bottom:1.75rem;padding:1.5rem;border:1px solid #e2e8f0;'
            f'border-radius:12px;background:white;">'
            f'<div style="font-size:.6rem;font-weight:700;text-transform:uppercase;color:#94a3b8;margin-bottom:.5rem;">'
            f'Q{i+1} · {q.get("type","MCQ")}</div>'
            f'<div style="font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:.875rem;">{S(q["question"])}</div>'
            f'<ul style="list-style:none;padding:0;margin:0;">{opts}</ul></div>'
        )
    return (
        f'<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{S(title)}</title>'
        f'<style>body{{font-family:sans-serif;max-width:800px;margin:3rem auto;padding:0 1.5rem;background:#f0f2f5;}}</style>'
        f'</head><body><h1 style="color:#0d1b2a;">{S(title)}</h1>'
        f'<p style="color:#64748b;">{datetime.datetime.now().strftime("%B %d, %Y")} · QuizGenius AI</p>'
        f'{rows}</body></html>'
    )

def get_fc_qs():
    qs = st.session_state.questions; f = st.session_state.fc_filter
    if f == "Bookmarked":
        return [(i, q) for i, q in enumerate(qs) if i in st.session_state.bookmarks]
    if f == "Mistakes":
        wa = {q["question"] for q in st.session_state.wrong_answers}
        return [(i, q) for i, q in enumerate(qs) if q["question"] in wa]
    return list(enumerate(qs))

# ─── RICH COMPONENTS ──────────────────────────────────────────────────────────
def render_flashcard(question, answer, card_num, total, is_bookmarked=False):
    pct = int(card_num / total * 100)
    bm_color = "#f59e0b" if is_bookmarked else "#94a3b8"
    html_str = f"""<!DOCTYPE html><html><head>
    <style>
      *{{margin:0;padding:0;box-sizing:border-box;}}
      body{{font-family:'DM Sans',sans-serif;background:transparent;padding:0;overflow:hidden;}}
      .prog{{width:100%;height:3px;background:#e8e8f0;border-radius:999px;margin-bottom:16px;overflow:hidden;}}
      .prog-fill{{height:100%;background:linear-gradient(90deg,#6c63ff,#a78bfa);border-radius:999px;width:{pct}%;}}
      .meta{{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;}}
      .meta-num{{font-size:.58rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#9ca3af;background:#f4f4f8;padding:3px 10px;border-radius:6px;}}
      .scene{{perspective:1200px;width:100%;height:230px;cursor:pointer;}}
      .card{{width:100%;height:100%;position:relative;transform-style:preserve-3d;transition:transform .6s cubic-bezier(.175,.885,.32,1.1);}}
      .card.flipped{{transform:rotateY(180deg);}}
      .face{{position:absolute;inset:0;backface-visibility:hidden;border-radius:18px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:1.75rem;}}
      .front{{background:white;border:1.5px solid #ede9fe;box-shadow:0 8px 32px rgba(108,99,255,.1);}}
      .back{{background:linear-gradient(145deg,#1a1035 0%,#2d1b69 100%);transform:rotateY(180deg);box-shadow:0 16px 48px rgba(108,99,255,.3);}}
      .q-label{{font-size:.55rem;font-weight:700;text-transform:uppercase;letter-spacing:.14em;color:#9ca3af;margin-bottom:.75rem;}}
      .q-text{{font-size:.98rem;font-weight:600;color:#1f1735;line-height:1.6;text-align:center;max-width:500px;}}
      .a-badge{{background:rgba(108,99,255,.2);border:1px solid rgba(108,99,255,.3);color:#c4b5fd;font-size:.58rem;font-weight:700;padding:3px 12px;border-radius:999px;margin-bottom:.875rem;text-transform:uppercase;letter-spacing:.08em;}}
      .a-text{{font-size:.95rem;font-weight:600;color:white;line-height:1.6;text-align:center;max-width:500px;}}
      .hint{{position:absolute;bottom:12px;font-size:.6rem;color:#9ca3af;font-style:italic;display:flex;align-items:center;gap:4px;}}
    </style>
    </head><body>
    <div class="prog"><div class="prog-fill"></div></div>
    <div class="meta">
      <span class="meta-num">Card {card_num} of {total}</span>
      <span style="color:{bm_color};font-size:1rem;">{'★' if is_bookmarked else '☆'}</span>
    </div>
    <div class="scene" onclick="this.querySelector('.card').classList.toggle('flipped')">
      <div class="card">
        <div class="face front">
          <div class="q-label">Question</div>
          <div class="q-text">{S(question)}</div>
          <div class="hint">🔄 Click to reveal answer</div>
        </div>
        <div class="face back">
          <div class="a-badge">Answer</div>
          <div class="a-text">{S(answer)}</div>
        </div>
      </div>
    </div>
    </body></html>"""
    components.html(html_str, height=300, scrolling=False)

def render_score_chart(score_history):
    if len(score_history) < 2: return
    recent = score_history[-12:]
    labels = json.dumps([s["date"] for s in recent])
    data   = json.dumps([s["pct"]  for s in recent])
    colors = json.dumps([
        "rgba(16,185,129,.85)" if s["pct"] >= 80
        else "rgba(108,99,255,.85)" if s["pct"] >= 60
        else "rgba(239,68,68,.85)"
        for s in recent
    ])
    html_str = f"""<!DOCTYPE html><html><head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
    <style>*{{margin:0;padding:0;box-sizing:border-box;}}body{{background:white;padding:16px;font-family:sans-serif;}}</style>
    </head><body>
    <canvas id="chart" height="150"></canvas>
    <script>
    new Chart(document.getElementById('chart').getContext('2d'),{{
      type:'line',
      data:{{
        labels:{labels},
        datasets:[{{label:'Score %',data:{data},borderColor:'#6c63ff',backgroundColor:'rgba(108,99,255,.07)',
          borderWidth:2.5,pointBackgroundColor:{colors},pointBorderColor:'white',pointBorderWidth:2,
          pointRadius:6,fill:true,tension:0.42}}]
      }},
      options:{{responsive:true,plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#1a1035',padding:10,cornerRadius:8}}}},
        scales:{{y:{{min:0,max:100,ticks:{{callback:v=>v+'%',color:'#9ca3af',font:{{size:10}}}},grid:{{color:'rgba(0,0,0,.04)'}}}},
          x:{{ticks:{{color:'#9ca3af',font:{{size:9}},maxRotation:30}},grid:{{display:false}}}}}}}}
    }});
    </script></body></html>"""
    components.html(html_str, height=220, scrolling=False)

def render_score_result(pct, correct, total):
    verdict = "Outstanding! 🌟" if pct >= 80 else "Good Work! 👍" if pct >= 60 else "Keep Studying! 📚"
    v_sub   = ("You've mastered this material." if pct >= 80
               else "Solid performance — a bit more review and you'll nail it." if pct >= 60
               else "Review the flashcards and study mode to reinforce concepts.")
    ring_color = "#10b981" if pct >= 80 else "#6c63ff" if pct >= 60 else "#ef4444"
    do_confetti = "true" if pct >= 80 else "false"
    html_str = f"""<!DOCTYPE html><html><head>
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.2/dist/confetti.browser.min.js"></script>
    <style>
    *{{margin:0;padding:0;box-sizing:border-box;}}
    body{{font-family:'DM Sans',sans-serif;background:linear-gradient(160deg,#1a1035,#2d1b69);border-radius:20px;overflow:hidden;}}
    .hero{{padding:2.5rem 2rem 2rem;text-align:center;position:relative;}}
    .dots{{position:absolute;inset:0;background-image:radial-gradient(circle at 2px 2px,rgba(255,255,255,.06) 1px,transparent 0);background-size:22px 22px;pointer-events:none;}}
    .ring{{width:100px;height:100px;border-radius:50%;border:3px solid {ring_color};background:rgba(255,255,255,.05);
      display:flex;align-items:center;justify-content:center;margin:0 auto 1.25rem;position:relative;}}
    .ring::before{{content:'';position:absolute;inset:-6px;border-radius:50%;border:1px solid {ring_color};opacity:.3;animation:pulse 2s ease-in-out infinite;}}
    @keyframes pulse{{0%,100%{{transform:scale(1);opacity:.3;}}50%{{transform:scale(1.08);opacity:.6;}}}}
    .pct{{font-size:1.9rem;font-weight:800;color:white;line-height:1;}}
    .verdict{{font-size:1.5rem;font-weight:800;color:white;margin-bottom:.375rem;}}
    .sub{{color:rgba(255,255,255,.55);font-size:.85rem;line-height:1.7;max-width:360px;margin:0 auto 1.25rem;}}
    .chips{{display:flex;gap:.75rem;justify-content:center;flex-wrap:wrap;padding-bottom:2rem;}}
    .chip{{padding:.35rem 1rem;border-radius:999px;font-size:.75rem;font-weight:700;}}
    .ok{{background:rgba(16,185,129,.2);color:#6ee7b7;border:1px solid rgba(16,185,129,.3);}}
    .ng{{background:rgba(239,68,68,.2);color:#fca5a5;border:1px solid rgba(239,68,68,.3);}}
    .info{{background:rgba(255,255,255,.1);color:rgba(255,255,255,.65);border:1px solid rgba(255,255,255,.15);}}
    </style></head><body>
    <div class="hero">
      <div class="dots"></div>
      <div class="ring"><div class="pct"><span id="ctr">0</span>%</div></div>
      <div class="verdict">{S(verdict)}</div>
      <p class="sub">{S(v_sub)}</p>
      <div class="chips">
        <span class="chip ok">✅ {correct} Correct</span>
        <span class="chip ng">❌ {total - correct} Wrong</span>
        <span class="chip info">📝 {total} Total</span>
      </div>
    </div>
    <script>
    var t={pct:.0f},e=document.getElementById('ctr'),c=0,s=Math.max(1,Math.ceil(t/60));
    var iv=setInterval(function(){{c=Math.min(c+s,t);e.textContent=Math.round(c);if(c>=t)clearInterval(iv);}},16);
    if({do_confetti}){{
      setTimeout(function(){{
        confetti({{particleCount:120,spread:80,origin:{{y:0.4}},colors:['#6c63ff','#a78bfa','#10b981','#f59e0b']}});
        setTimeout(function(){{confetti({{particleCount:60,angle:60,spread:55,origin:{{x:0,y:0.5}}}});}},300);
        setTimeout(function(){{confetti({{particleCount:60,angle:120,spread:55,origin:{{x:1,y:0.5}}}});}},600);
      }},200);
    }}
    </script></body></html>"""
    components.html(html_str, height=340, scrolling=False)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=Sora:wght@700;800;900&display=swap');

/* ── RESETS ── */
#MainMenu,footer,header,.stDeployButton{visibility:hidden!important;display:none!important;}
[data-testid="collapsedControl"],section[data-testid="stSidebar"]{display:none!important;}
.stApp>header{display:none!important;height:0!important;}
html,body,.stApp,.main,.block-container,[data-testid="stAppViewBlockContainer"],
[data-testid="stAppViewContainer"],[data-testid="stMainBlockContainer"],
section.main>div:first-child,div[class*="block-container"]{padding-top:0!important;margin-top:0!important;}
.block-container{max-width:100%!important;padding-bottom:0!important;}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(108,99,255,.25);border-radius:999px;}

/* ── VARIABLES ── */
:root{
  --p:#6c63ff;--pd:#5a52e8;--plight:#ede9fe;--plighter:#f5f3ff;
  --bg:#f8f7ff;--w:#ffffff;--bd:#e8e4f8;
  --tx:#1a1035;--mu:#6b7280;--fa:#9ca3af;
  --gr:#10b981;--re:#ef4444;--am:#f59e0b;
  --grad:linear-gradient(135deg,#1a1035,#2d1b69);
  --grad2:linear-gradient(135deg,#6c63ff,#a78bfa);
  --sh:0 2px 12px rgba(108,99,255,.08);
  --sh2:0 8px 32px rgba(108,99,255,.15);
  --sh3:0 20px 56px rgba(108,99,255,.22);}

body,.stApp{font-family:'DM Sans',sans-serif!important;background:var(--bg)!important;}

/* ── BUTTONS ── */
.stButton>button{
  font-family:'DM Sans',sans-serif!important;font-weight:700!important;
  border-radius:10px!important;border:none!important;
  transition:all .2s cubic-bezier(.4,0,.2,1)!important;}
.stButton>button[kind="primary"]{
  background:var(--grad2)!important;color:#fff!important;
  box-shadow:0 4px 16px rgba(108,99,255,.3)!important;}
.stButton>button[kind="primary"]:hover{
  transform:translateY(-2px)!important;box-shadow:var(--sh3)!important;}
.stButton>button[kind="secondary"]{
  background:var(--w)!important;color:var(--mu)!important;
  border:1.5px solid var(--bd)!important;}
.stButton>button[kind="secondary"]:hover{
  color:var(--p)!important;border-color:var(--p)!important;
  background:var(--plighter)!important;transform:translateY(-1px)!important;}
.stDownloadButton>button{
  background:var(--grad2)!important;color:#fff!important;
  font-family:'DM Sans',sans-serif!important;font-weight:700!important;
  border-radius:10px!important;border:none!important;
  box-shadow:0 4px 16px rgba(108,99,255,.28)!important;
  transition:all .2s!important;}
.stDownloadButton>button:hover{transform:translateY(-2px)!important;box-shadow:var(--sh3)!important;}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"]{
  background:white;border:2px dashed var(--bd);border-radius:16px;
  transition:border-color .2s,background .2s;}
[data-testid="stFileUploader"]:hover{border-color:var(--p);background:var(--plighter);}

/* ── INPUTS ── */
.stTextInput input{
  border-radius:10px!important;border:1.5px solid var(--bd)!important;
  padding:.625rem .875rem!important;font-family:'DM Sans',sans-serif!important;
  transition:border-color .18s,box-shadow .18s!important;}
.stTextInput input:focus{
  border-color:var(--p)!important;
  box-shadow:0 0 0 3px rgba(108,99,255,.12)!important;}

/* ── RADIO (answer cards) ── */
div[data-testid="stRadio"]>div{gap:.5rem!important;flex-direction:column!important;display:flex!important;}
div[data-testid="stRadio"]>div>label{
  background:white!important;padding:1rem 1.375rem!important;border-radius:12px!important;
  border:2px solid var(--bd)!important;font-size:.9375rem!important;font-weight:600!important;
  color:var(--tx)!important;width:100%!important;margin:0!important;
  transition:all .2s!important;cursor:pointer!important;}
div[data-testid="stRadio"]>div>label:hover{
  border-color:var(--p)!important;background:var(--plighter)!important;transform:translateX(4px)!important;}
div[data-testid="stRadio"]>div>label:has(input:checked){
  border-color:var(--p)!important;background:linear-gradient(135deg,#f5f3ff,#ede9fe)!important;
  color:#4c1d95!important;box-shadow:0 0 0 3px rgba(108,99,255,.12)!important;transform:translateX(4px)!important;}
div[data-testid="stRadio"]>div>label>div:first-child{display:none!important;}

/* ── FORM ── */
[data-testid="stForm"]{
  border-radius:0 0 16px 16px!important;border:1px solid #ede9fe!important;border-top:none!important;
  box-shadow:var(--sh2)!important;padding:1.375rem 1.5rem 1.5rem!important;background:white!important;}

/* ── AUTH ── */
.auth-label{font-size:.78rem;font-weight:700;color:var(--tx);margin-bottom:.3rem;margin-top:.875rem;}
.auth-divider{text-align:center;font-size:.72rem;color:var(--fa);margin:.75rem 0;position:relative;}
.auth-divider::before,.auth-divider::after{content:'';position:absolute;top:50%;width:38%;height:1px;background:var(--bd);}
.auth-divider::before{left:0;}.auth-divider::after{right:0;}

/* ── PAGE WRAPPERS ── */
.page-wrap{max-width:860px;margin:0 auto;padding:2.5rem 1.5rem 4rem;}
.page-bc{font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:var(--fa);margin-bottom:.4rem;}
.page-h1{font-family:'Sora',sans-serif;font-size:2.25rem;font-weight:800;color:var(--tx);letter-spacing:-.03em;margin-bottom:.35rem;}
.page-sub{font-size:.9rem;color:var(--mu);margin-bottom:2.5rem;line-height:1.75;}

/* ── CARDS ── */
.glass-card{
  background:rgba(255,255,255,.85);backdrop-filter:blur(12px);
  border:1px solid rgba(108,99,255,.12);border-radius:18px;
  box-shadow:var(--sh);transition:all .25s;}
.glass-card:hover{box-shadow:var(--sh2);transform:translateY(-2px);}

/* ── NAVBAR (set by JS injection) ── */

/* ── HERO ── */
.hero-section{background:transparent;padding:4rem 2rem 2rem;position:relative;}
.hero-badge{display:inline-flex;align-items:center;gap:.5rem;background:rgba(108,99,255,.08);
  border:1px solid rgba(108,99,255,.18);color:var(--p);font-size:.6rem;font-weight:700;
  padding:.3rem .9rem;border-radius:999px;letter-spacing:.06em;text-transform:uppercase;margin-bottom:1.5rem;}
.hero-h1{font-family:'Sora',sans-serif;font-size:3.75rem;font-weight:900;color:var(--tx);
  letter-spacing:-.04em;line-height:1.05;margin-bottom:1.25rem;}
.hero-h1 .acc{color:var(--p);}
.hero-p{font-size:.95rem;color:var(--mu);line-height:1.85;margin-bottom:1.75rem;max-width:460px;}

/* ── STATS BAR ── */
.stats-bar{background:var(--grad);padding:.1rem 0;}
.stats-inner{max-width:1200px;margin:0 auto;display:grid;grid-template-columns:repeat(4,1fr);}
.sc{text-align:center;padding:2.25rem 1.5rem;border-right:1px solid rgba(255,255,255,.08);}
.sc:last-child{border-right:none;}
.sc-n{font-family:'Sora',sans-serif;font-size:2.25rem;font-weight:900;color:white;display:block;letter-spacing:-.04em;}
.sc-l{font-size:.58rem;color:rgba(255,255,255,.45);font-weight:600;text-transform:uppercase;letter-spacing:.12em;margin-top:.4rem;display:block;}

/* ── HOW IT WORKS ── */
.section{padding:5rem 2rem;}.section-inner{max-width:1200px;margin:0 auto;}
.sec-eyebrow{font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.14em;color:var(--p);margin-bottom:.5rem;}
.sec-title{font-family:'Sora',sans-serif;font-size:2.1rem;font-weight:800;color:var(--tx);letter-spacing:-.04em;margin-bottom:.75rem;}
.sec-sub{font-size:.9rem;color:var(--mu);max-width:460px;line-height:1.8;}
.hw-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.5rem;margin-top:3rem;}
.hw-card{background:var(--w);border:1px solid var(--bd);border-radius:20px;padding:2rem;
  transition:all .3s;position:relative;overflow:hidden;}
.hw-card:hover{transform:translateY(-5px);box-shadow:var(--sh2);}
.hw-card::before{content:attr(data-n);position:absolute;top:-12px;right:1rem;font-family:'Sora',sans-serif;
  font-size:5rem;font-weight:900;color:rgba(108,99,255,.04);line-height:1;pointer-events:none;}
.hw-ico{width:52px;height:52px;border-radius:14px;display:flex;align-items:center;justify-content:center;
  font-size:1.4rem;margin-bottom:1rem;box-shadow:var(--sh);}
.hw-t{font-family:'Sora',sans-serif;font-size:1rem;font-weight:700;color:var(--tx);margin-bottom:.5rem;}
.hw-p{font-size:.875rem;color:var(--mu);line-height:1.75;margin:0;}

/* ── FEATURE CARDS ── */
.feat-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:1.125rem;}
.feat-card{background:white;border:1px solid var(--bd);border-radius:15px;padding:1.375rem;
  display:flex;align-items:flex-start;gap:.875rem;transition:all .22s;}
.feat-card:hover{box-shadow:var(--sh2);transform:translateY(-3px);border-color:rgba(108,99,255,.25);}
.feat-ico{width:42px;height:42px;border-radius:11px;display:flex;align-items:center;justify-content:center;
  font-size:1.1rem;flex-shrink:0;}
.feat-t{font-size:.82rem;font-weight:700;color:var(--tx);margin-bottom:.18rem;}
.feat-s{font-size:.72rem;color:var(--mu);line-height:1.65;}

/* ── DIFFICULTY CARDS ── */
.df-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.5rem;margin-top:3rem;}
.df-card{background:white;border:1px solid var(--bd);border-radius:20px;padding:2rem;transition:all .3s;}
.df-card:hover{transform:translateY(-5px);box-shadow:var(--sh2);}
.df-card.e{border-top:4px solid var(--gr);}
.df-card.m{border-top:4px solid var(--p);}
.df-card.h{border-top:4px solid var(--re);}
.df-pill{display:inline-flex;font-size:.58rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;padding:3px 10px;border-radius:999px;margin-bottom:.5rem;}
.df-pill.e{background:#d1fae5;color:#064e3b;}.df-pill.m{background:#ede9fe;color:#4c1d95;}.df-pill.h{background:#fee2e2;color:#991b1b;}
.df-name{font-family:'Sora',sans-serif;font-size:1.4rem;font-weight:800;color:var(--tx);margin-bottom:.5rem;}
.df-desc{font-size:.875rem;color:var(--mu);line-height:1.7;margin-bottom:.875rem;}

/* ── GENERATE PAGE ── */
.pdf-banner{background:white;border:1px solid var(--bd);border-radius:14px;padding:1.125rem 1.5rem;
  display:flex;align-items:center;justify-content:space-between;margin-bottom:1.25rem;
  gap:1rem;box-shadow:var(--sh);}
.pdf-icon{width:46px;height:46px;border-radius:12px;background:linear-gradient(135deg,#fee2e2,#fecaca);
  display:flex;align-items:center;justify-content:center;font-size:1.25rem;flex-shrink:0;}
.pdf-info{flex:1;min-width:0;}
.pdf-name{font-size:.9rem;font-weight:700;color:var(--tx);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pdf-meta{font-size:.72rem;color:var(--mu);margin-top:.2rem;}
.diff-badge{display:inline-flex;align-items:center;gap:4px;font-size:.62rem;font-weight:700;padding:2px 8px;border-radius:6px;margin-left:.5rem;}
.diff-badge.Easy{background:#d1fae5;color:#064e3b;}.diff-badge.Medium{background:#ede9fe;color:#4c1d95;}.diff-badge.Hard{background:#fee2e2;color:#991b1b;}
.stat4{display:grid;grid-template-columns:repeat(4,1fr);gap:.875rem;margin-bottom:1.5rem;}
.stat4-card{background:white;border:1px solid var(--bd);border-radius:13px;padding:1.25rem;transition:all .2s;}
.stat4-card:hover{box-shadow:var(--sh2);transform:translateY(-2px);}
.stat4-label{font-size:.58rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--fa);}
.stat4-val{font-family:'Sora',sans-serif;font-size:1.5rem;font-weight:800;color:var(--tx);margin-top:.25rem;}
.stat4-card.accent{border-left:3px solid var(--p);background:linear-gradient(135deg,white,var(--plighter));}
.stat4-card.accent .stat4-val{color:var(--p);}
.cfg-card{background:white;border:1px solid var(--bd);border-radius:14px;padding:1.5rem;margin-bottom:1rem;box-shadow:var(--sh);}
.cfg-title{font-size:.9rem;font-weight:700;color:var(--tx);}
.cfg-hint{font-size:.78rem;color:var(--mu);margin-top:.2rem;}
.cfg-pill{background:linear-gradient(135deg,var(--plighter),rgba(108,99,255,.08));border:1px solid rgba(108,99,255,.15);
  border-radius:12px;padding:.625rem 1rem;text-align:center;min-width:78px;}
.cfg-num{font-family:'Sora',sans-serif;font-size:1.75rem;font-weight:900;color:var(--p);line-height:1;display:block;}
.cfg-lbl{font-size:.52rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:rgba(108,99,255,.45);}
.qt-card{border:2px solid var(--bd);border-radius:13px;padding:1.125rem;text-align:center;
  background:white;transition:all .22s;cursor:pointer;position:relative;overflow:hidden;}
.qt-card:hover{transform:translateY(-2px);box-shadow:var(--sh2);}
.qt-card.sel{border-color:var(--p);background:var(--plighter);box-shadow:0 0 0 3px rgba(108,99,255,.1);}
.qt-ico{font-size:1.625rem;margin-bottom:.5rem;}
.qt-t{font-size:.82rem;font-weight:700;color:var(--tx);}
.qt-s{font-size:.66rem;color:var(--mu);margin-top:.18rem;}
.prog-wrap{background:white;border:1px solid var(--bd);border-radius:13px;padding:1.375rem 1.5rem;margin-top:1rem;box-shadow:var(--sh);}
.prog-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:.75rem;}
.prog-label{font-size:.8rem;font-weight:700;color:var(--p);}
.prog-pct{font-size:.8rem;font-weight:700;color:var(--fa);}
.prog-bar{width:100%;height:7px;background:#ede9fe;border-radius:999px;overflow:hidden;}
.prog-fill{height:100%;background:var(--grad2);border-radius:999px;transition:width .5s cubic-bezier(.4,0,.2,1);}
.prog-note{text-align:center;font-size:.68rem;color:var(--fa);margin-top:.875rem;font-style:italic;}
.prev-card{background:white;border:2px solid var(--p);border-radius:15px;padding:1.5rem;margin-top:1.25rem;margin-bottom:1.5rem;box-shadow:0 0 0 4px rgba(108,99,255,.06);}
.prev-badge{background:#ede9fe;color:#4c1d95;font-size:.58rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;padding:3px 10px;border-radius:6px;display:inline-block;margin-bottom:.875rem;}
.prev-q{font-size:.95rem;font-weight:700;color:var(--tx);margin-bottom:.875rem;line-height:1.55;}

/* ── STUDY ── */
.study-wrap{max-width:960px;margin:0 auto;padding:0 1.5rem 4rem;}
.study-title{font-family:'Sora',sans-serif;font-size:1.5rem;font-weight:800;color:var(--tx);}
.study-badge{background:linear-gradient(135deg,var(--plighter),rgba(108,99,255,.08));color:var(--p);
  font-size:.58rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;padding:4px 12px;
  border-radius:999px;border:1px solid rgba(108,99,255,.15);}
.mcq-card{background:white;border:1px solid var(--bd);border-radius:16px;overflow:hidden;margin-bottom:1.25rem;box-shadow:var(--sh);transition:all .25s;}
.mcq-card:hover{box-shadow:var(--sh2);transform:translateY(-2px);}
.mcq-head{padding:1.25rem 1.375rem .75rem;display:flex;align-items:center;justify-content:space-between;}
.mcq-q-num{font-size:.58rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:var(--fa);
  background:#f8f7ff;border:1px solid var(--bd);border-radius:6px;padding:3px 10px;}
.mcq-type-badge{font-size:.58rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;padding:3px 10px;border-radius:6px;background:var(--plighter);color:var(--p);}
.mcq-q{padding:.25rem 1.375rem 1.125rem;font-size:.95rem;font-weight:600;color:var(--tx);line-height:1.65;}
.opt-ok{display:flex;align-items:center;padding:.75rem 1rem;margin:.25rem 1.125rem;border-radius:10px;
  border:2px solid var(--gr);background:#ecfdf5;transition:all .18s;}
.opt-no{display:flex;align-items:center;padding:.75rem 1rem;margin:.25rem 1.125rem;border-radius:10px;
  border:1px solid var(--bd);background:#f8f7ff;transition:all .18s;}
.opt-no:hover{border-color:var(--bd);background:#f1f0fc;}
.opt-lt-ok{width:26px;height:26px;border-radius:50%;background:var(--gr);color:white;
  display:flex;align-items:center;justify-content:center;font-size:.68rem;font-weight:700;margin-right:.875rem;flex-shrink:0;}
.opt-lt{width:26px;height:26px;border-radius:50%;background:white;color:var(--mu);
  display:flex;align-items:center;justify-content:center;font-size:.68rem;font-weight:600;margin-right:.875rem;flex-shrink:0;border:1px solid var(--bd);}
.opt-tx-ok{font-size:.875rem;font-weight:600;color:#064e3b;flex:1;}
.opt-tx{font-size:.875rem;color:var(--mu);flex:1;}
.mcq-footer{border-top:1px solid var(--bd);padding:.875rem 1.375rem;display:flex;align-items:center;justify-content:space-between;gap:.875rem;background:#faf9ff;}

/* ── TEST ── */
.tcard-info{background:linear-gradient(135deg,#f5f3ff,#ede9fe);border-left:4px solid var(--p);
  border-radius:13px;padding:1rem 1.375rem;font-size:.875rem;color:#4c1d95;margin-bottom:1.5rem;
  box-shadow:0 0 0 1px rgba(108,99,255,.1);}
.td-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.25rem;margin-bottom:1.5rem;}
.td-card{background:white;border:2px solid var(--bd);border-radius:18px;padding:2rem;
  position:relative;transition:all .25s;text-align:center;}
.td-card:hover{transform:translateY(-4px);box-shadow:var(--sh2);}
.td-card.feat{border-color:var(--p);box-shadow:0 0 0 4px rgba(108,99,255,.08);}
.td-pop{position:absolute;top:.875rem;right:.875rem;background:var(--grad2);color:white;
  font-size:.55rem;font-weight:700;padding:2px 8px;border-radius:999px;text-transform:uppercase;}
.td-ico{width:58px;height:58px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:1.625rem;margin:0 auto .875rem;box-shadow:var(--sh);}
.td-ico.e{background:linear-gradient(135deg,#d1fae5,#a7f3d0);}
.td-ico.m{background:linear-gradient(135deg,#ede9fe,#ddd6fe);}
.td-ico.h{background:linear-gradient(135deg,#fee2e2,#fecaca);}
.td-name{font-family:'Sora',sans-serif;font-size:1.1rem;font-weight:800;color:var(--tx);margin-bottom:.375rem;}
.td-hint{font-size:.78rem;color:var(--mu);line-height:1.65;}
.tq-progress{background:white;border:1px solid var(--bd);border-radius:14px;padding:.875rem 1.375rem;
  margin-bottom:1.5rem;display:flex;align-items:center;gap:1.25rem;box-shadow:var(--sh);}
.tq-pbar{flex:1;height:6px;background:#ede9fe;border-radius:999px;overflow:hidden;}
.tq-pfill{height:100%;background:var(--grad2);border-radius:999px;transition:width .4s;}
.tq-timer{color:var(--p);font-size:.875rem;font-weight:700;display:flex;align-items:center;gap:4px;}
.tq{background:white;border:1px solid var(--bd);border-radius:16px;padding:1.5rem;box-shadow:var(--sh);margin-bottom:.875rem;}
.tq-top{display:flex;align-items:center;gap:.625rem;margin-bottom:.875rem;flex-wrap:wrap;}
.tq-num{background:var(--grad2);color:white;padding:3px 11px;border-radius:7px;font-size:.68rem;font-weight:700;}
.tq-de{background:#ede9fe;color:#4c1d95;padding:3px 10px;border-radius:999px;font-size:.66rem;font-weight:700;}
.tq-dm{background:#d1fae5;color:#064e3b;padding:3px 10px;border-radius:999px;font-size:.66rem;font-weight:700;}
.tq-dh{background:#fee2e2;color:#991b1b;padding:3px 10px;border-radius:999px;font-size:.66rem;font-weight:700;}
.tq-q{font-size:.975rem;font-weight:700;color:var(--tx);line-height:1.6;margin-bottom:.875rem;}
.rv{display:flex;align-items:flex-start;gap:.875rem;padding:.875rem 1.125rem;border-radius:12px;
  background:#f8f7ff;border-left:4px solid transparent;margin-bottom:.5rem;transition:all .18s;}
.rv:hover{transform:translateX(2px);}
.rv-c{border-left-color:var(--gr);background:#f0fdf4;}
.rv-w{border-left-color:var(--re);background:#fef2f2;}
.rv-q{font-size:.875rem;font-weight:700;color:var(--tx);margin-bottom:.25rem;}
.rv-a{font-size:.8rem;color:var(--mu);}
.rv-lbl{font-size:.63rem;font-weight:700;padding:3px 10px;border-radius:999px;white-space:nowrap;flex-shrink:0;}
.rv-lbl.ok{background:#d1fae5;color:#064e3b;}.rv-lbl.ng{background:#fee2e2;color:#991b1b;}

/* ── DASHBOARD ── */
.dash-wrap{max-width:1000px;margin:0 auto;padding:2.5rem 1.5rem 4rem;}
.dash-h{font-family:'Sora',sans-serif;font-size:2rem;font-weight:900;color:var(--tx);letter-spacing:-.04em;}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.875rem;margin-bottom:1.25rem;}
.stat-card{background:white;border:1px solid var(--bd);border-radius:15px;padding:1.5rem;text-align:center;box-shadow:var(--sh);transition:all .22s;}
.stat-card:hover{transform:translateY(-3px);box-shadow:var(--sh2);}
.stat-ico{font-size:1.625rem;margin-bottom:.5rem;display:block;}
.stat-v{font-family:'Sora',sans-serif;font-size:1.75rem;font-weight:900;color:var(--p);letter-spacing:-.04em;line-height:1;}
.stat-l{font-size:.58rem;font-weight:600;color:var(--fa);text-transform:uppercase;letter-spacing:.09em;margin-top:.375rem;}
.sh-row{display:flex;align-items:center;gap:.875rem;background:white;border:1px solid var(--bd);
  border-radius:13px;padding:.875rem 1.375rem;margin-bottom:.5rem;box-shadow:var(--sh);transition:all .2s;}
.sh-row:hover{transform:translateX(3px);box-shadow:var(--sh2);}
.sh-diff{font-size:.58rem;font-weight:700;text-transform:uppercase;padding:3px 10px;border-radius:999px;white-space:nowrap;flex-shrink:0;}
.sh-diff.Easy{background:#d1fae5;color:#064e3b;}.sh-diff.Medium{background:#ede9fe;color:#4c1d95;}.sh-diff.Hard{background:#fee2e2;color:#991b1b;}
.sh-score{font-family:'Sora',sans-serif;font-size:1.15rem;font-weight:900;color:var(--tx);min-width:52px;text-align:center;}
.sh-meta{flex:1;min-width:0;}
.sh-pdf{font-weight:600;color:var(--tx);font-size:.82rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.sh-date{color:var(--fa);font-size:.68rem;margin-top:.1rem;}
.sh-pbar{width:88px;height:6px;background:#ede9fe;border-radius:999px;overflow:hidden;flex-shrink:0;}
.sh-pfill{height:100%;border-radius:999px;transition:width .4s;}
.sh-pct{font-size:.875rem;font-weight:700;min-width:44px;text-align:right;}
.ds-section-title{font-family:'Sora',sans-serif;font-size:1rem;font-weight:800;color:var(--tx);
  margin:2rem 0 1rem;display:flex;align-items:center;gap:.5rem;}
.ds-section-title::after{content:'';flex:1;height:1px;background:var(--bd);}
.chart-wrap{background:white;border:1px solid var(--bd);border-radius:15px;padding:1.5rem;margin-bottom:1.25rem;box-shadow:var(--sh);}

/* ── FOOTER ── */
.site-footer{background:var(--grad);padding:4rem 2rem 1.5rem;}
.ft-inner{max-width:1200px;margin:0 auto;}
.ft-top{display:grid;grid-template-columns:2fr 1fr 1fr 1.5fr;gap:3rem;margin-bottom:3rem;}
.ft-brand{display:flex;align-items:center;gap:.75rem;margin-bottom:1rem;}
.ft-logo{width:36px;height:36px;border-radius:9px;background:rgba(255,255,255,.1);
  display:flex;align-items:center;justify-content:center;font-size:1.125rem;}
.ft-name{font-family:'Sora',sans-serif;font-size:.95rem;font-weight:800;color:white;}
.ft-desc{font-size:.85rem;color:rgba(255,255,255,.38);line-height:1.8;}
.ft-hd{font-size:.6rem;font-weight:700;color:rgba(255,255,255,.5);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.875rem;}
.ft-lk{display:block;font-size:.85rem;color:rgba(255,255,255,.35);margin-bottom:.5rem;cursor:pointer;transition:color .15s;}
.ft-lk:hover{color:rgba(255,255,255,.75);}
.ft-bot{border-top:1px solid rgba(255,255,255,.07);padding-top:1.5rem;display:flex;justify-content:space-between;
  align-items:center;flex-wrap:wrap;gap:1rem;font-size:.66rem;color:rgba(255,255,255,.22);text-transform:uppercase;letter-spacing:.07em;}

/* ── ABOUT ── */
.about-wrap{max-width:1000px;margin:0 auto;padding:3rem 1.5rem 4rem;}

/* ── RESPONSIVE ── */
@media(max-width:768px){
  .hw-grid,.df-grid,.ft-top,.td-grid,.feat-grid,.stat-grid,.stat4{grid-template-columns:1fr;}
  .hero-h1{font-size:2.25rem;}
  .stats-inner{grid-template-columns:repeat(2,1fr);}
}
</style>""", unsafe_allow_html=True)

# ── Navbar CSS injection ──────────────────────────────────────────────────────
components.html("""<script>
(function(){
  var old=window.parent.document.getElementById('qg-nb-css');
  if(old)old.remove();
  var s=window.parent.document.createElement('style');
  s.id='qg-nb-css';
  s.textContent=`
    body,.stApp{background:#f8f7ff!important;}
    section[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"]{display:none!important;}
    [data-testid="stAppViewBlockContainer"]>div:first-child{
      position:sticky!important;top:0!important;z-index:9999!important;
      background:white!important;border-bottom:1px solid #ede9fe!important;
      box-shadow:0 1px 12px rgba(108,99,255,.08)!important;}
    [data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="stHorizontalBlock"]{
      align-items:center!important;padding:0 2.5rem!important;min-height:64px!important;
      gap:0!important;max-width:1440px!important;margin:0 auto!important;}
    [data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="column"]{
      display:flex!important;align-items:center!important;padding-top:0!important;padding-bottom:0!important;}
    .nb-brand{display:flex;align-items:center;gap:10px;white-space:nowrap;}
    .nb-logo{width:36px;height:36px;border-radius:9px;background:linear-gradient(135deg,#6c63ff,#a78bfa);
      display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 4px 12px rgba(108,99,255,.3);}
    .nb-title{font-family:'Sora',sans-serif;font-size:.9rem;font-weight:800;color:#1a1035;letter-spacing:-.02em;}
    .nb-title span{color:#6c63ff;}
    [data-testid="stAppViewBlockContainer"]>div:first-child button[kind="secondary"],
    [data-testid="stAppViewBlockContainer"]>div:first-child button[kind="primary"]{
      font-family:'DM Sans',sans-serif!important;font-size:.875rem!important;font-weight:500!important;
      background:transparent!important;border:none!important;border-radius:0!important;
      box-shadow:none!important;height:64px!important;padding:0 14px!important;
      white-space:nowrap!important;width:auto!important;
      border-bottom:2px solid transparent!important;
      transition:color .15s,border-color .15s!important;color:#6b7280!important;}
    [data-testid="stAppViewBlockContainer"]>div:first-child button[kind="secondary"]:hover{
      color:#6c63ff!important;background:transparent!important;
      border-bottom:2px solid #ede9fe!important;transform:none!important;}
    [data-testid="stAppViewBlockContainer"]>div:first-child button[kind="primary"]{
      color:#6c63ff!important;font-weight:700!important;border-bottom:2px solid #6c63ff!important;}
    [data-testid="stAppViewBlockContainer"]>div:first-child button:disabled{
      opacity:.3!important;background:transparent!important;border-bottom:2px solid transparent!important;}
    .nb-user{display:flex;align-items:center;gap:8px;justify-content:flex-end;width:100%;}
    .nb-sep{width:1px;height:20px;background:#ede9fe;flex-shrink:0;margin:0 4px;}
    .nb-avatar{width:32px;height:32px;border-radius:50%;
      background:linear-gradient(135deg,#ede9fe,#ddd6fe);border:2px solid #c4b5fd;
      display:flex;align-items:center;justify-content:center;
      font-size:.72rem;font-weight:800;color:#5b21b6;font-family:'DM Sans',sans-serif;flex-shrink:0;}
    [data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="column"]:last-child [data-testid="stButton"]{
      position:absolute!important;width:0!important;height:0!important;
      overflow:hidden!important;opacity:0!important;pointer-events:none!important;clip:rect(0,0,0,0)!important;}
  `;
  window.parent.document.head.appendChild(s);
})();
</script>""", height=0, scrolling=False)

# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""<style>
    body,.stApp,[data-testid="stAppViewContainer"],section[data-testid="stMain"],
    [data-testid="stMainBlockContainer"],.block-container{
      background:linear-gradient(135deg,#f5f3ff 0%,#ede9fe 40%,#ddd6fe 100%)!important;
      background-size:400% 400%!important;animation:authGrad 10s ease infinite!important;}
    @keyframes authGrad{0%{background-position:0% 50%;}50%{background-position:100% 50%;}100%{background-position:0% 50%;}}
    </style>""", unsafe_allow_html=True)

    st.markdown("<div style='height:2.5rem;'></div>", unsafe_allow_html=True)
    _, ac, _ = st.columns([1, 1.1, 1])
    with ac:
        st.markdown("""
        <div style="background:rgba(255,255,255,.92);backdrop-filter:blur(20px);border-radius:24px;
          padding:2rem 2rem 1.5rem;box-shadow:0 8px 48px rgba(108,99,255,.18),0 1px 0 rgba(255,255,255,.8);
          border:1px solid rgba(255,255,255,.7);margin-bottom:.75rem;">
          <div style="display:flex;flex-direction:column;align-items:center;">
            <div style="width:68px;height:68px;border-radius:18px;
              background:linear-gradient(135deg,#6c63ff,#a78bfa);
              display:flex;align-items:center;justify-content:center;font-size:2.25rem;
              box-shadow:0 12px 32px rgba(108,99,255,.4);margin-bottom:1.25rem;
              animation:float 4s ease-in-out infinite;">🧠</div>
            <style>@keyframes float{0%,100%{transform:translateY(0);}50%{transform:translateY(-5px);}}</style>
            <div style="font-family:'Sora',sans-serif;font-size:1.625rem;font-weight:900;color:#1a1035;letter-spacing:-.04em;margin-bottom:.375rem;">QuizGenius AI</div>
            <div style="font-size:.85rem;color:#6b7280;text-align:center;line-height:1.7;max-width:280px;">
              Your AI-powered study partner.<br>Sign in to continue.</div>
            <div style="display:flex;gap:.5rem;margin-top:1.25rem;flex-wrap:wrap;justify-content:center;">
              <span style="background:rgba(108,99,255,.1);color:#4c1d95;border:1px solid rgba(108,99,255,.2);font-size:.63rem;font-weight:700;padding:4px 12px;border-radius:999px;">📄 PDF Upload</span>
              <span style="background:rgba(16,185,129,.1);color:#065f46;border:1px solid rgba(16,185,129,.2);font-size:.63rem;font-weight:700;padding:4px 12px;border-radius:999px;">🤖 Groq AI</span>
              <span style="background:rgba(245,158,11,.1);color:#92400e;border:1px solid rgba(245,158,11,.2);font-size:.63rem;font-weight:700;padding:4px 12px;border-radius:999px;">⚡ Fast & Free</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        t1, t2 = st.columns(2)
        with t1:
            if st.button("Sign In", key="auth_tab_login",
                         type="primary" if st.session_state.auth_mode == "login" else "secondary",
                         use_container_width=True):
                st.session_state.auth_mode = "login"; st.rerun()
        with t2:
            if st.button("Create Account", key="auth_tab_signup",
                         type="primary" if st.session_state.auth_mode == "signup" else "secondary",
                         use_container_width=True):
                st.session_state.auth_mode = "signup"; st.rerun()

        if st.session_state.auth_mode == "login":
            with st.form("login_form"):
                st.markdown('<div class="auth-label">Username</div>', unsafe_allow_html=True)
                lu = st.text_input("u", placeholder="Enter your username", label_visibility="collapsed")
                st.markdown('<div class="auth-label">Password</div>', unsafe_allow_html=True)
                lp = st.text_input("p", type="password", placeholder="Enter your password", label_visibility="collapsed")
                if st.form_submit_button("Sign In →", use_container_width=True, type="primary"):
                    if not lu.strip() or not lp.strip(): st.error("Please fill in all fields.")
                    else:
                        ok, msg = do_login(lu.strip(), lp.strip())
                        if ok: st.rerun()
                        else: st.error(msg)
            st.markdown('<div class="auth-divider">or continue without account</div>', unsafe_allow_html=True)
            if st.button("Continue as Guest →", type="secondary", use_container_width=True, key="guest_btn"):
                st.session_state.logged_in = True; st.session_state.current_user = "__guest__"; st.rerun()
        else:
            with st.form("signup_form"):
                st.markdown('<div class="auth-label">Display Name</div>', unsafe_allow_html=True)
                sdn = st.text_input("dn", placeholder="Your name (optional)", label_visibility="collapsed")
                st.markdown('<div class="auth-label">Username</div>', unsafe_allow_html=True)
                su  = st.text_input("su", placeholder="At least 3 characters", label_visibility="collapsed", key="su")
                st.markdown('<div class="auth-label">Password</div>', unsafe_allow_html=True)
                sp  = st.text_input("sp", type="password", placeholder="At least 6 characters", label_visibility="collapsed", key="sp")
                st.markdown('<div class="auth-label">Confirm Password</div>', unsafe_allow_html=True)
                sp2 = st.text_input("sp2", type="password", placeholder="Repeat password", label_visibility="collapsed", key="sp2")
                if st.form_submit_button("Create My Account →", use_container_width=True, type="primary"):
                    if sp != sp2: st.error("Passwords do not match.")
                    else:
                        ok, msg = do_signup(su.strip(), sp.strip(), sdn.strip())
                        if ok: st.rerun()
                        else: st.error(msg)

        st.markdown('<div style="text-align:center;font-size:.68rem;color:#9ca3af;margin-top:1.25rem;font-family:DM Sans,sans-serif;">© 2025 QuizGenius AI · Design by Vishwas Patel</div>', unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════════════════════
cp      = st.session_state.current_page
gen_ok  = st.session_state.has_generated
uname   = st.session_state.current_user or "User"
is_guest = uname == "__guest__"
display_name = "Guest" if is_guest else uname.capitalize()
init    = display_name[0].upper()

_c0,_c1,_c2,_c3,_c4,_c5,_c6 = st.columns([2,0.75,0.85,0.65,0.9,0.65,1.8])
with _c0:
    st.markdown('<div class="nb-brand"><div class="nb-logo">💡</div><div><div class="nb-title">QuizGenius <span>AI</span></div></div></div>', unsafe_allow_html=True)
with _c1:
    if st.button("Home",      key="n_home",  type="primary" if cp=="Home"      else "secondary"): go("Home")
with _c2:
    if st.button("Generate",  key="n_gen",   type="primary" if cp=="Generate"  else "secondary"): go("Generate")
with _c3:
    if st.button("Study",     key="n_study", type="primary" if cp in ("Study","Flashcard","Test") else "secondary", disabled=not gen_ok): go("Study")
with _c4:
    if st.button("Dashboard", key="n_dash",  type="primary" if cp=="Dashboard" else "secondary"): go("Dashboard")
with _c5:
    if st.button("About",     key="n_about", type="primary" if cp=="About"     else "secondary"): go("About")
with _c6:
    st.markdown(f'''<div class="nb-user">
      <div class="nb-sep"></div>
      <div class="nb-avatar">{init}</div>
      <button class="nb-signout-btn" onclick="
        var btns=window.parent.document.querySelectorAll('button');
        for(var i=0;i<btns.length;i++){{if(btns[i].innerText.trim()=='\U0001f6aa'){{btns[i].click();break;}}}}
      " title="Sign Out"
      style="width:30px;height:30px;border-radius:8px;background:white;border:1px solid #ede9fe;
        display:flex;align-items:center;justify-content:center;cursor:pointer;color:#9ca3af;
        font-size:.85rem;transition:all .18s;margin-left:4px;">↪</button>
    </div>''', unsafe_allow_html=True)
    if st.button("🚪", key="n_logout", type="secondary"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

# ─── API KEY CHECK (show banner if missing) ───────────────────────────────────
if not get_groq_key() and cp in ("Generate",):
    st.markdown("""
    <div style="background:linear-gradient(135deg,#fffbeb,#fef3c7);border:1px solid #fde68a;
      border-radius:12px;padding:1rem 1.5rem;margin:1rem 2rem;font-size:.875rem;color:#92400e;
      display:flex;align-items:center;gap:.75rem;">
      ⚠️ <strong>Groq API key not set.</strong> Add it below or set the <code>GROQ_API_KEY</code> environment variable on Render.
    </div>
    """, unsafe_allow_html=True)
    with st.expander("🔑 Enter Groq API Key"):
        key_in = st.text_input("Groq API Key", type="password", placeholder="gsk_...", label_visibility="collapsed")
        if key_in:
            st.session_state.groq_key_input = key_in
            st.success("Key saved for this session!")

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if cp == "Home":
    sh = st.session_state.score_history
    tst  = len(sh)
    avg  = round(sum(s["pct"] for s in sh) / max(tst, 1), 1) if sh else 0
    best = max((s["pct"] for s in sh), default=0)
    tqg  = len(st.session_state.questions)

    mockup = """
    <div style="background:white;border-radius:18px;padding:1.375rem;
      box-shadow:0 8px 40px rgba(108,99,255,.15);border:1px solid #ede9fe;position:relative;z-index:1;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.875rem;">
        <span style="font-size:.55rem;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:.08em;background:#f8f7ff;padding:3px 10px;border-radius:6px;border:1px solid #ede9fe;">Q3 of 10 · Medium</span>
        <span style="font-size:.6rem;font-weight:500;color:#6b7280;">⏱ 38s</span>
      </div>
      <div style="font-size:.8rem;font-weight:700;color:#1a1035;line-height:1.6;margin-bottom:.875rem;">What is the primary function of mitochondria in a cell?</div>
      <div style="border:2px solid #10b981;border-radius:9px;padding:.45rem .875rem;margin-bottom:.35rem;font-size:.7rem;color:#1a1035;font-weight:600;display:flex;align-items:center;gap:.5rem;">
        <span style="width:17px;height:17px;border-radius:50%;background:#10b981;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:.55rem;color:white;font-weight:700;">✓</span>
        A) Produce ATP through cellular respiration
      </div>
      <div style="padding:.45rem .875rem;margin-bottom:.35rem;font-size:.7rem;color:#9ca3af;">B) Synthesise proteins for the nucleus</div>
      <div style="padding:.45rem .875rem;margin-bottom:.35rem;font-size:.7rem;color:#9ca3af;">C) Control cell division and growth</div>
      <div style="padding:.45rem .875rem;font-size:.7rem;color:#9ca3af;">D) Break down waste via autophagy</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.45rem;margin-top:.875rem;margin-bottom:.75rem;">
        <div style="background:#1a1035;color:white;border-radius:8px;padding:.5rem;text-align:center;font-size:.58rem;font-weight:700;">📖 Study</div>
        <div style="background:linear-gradient(135deg,#6c63ff,#a78bfa);color:white;border-radius:8px;padding:.5rem;text-align:center;font-size:.58rem;font-weight:700;">🎴 Flashcards</div>
        <div style="background:linear-gradient(135deg,#10b981,#047857);color:white;border-radius:8px;padding:.5rem;text-align:center;font-size:.58rem;font-weight:700;">🎯 Test</div>
      </div>
      <div style="background:#f5f3ff;border:1px solid #ede9fe;border-radius:9px;padding:.625rem 1rem;display:flex;align-items:center;gap:.625rem;">
        <span style="font-size:1.125rem;">🏆</span>
        <div>
          <div style="font-size:.62rem;font-weight:700;color:#4c1d95;">Score: 9/10 · 90% — Excellent!</div>
          <div style="font-size:.55rem;color:#7c3aed;margin-top:.1rem;">Streak: 5 sessions</div>
        </div>
      </div>
    </div>"""

    st.markdown(f"""
    <div class="hero-section">
      <div style="max-width:1240px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:4rem;align-items:center;padding-bottom:2rem;">
        <div>
          <div class="hero-badge">✨ Next-Gen AI Study Platform</div>
          <h1 class="hero-h1">Turn any PDF into<br><span class="acc">exam-ready</span><br>quizzes instantly</h1>
          <p class="hero-p">QuizGenius AI transforms your study material into adaptive flashcards, MCQs, True/False, and fill-in-the-blank quizzes — powered by Groq's Llama 3.</p>
          <div style="display:flex;flex-wrap:nowrap;gap:.5rem;margin-bottom:2rem;">
            <div style="background:white;border:1px solid #ede9fe;border-radius:10px;padding:.625rem .875rem;font-size:.72rem;font-weight:600;color:#6b7280;display:flex;align-items:center;gap:.4rem;box-shadow:0 1px 3px rgba(108,99,255,.06);">📄 PDF & OCR</div>
            <div style="background:white;border:1px solid #ede9fe;border-radius:10px;padding:.625rem .875rem;font-size:.72rem;font-weight:600;color:#6b7280;display:flex;align-items:center;gap:.4rem;box-shadow:0 1px 3px rgba(108,99,255,.06);">🎴 Flashcards</div>
            <div style="background:white;border:1px solid #ede9fe;border-radius:10px;padding:.625rem .875rem;font-size:.72rem;font-weight:600;color:#6b7280;display:flex;align-items:center;gap:.4rem;box-shadow:0 1px 3px rgba(108,99,255,.06);">⏱ Timed Tests</div>
            <div style="background:white;border:1px solid #ede9fe;border-radius:10px;padding:.625rem .875rem;font-size:.72rem;font-weight:600;color:#6b7280;display:flex;align-items:center;gap:.4rem;box-shadow:0 1px 3px rgba(108,99,255,.06);">📊 Analytics</div>
          </div>
        </div>
        <div>{mockup}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _l, hb1, hb2, _r = st.columns([0.5, 1.3, 1.1, 0.5])
    with hb1:
        if st.button("⚡  Start Generating Now", key="hero_cta", type="primary", use_container_width=True): go("Generate")
    with hb2:
        if st.button("📊  My Dashboard", key="hero_dash", type="secondary", use_container_width=True): go("Dashboard")

    st.markdown(f'<div class="stats-bar"><div class="stats-inner"><div class="sc"><span class="sc-n">{tqg if tqg else "10K+"}</span><span class="sc-l">Questions Generated</span></div><div class="sc"><span class="sc-n">{tst if tst else "500+"}</span><span class="sc-l">Tests Taken</span></div><div class="sc"><span class="sc-n">{f"{avg}%" if sh else "95%"}</span><span class="sc-l">Average Score</span></div><div class="sc"><span class="sc-n">{f"{best}%" if sh else "Top"}</span><span class="sc-l">Best Score</span></div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="section-inner"><div class="sec-eyebrow">HOW IT WORKS</div><div class="sec-title">Three steps to smarter learning</div><p class="sec-sub">Upload any PDF, generate AI questions, then study and test with full progress tracking.</p><div class="hw-grid"><div class="hw-card" data-n="1"><div class="hw-ico" style="background:linear-gradient(135deg,#dbeafe,#bfdbfe);">📄</div><div class="hw-t">1. Upload PDF</div><p class="hw-p">Drag and drop any PDF. OCR handles scanned documents automatically via Tesseract.</p></div><div class="hw-card" data-n="2"><div class="hw-ico" style="background:linear-gradient(135deg,#d1fae5,#a7f3d0);">🧠</div><div class="hw-t">2. AI Generates</div><p class="hw-p">Groq-powered Llama 3 extracts concepts and creates MCQ, True/False, or Fill-in-the-blank questions at lightning speed.</p></div><div class="hw-card" data-n="3"><div class="hw-ico" style="background:linear-gradient(135deg,#ede9fe,#ddd6fe);">🎯</div><div class="hw-t">3. Study & Excel</div><p class="hw-p">Flip flashcards, bookmark tough questions, take timed tests, and track your improvement over sessions.</p></div></div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section" style="background:white;"><div class="section-inner"><div class="sec-eyebrow">FEATURES</div><div class="sec-title">Everything you need to excel</div><p class="sec-sub">12 powerful features built for serious learners.</p><br><div class="feat-grid"><div class="feat-card"><div class="feat-ico" style="background:#ede9fe;">🎴</div><div><div class="feat-t">Flashcard Mode</div><div class="feat-s">3D flip cards with progress tracking and bookmarks.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:#fef3c7;">⭐</div><div><div class="feat-t">Smart Bookmarks</div><div class="feat-s">Star any question and filter sessions to bookmarked items.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:#d1fae5;">🎯</div><div><div class="feat-t">Auto Difficulty Detection</div><div class="feat-s">AI analyses PDF complexity and recommends Easy, Medium, or Hard.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:#fee2e2;">❌</div><div><div class="feat-t">Wrong Answer Tracker</div><div class="feat-s">Test mistakes are saved so you can revisit and improve.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:#ede9fe;">⏱</div><div><div class="feat-t">Per-Question Timer</div><div class="feat-s">Countdown per question to simulate real exam pressure.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:#ccfbf1;">📊</div><div><div class="feat-t">Score History & Chart</div><div class="feat-s">Track every test and visualise improvement over time.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:#dbeafe;">🗂️</div><div><div class="feat-t">Topic Filter</div><div class="feat-s">Extract chapters from your PDF and focus on specific topics.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:#fef3c7;">🔀</div><div><div class="feat-t">3 Question Types</div><div class="feat-s">MCQ, True/False, or Fill-in-the-blank to match your study style.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:#d1fae5;">👁</div><div><div class="feat-t">Preview Before Generate</div><div class="feat-s">Preview one sample question before generating the full set.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:#ede9fe;">📤</div><div><div class="feat-t">Export as HTML</div><div class="feat-s">Download a formatted quiz sheet for offline study or printing.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:#fee2e2;">👤</div><div><div class="feat-t">User Accounts</div><div class="feat-s">Sign in to save score history and bookmarks in session.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:#dbeafe;">⚡</div><div><div class="feat-t">Groq-Powered Speed</div><div class="feat-s">Llama 3 via Groq API — much faster than local Ollama.</div></div></div></div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="section-inner"><div class="sec-eyebrow">DIFFICULTY LEVELS</div><div class="sec-title">Choose your challenge</div><p class="sec-sub">Three adaptive difficulty levels engineered to match your learning stage.</p><div class="df-grid"><div class="df-card e"><div style="font-size:2rem;margin-bottom:.75rem;">🌱</div><span class="df-pill e">Foundational</span><div class="df-name">Easy</div><p class="df-desc">Direct recall, key terminology, and fundamental concepts. Perfect for first-pass learning.</p></div><div class="df-card m"><div style="font-size:2rem;margin-bottom:.75rem;">📈</div><span class="df-pill m">Intermediate</span><div class="df-name">Medium</div><p class="df-desc">Applied comprehension. Questions require connecting concepts and understanding causality.</p></div><div class="df-card h"><div style="font-size:2rem;margin-bottom:.75rem;">🔥</div><span class="df-pill h">Mastery</span><div class="df-name">Hard</div><p class="df-desc">Advanced synthesis and critical analysis. Designed for top-tier exam preparation.</p></div></div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="site-footer"><div class="ft-inner"><div class="ft-top"><div><div class="ft-brand"><div class="ft-logo">🧠</div><span class="ft-name">QuizGenius AI</span></div><p class="ft-desc">The most advanced AI study platform. Transform any PDF into an adaptive learning experience.</p></div><div><div class="ft-hd">Product</div><span class="ft-lk">Quiz Generator</span><span class="ft-lk">Flashcard Mode</span><span class="ft-lk">Adaptive Testing</span></div><div><div class="ft-hd">Resources</div><span class="ft-lk">Documentation</span><span class="ft-lk">Help Center</span><span class="ft-lk">About</span></div><div><div class="ft-hd">Connect</div><p style="font-size:.85rem;color:rgba(255,255,255,.38);line-height:1.8;">patelvishwas702@gmail.com</p></div></div><div class="ft-bot"><span>© 2025 QuizGenius AI. All rights reserved.</span><span>Design by Vishwas Patel</span></div></div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# GENERATE
# ══════════════════════════════════════════════════════════════════════════════
elif cp == "Generate":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="page-bc">Workspace › AI Creation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-h1">Generate Study Material</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Upload your lecture notes or textbook to create AI-powered quizzes instantly.</p>', unsafe_allow_html=True)

    if not st.session_state.pdf_text.strip():
        uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
        if uploaded_file:
            with st.spinner("Reading PDF..."):
                text = get_pdf_text(uploaded_file)
            if text.strip():
                clear_pdf_state()
                st.session_state.pdf_text      = text
                st.session_state.pdf_filename  = uploaded_file.name
                st.session_state.pdf_size      = uploaded_file.size
                st.session_state.pdf_hash      = hashlib.md5(text.encode()).hexdigest()
                st.session_state.detected_difficulty = detect_difficulty(text)
                st.session_state.topics        = extract_topics(text)
                st.rerun()
            else:
                st.error("Could not extract text. Please try another PDF.")
    else:
        pdf_text = st.session_state.pdf_text
        wc = len(pdf_text.split()); max_q = calc_max_q(wc)
        fn = st.session_state.pdf_filename or "Uploaded PDF"
        sz = st.session_state.pdf_size / 1024
        dd = st.session_state.detected_difficulty

        pb1, pb2 = st.columns([5, 1])
        with pb1:
            st.markdown(f'<div class="pdf-banner"><div class="pdf-icon">📄</div><div class="pdf-info"><div class="pdf-name">{S(fn)}</div><div class="pdf-meta">{wc:,} words &nbsp;·&nbsp; {sz:.1f} KB &nbsp;·&nbsp; AI suggests: <span class="diff-badge {dd}">{dd}</span></div></div></div>', unsafe_allow_html=True)
        with pb2:
            if st.button("Change", key="change_pdf", type="secondary", use_container_width=True):
                clear_pdf_state(); st.rerun()

        st.markdown(f'<div class="stat4"><div class="stat4-card"><div class="stat4-label">Words</div><div class="stat4-val">{wc:,}</div></div><div class="stat4-card"><div class="stat4-label">Characters</div><div class="stat4-val">{len(pdf_text):,}</div></div><div class="stat4-card"><div class="stat4-label">File Size</div><div class="stat4-val">{sz:.0f} KB</div></div><div class="stat4-card accent"><div class="stat4-label">Max Questions</div><div class="stat4-val">{max_q}</div></div></div>', unsafe_allow_html=True)

        if st.session_state.topics:
            with st.expander(f"🗂️ Topic Filter — {len(st.session_state.topics)} sections detected"):
                st.caption("Select sections to focus on. Leave empty for the full document.")
                sel = st.multiselect("Topics", st.session_state.topics, default=st.session_state.selected_topics, label_visibility="collapsed")
                st.session_state.selected_topics = sel
                if sel: st.success(f"Focusing on {len(sel)} topic(s).")

        st.markdown('<div class="cfg-card">', unsafe_allow_html=True)
        num_q = st.slider("Questions", 1, max_q, min(10, max_q), label_visibility="collapsed")
        st.markdown(f'<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;margin-top:.5rem;"><div><div class="cfg-title">Question Count</div><div class="cfg-hint">How many questions to generate from this document.</div></div><div class="cfg-pill"><span class="cfg-num">{num_q}</span><div class="cfg-lbl">Questions</div></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cfg-card"><div style="font-size:.9rem;font-weight:700;color:var(--tx);margin-bottom:.875rem;">Question Type</div>', unsafe_allow_html=True)
        cqt = st.session_state.q_type
        qa1, qa2, qa3 = st.columns(3)
        for col, qtype, ico, lbl, sub in [
            (qa1, "MCQ",  "📝", "Multiple Choice", "4 options, one correct"),
            (qa2, "TF",   "✅", "True / False",     "Fact-based statements"),
            (qa3, "FIB",  "✏️", "Fill in Blank",    "Complete the sentence"),
        ]:
            with col:
                st.markdown(f'<div class="qt-card {"sel" if cqt==qtype else ""}"><div class="qt-ico">{ico}</div><div class="qt-t">{lbl}</div><div class="qt-s">{sub}</div></div>', unsafe_allow_html=True)
                if st.button(f"Select {lbl}", key=f"qt_{qtype}", type="primary" if cqt == qtype else "secondary", use_container_width=True):
                    st.session_state.q_type = qtype; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        _, gc, _ = st.columns([1, 2, 1])
        with gc:
            preview_clicked = st.button("👁  Preview Sample Question", key="prev_btn", type="secondary", use_container_width=True)
            gen_clicked     = st.button("⚡  Generate Questions", type="primary", use_container_width=True, key="gen_btn")
            if st.session_state.generation_done:
                if st.button("📖  Open Study Mode →", type="primary", use_container_width=True, key="goto_s2"): go("Study")

        if preview_clicked:
            if not get_groq_key():
                st.error("Please enter your Groq API key first (see the banner above).")
            else:
                with st.spinner("Generating preview via Groq..."):
                    try:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100) if LANGCHAIN_AVAILABLE else None
                        chunks   = splitter.split_text(pdf_text) if splitter else [pdf_text[:1500]]
                        pq       = llm_generate(chunks[0], st.session_state.q_type, dd)
                        st.session_state.preview_question = pq
                        st.session_state.show_preview     = True
                    except Exception as e:
                        st.error(f"Preview failed: {e}")

        if st.session_state.show_preview and st.session_state.preview_question:
            pq   = st.session_state.preview_question
            popts = "".join(
                f'<div style="padding:.5rem .875rem;border-radius:9px;margin:.3rem 0;'
                f'background:{"#ecfdf5" if o[0]==pq["correct"] else "#f8f7ff"};'
                f'border-left:3px solid {"#10b981" if o[0]==pq["correct"] else "transparent"};'
                f'font-size:.875rem;font-weight:{"700" if o[0]==pq["correct"] else "400"};'
                f'color:{"#064e3b" if o[0]==pq["correct"] else "#6b7280"};">{S(o)}</div>'
                for o in pq["options"] if o.strip()
            )
            st.markdown(f'<div class="prev-card"><div class="prev-badge">Preview · {S(pq["type"])} · {S(pq["difficulty"])}</div><div class="prev-q">{S(pq["question"])}</div>{popts}</div>', unsafe_allow_html=True)

        prog_ph = st.empty()

        if gen_clicked:
            if not get_groq_key():
                st.error("Please enter your Groq API key first.")
            else:
                cur_hash = st.session_state.pdf_hash
                for k, v in [
                    ("questions",[]),("test_questions",[]),("has_generated",False),
                    ("has_test_generated",False),("generation_done",False),
                    ("questions_pdf_hash",""),("selected_difficulty",None),
                    ("user_answers",{}),("test_submitted",False),
                    ("vector_store_texts",[]),("bookmarks",[]),("wrong_answers",[])
                ]:
                    st.session_state[k] = v
                st.session_state.quiz_key += 1

                def upd(pct, msg):
                    prog_ph.markdown(f'<div class="prog-wrap"><div class="prog-top"><span class="prog-label">⚡ {msg}</span><span class="prog-pct">{pct}%</span></div><div class="prog-bar"><div class="prog-fill" style="width:{pct}%"></div></div><p class="prog-note">Groq + Llama 3 is working. Usually takes under 30 seconds.</p></div>', unsafe_allow_html=True)

                upd(10, "Splitting document into chunks...")
                splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100) if LANGCHAIN_AVAILABLE else None

                use_text = pdf_text
                if st.session_state.selected_topics:
                    filtered = ""
                    for t in st.session_state.selected_topics:
                        idx2 = pdf_text.lower().find(t.lower())
                        if idx2 >= 0: filtered += pdf_text[idx2:idx2 + 3000] + "\n\n"
                    if filtered.strip(): use_text = filtered

                chunks = splitter.split_text(use_text) if splitter else [use_text[i:i+1200] for i in range(0, len(use_text), 1100)]
                st.session_state.vector_store_texts = chunks

                upd(25, "Building search index...")
                q_type = st.session_state.q_type
                temp_qs = []

                for i in range(num_q):
                    upd(25 + int((i + 1) * 70 / num_q), f"Generating question {i+1} of {num_q}...")
                    relevant = simple_similarity_search(f"key concepts important facts", chunks, k=3)
                    ctx = "\n".join(relevant[:2]) if relevant else chunks[i % len(chunks)]
                    try:
                        temp_qs.append(llm_generate(ctx, q_type, dd))
                    except Exception as e:
                        temp_qs.append({
                            "question": f"[Generation error: {str(e)[:60]}]",
                            "options":  ["A) --","B) --","C) --","D) --"],
                            "correct":  "A", "context": ctx[:200],
                            "type":     q_type, "difficulty": dd
                        })

                upd(100, "Complete! Questions ready.")
                st.session_state.questions           = temp_qs
                st.session_state.questions_pdf_hash  = cur_hash
                st.session_state.has_generated       = True
                st.session_state.generation_done     = True
                st.rerun()

        if st.session_state.generation_done and st.session_state.has_generated:
            qt_lbl = {"MCQ":"Multiple Choice","TF":"True/False","FIB":"Fill-in-the-Blank"}.get(st.session_state.q_type, "")
            st.success(f"✅ Generated {len(st.session_state.questions)} {qt_lbl} questions successfully!")
            _, sc2, _ = st.columns([1, 2, 1])
            with sc2:
                if st.button("📖 Open Study Mode", type="primary", use_container_width=True, key="goto_study"): go("Study")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STUDY / FLASHCARD / TEST
# ══════════════════════════════════════════════════════════════════════════════
elif cp in ("Study", "Flashcard", "Test"):
    if not gen_ok:
        st.markdown('<div style="max-width:520px;margin:5rem auto;text-align:center;padding:2rem;"><div style="font-size:3.5rem;margin-bottom:1rem;">📄</div><div style="font-family:Sora,sans-serif;font-size:1.4rem;font-weight:800;color:var(--tx);margin-bottom:.5rem;">No Questions Yet</div><div style="font-size:.875rem;color:var(--mu);margin-bottom:2rem;">Upload a PDF and generate questions to unlock Study, Flashcards and Test modes.</div></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("⚡ Go Generate", type="primary", use_container_width=True): go("Generate")
        st.stop()

    if (st.session_state.questions_pdf_hash and st.session_state.pdf_hash
            and st.session_state.questions_pdf_hash != st.session_state.pdf_hash):
        st.warning("A new PDF was loaded. Please regenerate questions.")
        if st.button("⚡ Regenerate Now", type="primary"): go("Generate")
        st.stop()

    st.markdown('<div class="study-wrap">', unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns([1, 1, 1])
    with tc1:
        if st.button("📖 Study",      key="tab_s",  type="primary" if cp=="Study"     else "secondary", use_container_width=True): go("Study")
    with tc2:
        if st.button("🎴 Flashcards", key="tab_fc", type="primary" if cp=="Flashcard" else "secondary", use_container_width=True): go("Flashcard")
    with tc3:
        if st.button("🎯 Test",       key="tab_t",  type="primary" if cp=="Test"      else "secondary", use_container_width=True):
            st.session_state.selected_difficulty = None; st.session_state.has_test_generated = False
            st.session_state.user_answers = {}; st.session_state.test_submitted = False; go("Test")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── STUDY ─────────────────────────────────────────────────────────────────
    if cp == "Study":
        qs = st.session_state.questions; bms = st.session_state.bookmarks; fn = st.session_state.pdf_filename or "PDF"
        sh1, sh2, sh3 = st.columns([2.5, 1, 1])
        with sh1:
            st.markdown(f'<div style="display:flex;align-items:center;gap:.875rem;margin-bottom:1.5rem;"><span class="study-title">Study Questions</span><span class="study-badge">{len(qs)} Qs · {st.session_state.q_type}</span></div>', unsafe_allow_html=True)
        with sh2:
            filter_mode = st.selectbox("Filter", ["All","Bookmarked","Wrong Answers"], label_visibility="collapsed", key="study_filter")
        with sh3:
            html_data = export_html(qs, f"QuizGenius — {fn}")
            st.download_button("📤 Export", html_data.encode(), f"quiz_{fn.replace('.pdf','')}.html", "text/html", use_container_width=True)

        all_text = "\n\n".join(f"Q{i+1}: {q['question']}\n" + "\n".join(q['options']) + f"\nAnswer: {q['correct']}" for i, q in enumerate(qs))
        with st.expander("📋 Copy all questions as text"):
            st.code(all_text, language=None)

        if filter_mode == "Bookmarked":
            display_qs = [(i,q) for i,q in enumerate(qs) if i in bms]
            if not display_qs: st.info("No bookmarks yet. Click Bookmark on any question.")
        elif filter_mode == "Wrong Answers":
            wa_txt = {q["question"] for q in st.session_state.wrong_answers}
            display_qs = [(i,q) for i,q in enumerate(qs) if q["question"] in wa_txt]
            if not display_qs: st.info("No wrong answers tracked yet. Take a Test first!")
        else:
            display_qs = list(enumerate(qs))

        for idx, q in display_qs:
            is_bm = idx in bms
            st.markdown(f'<div class="mcq-card"><div class="mcq-head"><span class="mcq-q-num">Q{idx+1}</span><span class="mcq-type-badge">{S(q.get("type","MCQ"))}</span></div><div class="mcq-q">{S(q["question"])}</div>', unsafe_allow_html=True)
            seen_lts = set()
            for opt in q["options"]:
                raw = opt.strip()
                if not raw: continue
                letter = raw[0]
                if letter not in "ABCD" or letter in seen_lts: continue
                seen_lts.add(letter); txt = clean_opt(raw)
                if letter == q["correct"]:
                    st.markdown(f'<div class="opt-ok"><div class="opt-lt-ok">{S(letter)}</div><div class="opt-tx-ok">{S(txt)}</div><span style="margin-left:auto;color:var(--gr);font-size:1rem;">✓</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="opt-no"><div class="opt-lt">{S(letter)}</div><div class="opt-tx">{S(txt)}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="mcq-footer">', unsafe_allow_html=True)
            bm_c, ctx_c = st.columns([1, 4])
            with bm_c:
                if st.button("⭐ Saved" if is_bm else "☆ Bookmark", key=f"bm_{idx}_{st.session_state.quiz_key}",
                             type="primary" if is_bm else "secondary", use_container_width=True):
                    if idx in st.session_state.bookmarks: st.session_state.bookmarks.remove(idx)
                    else: st.session_state.bookmarks.append(idx)
                    persist_user_data(); st.rerun()
            with ctx_c:
                with st.expander(f"Source context — Q{idx+1}"):
                    st.markdown(f'<div style="background:#f8f7ff;border-left:4px solid var(--p);padding:.875rem 1.125rem;border-radius:8px;font-size:.875rem;color:var(--mu);line-height:1.7;">{S(q["context"])}</div>', unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

    # ── FLASHCARD ──────────────────────────────────────────────────────────────
    elif cp == "Flashcard":
        fc_qs = get_fc_qs(); total_fc = len(fc_qs)
        st.markdown('<div style="max-width:680px;margin:0 auto;">', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;margin-bottom:1.5rem;"><div style="font-family:Sora,sans-serif;font-size:1.875rem;font-weight:800;color:var(--tx);margin-bottom:.375rem;">Flashcards</div><div style="font-size:.875rem;color:var(--mu);">Click the card to reveal the answer.</div></div>', unsafe_allow_html=True)

        f1, f2, f3 = st.columns(3)
        for fi, flt in zip([f1, f2, f3], ["All", "Bookmarked", "Mistakes"]):
            with fi:
                if st.button(flt, key=f"fcf_{flt}", type="primary" if st.session_state.fc_filter == flt else "secondary", use_container_width=True):
                    st.session_state.fc_filter = flt; st.session_state.fc_idx = 0; st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if not fc_qs:
            st.info("No cards match this filter." if st.session_state.fc_filter != "All" else "No questions yet.")
        else:
            idx = min(st.session_state.fc_idx, total_fc - 1)
            orig_idx, q = fc_qs[idx]
            is_bm2  = orig_idx in st.session_state.bookmarks
            answer_txt = clean_opt(next((o for o in q["options"] if o.startswith(q["correct"])), q["correct"]))
            render_flashcard(q["question"], answer_txt, idx + 1, total_fc, is_bm2)

            st.markdown("<br>", unsafe_allow_html=True)
            n1, n2, n3, n4 = st.columns(4)
            with n1:
                if st.button("← Prev",   key="fc_prev",   type="secondary", use_container_width=True):
                    st.session_state.fc_idx = (idx - 1) % total_fc; st.rerun()
            with n2:
                if st.button("🔀 Random", key="fc_random", type="secondary", use_container_width=True):
                    import random; st.session_state.fc_idx = random.randint(0, total_fc - 1); st.rerun()
            with n3:
                if st.button("Next →",   key="fc_next",   type="secondary", use_container_width=True):
                    st.session_state.fc_idx = (idx + 1) % total_fc; st.rerun()
            with n4:
                if st.button("⭐ Saved" if is_bm2 else "☆ Save", key=f"fc_bm_{idx}",
                             type="primary" if is_bm2 else "secondary", use_container_width=True):
                    if orig_idx in st.session_state.bookmarks: st.session_state.bookmarks.remove(orig_idx)
                    else: st.session_state.bookmarks.append(orig_idx)
                    persist_user_data(); st.rerun()
            st.markdown(f'<div style="text-align:center;margin-top:.875rem;font-size:.85rem;font-weight:600;color:var(--fa);">{idx+1} / {total_fc} · {st.session_state.fc_filter}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── TEST ───────────────────────────────────────────────────────────────────
    elif cp == "Test":
        if not st.session_state.selected_difficulty:
            st.markdown('<div class="tcard-info"><strong>📝 Note:</strong> Test questions are generated fresh via Groq for each difficulty level.</div>', unsafe_allow_html=True)
            tm1, tm2 = st.columns([2, 2])
            with tm1:
                timed = st.checkbox("⏱ Enable per-question countdown timer", value=st.session_state.timed_mode)
                st.session_state.timed_mode = timed
            if timed:
                with tm2:
                    secs = st.slider("Seconds per question", 15, 120, st.session_state.per_q_time, 5, label_visibility="collapsed")
                    st.session_state.per_q_time = secs; st.caption(f"⏱ {secs} seconds per question")

            st.markdown('<br><div style="text-align:center;"><div style="font-family:Sora,sans-serif;font-size:1.5rem;font-weight:800;color:var(--tx);margin-bottom:.375rem;">Select Your Challenge Level</div><div style="font-size:.875rem;color:var(--mu);margin-bottom:1.75rem;">Each level generates fresh questions tailored to difficulty.</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="td-grid"><div class="td-card"><div class="td-ico e">🌱</div><div class="td-name">Beginner</div><div class="td-hint">Foundational concepts. 5 questions.</div></div><div class="td-card feat"><span class="td-pop">POPULAR</span><div class="td-ico m">🎓</div><div class="td-name">Standard</div><div class="td-hint">Applied scenarios. 7 questions.</div></div><div class="td-card"><div class="td-ico h">🔥</div><div class="td-name">Expert</div><div class="td-hint">Complex synthesis. 10 questions.</div></div></div>', unsafe_allow_html=True)

            d1, d2, d3 = st.columns(3)
            with d1:
                if st.button("🌱 Start Easy",   key="easy", use_container_width=True, type="secondary"):
                    st.session_state.selected_difficulty = "Easy"; st.session_state.test_started_at = int(time.time()*1000); st.rerun()
            with d2:
                if st.button("🎓 Start Medium", key="med",  use_container_width=True, type="primary"):
                    st.session_state.selected_difficulty = "Medium"; st.session_state.test_started_at = int(time.time()*1000); st.rerun()
            with d3:
                if st.button("🔥 Start Hard",   key="hard", use_container_width=True, type="secondary"):
                    st.session_state.selected_difficulty = "Hard"; st.session_state.test_started_at = int(time.time()*1000); st.rerun()

        elif not st.session_state.has_test_generated:
            diff = st.session_state.selected_difficulty; n = {"Easy":5,"Medium":7,"Hard":10}[diff]
            if not get_groq_key():
                st.error("Groq API key required. Please set it first.")
                if st.button("← Back"): st.session_state.selected_difficulty = None; st.rerun()
            else:
                st.markdown(f'<div style="background:white;border:1px solid var(--bd);border-radius:16px;padding:2.5rem;text-align:center;box-shadow:var(--sh);"><div style="width:60px;height:60px;border-radius:50%;background:linear-gradient(135deg,#ede9fe,#ddd6fe);display:flex;align-items:center;justify-content:center;margin:0 auto 1.25rem;font-size:1.625rem;">🧠</div><div style="font-family:Sora,sans-serif;font-size:1.15rem;font-weight:800;color:var(--tx);margin-bottom:.375rem;">Generating {diff} Test via Groq</div><div style="font-size:.875rem;color:var(--mu);">Creating {n} fresh questions...</div></div>', unsafe_allow_html=True)
                prog = st.progress(0, text=f"Generating {diff} test...")
                st.session_state.test_questions = []; st.session_state.user_answers = {}; st.session_state.test_submitted = False
                chunks = st.session_state.vector_store_texts; q_type = st.session_state.q_type; temp_tqs = []
                for i in range(n):
                    prog.progress(int((i + 1) * 100 / n), text=f"Question {i+1}/{n}...")
                    relevant = simple_similarity_search(f"{diff} level concepts important", chunks, k=3)
                    ctx = "\n".join(relevant[:2]) if relevant else (chunks[i % len(chunks)] if chunks else st.session_state.pdf_text[:1500])
                    try:
                        temp_tqs.append(llm_generate(ctx, q_type, diff))
                    except Exception as e:
                        temp_tqs.append({
                            "question": f"[Error: {str(e)[:60]}]",
                            "options":  ["A) --","B) --","C) --","D) --"],
                            "correct":  "A", "context": "", "type": q_type, "difficulty": diff
                        })
                prog.progress(100, text="Ready!")
                st.session_state.test_questions = temp_tqs; st.session_state.has_test_generated = True
                prog.empty(); st.rerun()

        elif not st.session_state.test_submitted:
            diff = st.session_state.selected_difficulty; total_q = len(st.session_state.test_questions)
            answered = len(st.session_state.user_answers); pct_done = int(answered / total_q * 100) if total_q else 0
            started = st.session_state.test_started_at or int(time.time() * 1000)
            timed = st.session_state.timed_mode; per_q = st.session_state.per_q_time

            st.markdown(f"""<div class="tq-progress">
              <div style="display:flex;flex-direction:column;gap:4px;flex:1;">
                <div style="display:flex;justify-content:space-between;">
                  <span style="font-size:.72rem;font-weight:700;color:var(--p);">{answered}/{total_q} answered</span>
                  <span style="font-size:.72rem;font-weight:700;color:var(--fa);">{pct_done}%</span>
                </div>
                <div class="tq-pbar"><div class="tq-pfill" style="width:{pct_done}%"></div></div>
              </div>
              <div class="tq-timer">⏱ <span id="tqft">00:00</span></div>
            </div>
            <script>
            (function(){{var s={started};setInterval(function(){{
              var el=document.getElementById("tqft");if(el){{
                var t=Math.floor((Date.now()-s)/1000);
                el.innerText=(Math.floor(t/60)<10?"0":"")+Math.floor(t/60)+":"+(t%60<10?"0":"")+t%60;
              }}
            }},500);}})();
            </script>""", unsafe_allow_html=True)

            for idx, q in enumerate(st.session_state.test_questions):
                d  = q.get("difficulty", diff)
                dc = {"Easy":"tq-dm","Medium":"tq-de","Hard":"tq-dh"}.get(d, "tq-de")
                st.markdown(f'<div class="tq"><div class="tq-top"><span class="tq-num">Q{idx+1}</span><span class="{dc}">{S(d)}</span></div><div class="tq-q">{S(q["question"])}</div></div>', unsafe_allow_html=True)
                clean_opts = []; seen_lts = set()
                for opt in q["options"]:
                    raw = opt.strip()
                    if not raw: continue
                    letter = raw[0]
                    if letter not in "ABCD" or letter in seen_lts: continue
                    seen_lts.add(letter); clean_opts.append(f"{letter}) {clean_opt(raw)}")
                ans = st.radio(f"q{idx}", clean_opts, index=None, key=f"tq_{idx}_{st.session_state.quiz_key}", label_visibility="collapsed")
                if ans: st.session_state.user_answers[idx] = ans[0]
                st.markdown("<br>", unsafe_allow_html=True)

            s1, s2, s3 = st.columns([1, 2, 1])
            with s2:
                if answered == total_q:
                    if st.button("✅ Submit Test", use_container_width=True, type="primary"):
                        save_score(diff, sum(1 for i, q in enumerate(st.session_state.test_questions)
                                            if st.session_state.user_answers.get(i) == q["correct"]),
                                   total_q, st.session_state.pdf_filename or "Unknown")
                        st.session_state.test_submitted = True; st.rerun()
                else:
                    st.warning(f"Answer all {total_q} questions before submitting. ({answered}/{total_q} done)")

        else:
            diff  = st.session_state.selected_difficulty
            corr  = sum(1 for i, q in enumerate(st.session_state.test_questions)
                        if st.session_state.user_answers.get(i) == q["correct"])
            total = len(st.session_state.test_questions)
            pct   = corr / total * 100 if total else 0
            wrong_count = total - corr

            render_score_result(pct, corr, total)

            if wrong_count > 0:
                st.markdown(f'<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:11px;padding:.875rem 1.375rem;font-size:.875rem;color:#92400e;margin:.875rem 0;">⚠️ {wrong_count} wrong answer(s) saved — review via Study → Wrong Answers or Flashcards → Mistakes.</div>', unsafe_allow_html=True)

            st.markdown('<div style="background:white;border:1px solid var(--bd);border-radius:18px;overflow:hidden;box-shadow:var(--sh2);margin-top:.875rem;">', unsafe_allow_html=True)
            st.markdown('<div style="padding:1.5rem;"><div style="font-size:.9rem;font-weight:700;color:var(--tx);margin-bottom:1rem;">📋 Detailed Review</div>', unsafe_allow_html=True)
            for idx, q in enumerate(st.session_state.test_questions):
                ua = st.session_state.user_answers.get(idx); ok = ua == q["correct"]
                lc = "#10b981" if ok else "#ef4444"; yt = ua or "--"; ct = q["correct"]
                for opt in q["options"]:
                    if opt and len(opt) > 1:
                        if opt[0] == ua: yt = clean_opt(opt)
                        if opt[0] == q["correct"]: ct = clean_opt(opt)
                wrong = f'<br>Correct: <span style="color:#10b981;font-weight:700;">{S(ct)}</span>' if not ok else ""
                st.markdown(f'<div class="rv {"rv-c" if ok else "rv-w"}"><div style="font-size:1rem;flex-shrink:0;color:{lc};">{"✓" if ok else "✗"}</div><div style="flex:1;"><div class="rv-q">Q{idx+1}: {S(q["question"])}</div><div class="rv-a">Your answer: <span style="color:{lc};font-weight:600;">{S(str(yt))}</span>{wrong}</div></div><span class="rv-lbl {"ok" if ok else "ng"}">{"Correct" if ok else "Wrong"}</span></div>', unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

            rb1, rb2, rb3, rb4 = st.columns(4)
            with rb1:
                if st.button("🔄 Retake",    use_container_width=True, type="secondary"):
                    st.session_state.user_answers = {}; st.session_state.test_submitted = False; st.session_state.quiz_key += 1; st.rerun()
            with rb2:
                if st.button("🎯 New Level", use_container_width=True, type="primary"):
                    st.session_state.selected_difficulty = None; st.session_state.has_test_generated = False
                    st.session_state.user_answers = {}; st.session_state.test_submitted = False; st.rerun()
            with rb3:
                if st.button("🎴 Flashcards", use_container_width=True, type="secondary"): go("Flashcard")
            with rb4:
                if st.button("📊 Dashboard",  use_container_width=True, type="secondary"): go("Dashboard")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif cp == "Dashboard":
    sh = st.session_state.score_history; qs = st.session_state.questions
    bms = st.session_state.bookmarks; was = st.session_state.wrong_answers
    total_tests = len(sh); avg_score = round(sum(s["pct"] for s in sh) / max(total_tests, 1), 1) if sh else 0
    best_score  = max((s["pct"] for s in sh), default=0)
    uname2 = st.session_state.current_user or "User"
    greeting = f"Welcome back, {uname2.capitalize()}!" if uname2 != "__guest__" else "Welcome, Guest!"

    st.markdown('<div class="dash-wrap">', unsafe_allow_html=True)
    st.markdown(f'<div class="page-bc">Your Progress › Dashboard</div><div class="dash-h">Learning Dashboard</div><div style="font-size:.9rem;color:var(--mu);margin-top:.375rem;margin-bottom:2rem;">{S(greeting)} Track your progress and study patterns.</div>', unsafe_allow_html=True)

    st.markdown(f'''<div class="stat-grid">
      <div class="stat-card"><span class="stat-ico">📊</span><div class="stat-v">{total_tests}</div><div class="stat-l">Tests Taken</div></div>
      <div class="stat-card"><span class="stat-ico">🎯</span><div class="stat-v">{avg_score}%</div><div class="stat-l">Avg Score</div></div>
      <div class="stat-card"><span class="stat-ico">🏆</span><div class="stat-v">{best_score}%</div><div class="stat-l">Best Score</div></div>
      <div class="stat-card"><span class="stat-ico">⭐</span><div class="stat-v">{len(bms)}</div><div class="stat-l">Bookmarks</div></div>
    </div>''', unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1: st.markdown(f'<div class="stat-card" style="margin-bottom:1.25rem;"><span class="stat-ico">📝</span><div class="stat-v">{len(qs)}</div><div class="stat-l">Questions Generated</div></div>', unsafe_allow_html=True)
    with d2: st.markdown(f'<div class="stat-card" style="margin-bottom:1.25rem;"><span class="stat-ico">❌</span><div class="stat-v">{len(was)}</div><div class="stat-l">Mistakes Tracked</div></div>', unsafe_allow_html=True)

    if len(sh) >= 2:
        st.markdown('<div class="ds-section-title">📈 Score Trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        render_score_chart(sh)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="ds-section-title">🗂️ Score History</div>', unsafe_allow_html=True)
    if not sh:
        st.markdown('<div style="background:white;border:1px solid var(--bd);border-radius:15px;padding:3rem;text-align:center;box-shadow:var(--sh);"><div style="font-size:2.5rem;margin-bottom:.875rem;">🧪</div><div style="font-family:Sora,sans-serif;font-size:1rem;font-weight:800;color:var(--tx);">No tests taken yet</div><div style="font-size:.875rem;color:var(--mu);margin-top:.375rem;">Complete a test to see your history here.</div></div>', unsafe_allow_html=True)
        _, bc, _ = st.columns([1,1,1])
        with bc:
            if st.button("🎯 Take a Test Now", type="primary", use_container_width=True): go("Test")
    else:
        for entry in reversed(sh):
            pct = entry["pct"]; bc2 = "#10b981" if pct >= 80 else "#6c63ff" if pct >= 60 else "#ef4444"
            st.markdown(f'<div class="sh-row"><span class="sh-diff {entry["diff"]}">{entry["diff"]}</span><div class="sh-score">{entry["score"]}/{entry["total"]}</div><div class="sh-meta"><div class="sh-pdf">{S(entry["pdf"])}</div><div class="sh-date">{entry["date"]}</div></div><div class="sh-pbar"><div class="sh-pfill" style="width:{pct}%;background:{bc2};"></div></div><span class="sh-pct" style="color:{bc2};">{pct}%</span></div>', unsafe_allow_html=True)
        _, cl, _ = st.columns([1,1,1])
        with cl:
            if st.button("🗑 Clear History", type="secondary", use_container_width=True):
                st.session_state.score_history = []; persist_user_data(); st.rerun()

    if was:
        st.markdown(f'<div class="ds-section-title">❌ Wrong Answers to Review ({len(was)})</div>', unsafe_allow_html=True)
        for wq in was[:4]:
            ct = clean_opt(next((o for o in wq["options"] if o.startswith(wq["correct"])), wq["correct"]))
            st.markdown(f'<div class="rv rv-w"><div style="color:var(--re);font-size:1rem;flex-shrink:0;">✗</div><div style="flex:1;"><div class="rv-q">{S(wq["question"])}</div><div class="rv-a">Correct: <span style="color:var(--gr);font-weight:600;">{S(ct)}</span></div></div></div>', unsafe_allow_html=True)
        if len(was) > 4: st.caption(f"+ {len(was)-4} more. See all in Study → Wrong Answers.")
        wa1, wa2 = st.columns(2)
        with wa1:
            if st.button("📖 Study Wrong Answers",  type="secondary", use_container_width=True): go("Study")
        with wa2:
            if st.button("🎴 Flashcard Mistakes", type="secondary", use_container_width=True):
                st.session_state.fc_filter = "Mistakes"; go("Flashcard")

    if bms:
        st.markdown(f'<div class="ds-section-title">⭐ Bookmarks ({len(bms)})</div>', unsafe_allow_html=True)
        for bi in bms[:3]:
            if bi < len(qs):
                st.markdown(f'<div class="rv" style="border-left:4px solid var(--am);background:#fffbeb;"><div style="flex-shrink:0;">⭐</div><div class="rv-q">{S(qs[bi]["question"])}</div></div>', unsafe_allow_html=True)
        if len(bms) > 3: st.caption(f"+ {len(bms)-3} more bookmarks.")
        if st.button("📖 Study Bookmarks", type="secondary"): go("Study")

    if uname2 == "__guest__":
        st.markdown('<div style="background:linear-gradient(135deg,#f5f3ff,#ede9fe);border:1px solid #ddd6fe;border-radius:13px;padding:1.25rem 1.5rem;font-size:.875rem;color:#4c1d95;margin-top:1.5rem;">💡 <strong>Note:</strong> You are in guest mode. Data resets on page refresh. Create an account to keep session data!</div>', unsafe_allow_html=True)
        _, bc3, _ = st.columns([1,1,1])
        with bc3:
            if st.button("Create Account", type="primary", use_container_width=True):
                for k in list(st.session_state.keys()): del st.session_state[k]; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif cp == "About":
    st.markdown('<div class="about-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="page-bc">QuizGenius AI › About</div><div class="page-h1">About QuizGenius AI</div><p class="page-sub">Revolutionising learning through adaptive AI-powered assessments.</p>', unsafe_allow_html=True)
    cs  = "background:white;border:1px solid #ede9fe;border-radius:17px;padding:2rem;margin-bottom:1.5rem;box-shadow:var(--sh);"
    ct2 = "font-family:Sora,sans-serif;font-size:1.05rem;font-weight:800;color:#1a1035;margin-bottom:.875rem;"
    cp2 = "font-size:.875rem;color:#6b7280;line-height:1.85;"
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(f'<div style="{cs}"><div style="{ct2}">🧠 Our Mission</div><p style="{cp2}">QuizGenius AI empowers students and professionals to learn smarter. Groq-powered Llama 3 generates adaptive quizzes with flashcards and real-time progress tracking. Fast, free, and cloud-deployable.</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="{cs}"><div style="{ct2}">🚀 Technology Stack</div><p style="{cp2}"><strong>Llama 3</strong> — via Groq API (cloud)<br><strong>LangChain</strong> — text splitting & retrieval<br><strong>Tesseract OCR</strong> — scanned PDF support<br><strong>Streamlit</strong> — web app framework<br><strong>Render</strong> — cloud deployment</p></div>', unsafe_allow_html=True)
    with a2:
        st.markdown(f'<div style="{cs}"><div style="{ct2}">✨ Key Features</div><p style="{cp2}">🎴 Flashcard mode with 3D flip<br>⭐ Bookmark questions<br>🎯 Auto difficulty detection<br>❌ Wrong answer tracker<br>⏱ Per-question timed mode<br>📊 Score history + chart<br>🗂️ Topic / chapter filter<br>🔀 MCQ / True-False / Fill-in-Blank<br>👁 Preview before generation<br>📤 Export as formatted HTML<br>⚡ Groq-powered speed</p></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="{cs}">
          <div style="{ct2}">📞 Contact & Connect</div>
          <a href="https://www.linkedin.com/in/vishwas-patel-ba91a2288/" target="_blank"
            style="display:flex;align-items:center;gap:.875rem;padding:.75rem 1rem;
            background:#f0f7ff;border:1.5px solid #bfdbfe;border-radius:12px;
            text-decoration:none;margin-bottom:.625rem;">
            <div style="width:38px;height:38px;border-radius:10px;background:#0077b5;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="white"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
            </div>
            <div><div style="font-size:.82rem;font-weight:700;color:#0077b5;">LinkedIn</div><div style="font-size:.7rem;color:#64748b;">Vishwas Patel</div></div>
            <div style="margin-left:auto;color:#94a3b8;font-weight:700;">↗</div>
          </a>
          <a href="mailto:patelvishwas702@gmail.com"
            style="display:flex;align-items:center;gap:.875rem;padding:.75rem 1rem;
            background:#fff5f5;border:1.5px solid #fecaca;border-radius:12px;text-decoration:none;">
            <div style="width:38px;height:38px;border-radius:10px;background:#ea4335;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="white"><path d="M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 0 1 0 19.366V5.457c0-2.023 2.309-3.178 3.927-1.964L5.455 4.64 12 9.548l6.545-4.91 1.528-1.145C21.69 2.28 24 3.434 24 5.457z"/></svg>
            </div>
            <div><div style="font-size:.82rem;font-weight:700;color:#ea4335;">Email</div><div style="font-size:.7rem;color:#64748b;">patelvishwas702@gmail.com</div></div>
            <div style="margin-left:auto;color:#94a3b8;font-weight:700;">↗</div>
          </a>
          <div style="padding-top:.875rem;border-top:1px solid #ede9fe;margin-top:.625rem;">
            <p style="font-size:.78rem;color:#6b7280;margin:0;">Design & Dev by <strong style="color:#1a1035;">Vishwas Patel</strong></p>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
