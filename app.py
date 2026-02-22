import streamlit as st
import streamlit.components.v1 as components
import hashlib, re, html as html_module, time, os, json, datetime
import PyPDF2

st.set_page_config(
    page_title="QuizGenius AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"   # ✅ current production model

# ─── OPTIONAL DEPS ────────────────────────────────────────────────────────────
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
    LC_AVAILABLE = True
except ImportError:
    LC_AVAILABLE = False

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
_D = {
    "logged_in": False, "current_user": None, "auth_mode": "login",
    "users": {}, "user_data": {},
    "questions": [], "test_questions": [], "has_generated": False,
    "has_test_generated": False, "quiz_key": 0,
    "pdf_text": "", "pdf_filename": "", "pdf_size": 0,
    "pdf_hash": "", "questions_pdf_hash": "",
    "user_answers": {}, "test_submitted": False,
    "current_page": "Home", "selected_difficulty": None,
    "chunks": [], "test_started_at": None, "generation_done": False,
    "bookmarks": [], "wrong_answers": [], "score_history": [],
    "detected_difficulty": "Medium", "topics": [], "selected_topics": [],
    "q_type": "MCQ", "fc_idx": 0, "fc_filter": "All",
    "timed_mode": False, "per_q_time": 45,
    "preview_question": None, "show_preview": False,
    "groq_key_input": "",
}
for k, v in _D.items():
    if k not in st.session_state:
        st.session_state[k] = v

S  = lambda t: html_module.escape(str(t))
def go(p): st.session_state.current_page = p; st.rerun()
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()
def get_key(): return GROQ_API_KEY or st.session_state.get("groq_key_input","")

# ─── AUTH ──────────────────────────────────────────────────────────────────────
def do_login(u, pw):
    users = st.session_state.users
    if u not in users: return False, "User not found."
    if users[u]["pw"] != hash_pw(pw): return False, "Incorrect password."
    st.session_state.logged_in = True; st.session_state.current_user = u
    ud = st.session_state.user_data.get(u, {})
    st.session_state.score_history  = ud.get("score_history", [])
    st.session_state.bookmarks      = ud.get("bookmarks", [])
    st.session_state.wrong_answers  = ud.get("wrong_answers", [])
    return True, "ok"

def do_signup(u, pw, name=""):
    if len(u) < 3:  return False, "Username ≥ 3 characters."
    if len(pw) < 6: return False, "Password ≥ 6 characters."
    if u in st.session_state.users: return False, "Username taken."
    st.session_state.users[u] = {"pw": hash_pw(pw), "name": name or u,
                                  "created": datetime.datetime.now().strftime("%Y-%m-%d")}
    return do_login(u, pw)

def persist():
    u = st.session_state.current_user
    if not u or u == "__guest__": return
    st.session_state.user_data[u] = {
        "score_history": st.session_state.score_history,
        "bookmarks":     st.session_state.bookmarks,
        "wrong_answers": st.session_state.wrong_answers,
    }

# ─── PDF ───────────────────────────────────────────────────────────────────────
def get_pdf_text(f):
    try:
        reader = PyPDF2.PdfReader(f)
        text   = "".join(p.extract_text() or "" for p in reader.pages)
        if len(text.strip()) < 100 and OCR_AVAILABLE:
            st.warning("Scanned PDF — running OCR…")
            f.seek(0)
            try:
                imgs = convert_from_bytes(f.read())
                return "".join(pytesseract.image_to_string(i)+"\n" for i in imgs)
            except Exception as e:
                st.error(f"OCR error: {e}"); return text
        return text
    except Exception as e:
        st.error(f"PDF error: {e}"); return ""

def calc_max_q(wc):
    if wc < 500: return 10
    elif wc < 1000: return 25
    elif wc < 2000: return 50
    elif wc < 5000: return 100
    return 150

def detect_diff(text):
    w = text.split()
    if not w: return "Medium"
    avg = sum(len(x) for x in w)/len(w)
    s   = [x for x in re.split(r'[.!?]+',text) if x.strip()]
    cpx = sum(1 for x in w if len(x)>9)/len(w)*100
    sc  = avg*1.5+(len(w)/max(len(s),1))*0.15+cpx*0.4
    return "Easy" if sc<12 else "Hard" if sc>20 else "Medium"

def extract_topics(text):
    topics=[]
    for line in text.split('\n'):
        line=line.strip()
        if not (3<len(line)<90): continue
        if re.match(r'^(chapter|section|unit|topic|part|module)\s+\d+',line,re.I): topics.append(line)
        elif re.match(r'^\d+[\.\)]\s+[A-Z]',line): topics.append(line)
        elif line.isupper() and 1<len(line.split())<=8: topics.append(line)
    return list(dict.fromkeys(topics))[:25]

def clear_pdf():
    for k,v in [("pdf_text",""),("pdf_filename",""),("pdf_size",0),("pdf_hash",""),
                ("questions_pdf_hash",""),("questions",[]),("test_questions",[]),
                ("has_generated",False),("has_test_generated",False),("generation_done",False),
                ("chunks",[]),("selected_difficulty",None),("user_answers",{}),
                ("test_submitted",False),("preview_question",None),("show_preview",False),
                ("topics",[]),("selected_topics",[])]:
        st.session_state[k]=v

# ─── LLM ───────────────────────────────────────────────────────────────────────
def call_groq(prompt):
    k = get_key()
    if not k: raise ValueError("No Groq API key. Add GROQ_API_KEY env var on Render.")
    if not GROQ_AVAILABLE: raise ImportError("groq package missing.")
    client = Groq(api_key=k)
    r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.7, max_tokens=600,
    )
    return r.choices[0].message.content

def keyword_search(query, texts, k=3):
    qw = set(query.lower().split())
    scored = sorted(texts, key=lambda t: len(qw & set(t.lower().split()))/max(len(qw),1), reverse=True)
    return scored[:k]

def build_prompt(ctx, q_type, diff="Medium"):
    lvl = {"Easy":"BASIC recall","Medium":"COMPREHENSION","Hard":"ANALYSIS"}.get(diff,"COMPREHENSION")
    ctx = ctx[:1400]
    if q_type=="TF":
        return f"Create ONE True/False question ({lvl}).\nOUTPUT ONLY:\nQuestion: [statement]\nAnswer: True\n\nContext:\n{ctx}"
    if q_type=="FIB":
        return f"Create ONE fill-in-blank MCQ ({lvl}). Use ___ for blank.\nOUTPUT ONLY:\nQuestion: [sentence with ___]\nA) [correct]\nB) [wrong]\nC) [wrong]\nD) [wrong]\nCorrect Answer: A\n\nContext:\n{ctx}"
    return f"Generate ONE MCQ ({lvl}).\nOUTPUT ONLY:\nQuestion: [question]\nA) [option]\nB) [option]\nC) [option]\nD) [option]\nCorrect Answer: [A/B/C/D]\n\nContext:\n{ctx}"

def clean_opt(opt):
    raw = opt.strip(); txt = raw[2:].strip() if len(raw)>2 else raw
    for s in ("Context:","Correct Answer:","Question:"):
        if s.lower() in txt.lower(): txt=txt[:txt.lower().index(s.lower())].strip()
    return txt.splitlines()[0].strip() if txt else "--"

def parse_mcq(raw):
    q=""; opts={}; c="A"
    for line in raw.splitlines():
        line=line.strip()
        if line.lower().startswith("question:"): q=line[line.index(":")+1:].strip()
        elif re.match(r'^[A-D]\)',line):
            lt=line[0]; txt=line[2:].strip()
            for s in ("Context:","Correct Answer:","Question:"):
                if s.lower() in txt.lower(): txt=txt[:txt.lower().index(s.lower())].strip()
            if txt: opts[lt]=f"{lt}) {txt.splitlines()[0].strip()}"
        elif re.match(r'^Correct Answer\s*:',line,re.I):
            m=re.search(r'[A-D]',line)
            if m: c=m.group(0)
    return q or "Parsing error",[opts.get(l,f"{l}) --") for l in "ABCD"],c

def parse_tf(raw):
    q=""; c="True"
    for line in raw.splitlines():
        line=line.strip()
        if line.lower().startswith("question:"): q=line[line.index(":")+1:].strip()
        elif re.match(r'^answer\s*:',line,re.I): c="True" if "true" in line.lower() else "False"
    return q or "Parsing error",["A) True","B) False"],"A" if c=="True" else "B"

def parse_fib(raw):
    q=""; opts={}; c="A"
    for line in raw.splitlines():
        line=line.strip()
        if line.lower().startswith("question:"): q=line[line.index(":")+1:].strip()
        elif re.match(r'^[A-D]\)',line):
            lt=line[0]; txt=line[2:].strip().splitlines()[0].strip()
            if txt: opts[lt]=f"{lt}) {txt}"
        elif re.match(r'^correct\s*answer\s*:',line,re.I):
            m=re.search(r'[A-D]',line)
            if m: c=m.group(0)
    return q or "Parsing error",[opts.get(l,f"{l}) --") for l in "ABCD"],c

def llm_gen(ctx, q_type, diff="Medium"):
    raw=call_groq(build_prompt(ctx,q_type,diff))
    if q_type=="TF":    q,opts,c=parse_tf(raw)
    elif q_type=="FIB": q,opts,c=parse_fib(raw)
    else:               q,opts,c=parse_mcq(raw)
    return {"question":q,"options":opts,"correct":c,"context":ctx[:300],"type":q_type,"difficulty":diff}

def save_score(diff, correct, total, pdf_name):
    pct=round(correct/max(total,1)*100,1)
    st.session_state.score_history.append({
        "date":datetime.datetime.now().strftime("%b %d %H:%M"),
        "diff":diff,"score":correct,"total":total,"pct":pct,"pdf":pdf_name
    })
    st.session_state.wrong_answers=[
        q for i,q in enumerate(st.session_state.test_questions)
        if st.session_state.user_answers.get(i)!=q["correct"]
    ]
    persist()

def get_fc_qs():
    qs=st.session_state.questions; f=st.session_state.fc_filter
    if f=="Bookmarked": return [(i,q) for i,q in enumerate(qs) if i in st.session_state.bookmarks]
    if f=="Mistakes":
        wa={q["question"] for q in st.session_state.wrong_answers}
        return [(i,q) for i,q in enumerate(qs) if q["question"] in wa]
    return list(enumerate(qs))

def export_html(questions, title):
    rows=""
    for i,q in enumerate(questions):
        opts="".join(
            f'<li style="padding:.5rem 1rem;border-radius:6px;margin:.25rem 0;'
            f'background:{"#0d2b1f" if o[0]==q["correct"] else "#1a1a2e"};'
            f'border-left:3px solid {"#00ff87" if o[0]==q["correct"] else "transparent"};'
            f'color:{"#00ff87" if o[0]==q["correct"] else "#8892a4"};font-size:.875rem;">'
            f'{S(o)}</li>'
            for o in q["options"] if o.strip()
        )
        rows+=(f'<div style="margin-bottom:1.5rem;padding:1.5rem;border:1px solid #2a2a3e;border-radius:12px;background:#12121f;">'
               f'<div style="font-size:.6rem;font-weight:700;text-transform:uppercase;color:#5a5a7a;margin-bottom:.5rem;">Q{i+1} · {q.get("type","MCQ")}</div>'
               f'<div style="font-size:1rem;font-weight:700;color:#e8e8ff;margin-bottom:.875rem;">{S(q["question"])}</div>'
               f'<ul style="list-style:none;padding:0;margin:0;">{opts}</ul></div>')
    return (f'<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{S(title)}</title>'
            f'<style>body{{font-family:system-ui;max-width:800px;margin:3rem auto;padding:0 1.5rem;background:#0a0a14;color:#e8e8ff;}}</style>'
            f'</head><body><h1 style="color:#e8e8ff;">{S(title)}</h1>'
            f'<p style="color:#5a5a7a;">{datetime.datetime.now().strftime("%B %d, %Y")} · QuizGenius AI</p>'
            f'{rows}</body></html>')

# ─── RICH COMPONENTS ───────────────────────────────────────────────────────────
def render_flashcard(question, answer, num, total, bookmarked=False):
    pct = int(num/total*100)
    bm  = "#f5c518" if bookmarked else "#3a3a5c"
    html_str = f"""<!DOCTYPE html><html><head>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@500;700;800&display=swap');
    *{{margin:0;padding:0;box-sizing:border-box;}}
    body{{font-family:'Cabinet Grotesk',sans-serif;background:transparent;overflow:hidden;}}
    .bar{{width:100%;height:3px;background:#1e1e3a;border-radius:999px;margin-bottom:14px;}}
    .fill{{height:100%;background:linear-gradient(90deg,#f5c518,#ff6b35);border-radius:999px;width:{pct}%;transition:width .4s;}}
    .meta{{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;}}
    .num{{font-size:.55rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#5a5a7a;background:#12121f;padding:3px 10px;border-radius:5px;border:1px solid #2a2a3e;}}
    .bm{{font-size:1.1rem;color:{bm};}}
    .scene{{perspective:1200px;width:100%;height:220px;cursor:pointer;}}
    .card{{width:100%;height:100%;position:relative;transform-style:preserve-3d;transition:transform .6s cubic-bezier(.175,.885,.32,1.1);}}
    .card.flip{{transform:rotateY(180deg);}}
    .face{{position:absolute;inset:0;backface-visibility:hidden;border-radius:16px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:1.75rem;}}
    .front{{background:#12121f;border:1px solid #2a2a3e;box-shadow:0 8px 32px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.04);}}
    .back{{background:linear-gradient(145deg,#0d1a0d,#0a2015);border:1px solid #1a4025;transform:rotateY(180deg);box-shadow:0 12px 40px rgba(0,255,135,.08);}}
    .qlabel{{font-size:.52rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#5a5a7a;margin-bottom:.75rem;}}
    .qtext{{font-size:.95rem;font-weight:700;color:#e8e8ff;line-height:1.6;text-align:center;max-width:480px;}}
    .badge{{background:rgba(0,255,135,.1);border:1px solid rgba(0,255,135,.2);color:#00ff87;font-size:.55rem;font-weight:700;padding:3px 12px;border-radius:999px;margin-bottom:.875rem;text-transform:uppercase;letter-spacing:.08em;}}
    .atext{{font-size:.95rem;font-weight:700;color:#00ff87;line-height:1.6;text-align:center;max-width:480px;}}
    .hint{{position:absolute;bottom:12px;font-size:.58rem;color:#3a3a5c;font-style:italic;}}
    </style></head><body>
    <div class="bar"><div class="fill"></div></div>
    <div class="meta"><span class="num">Card {num} / {total}</span><span class="bm">{'★' if bookmarked else '☆'}</span></div>
    <div class="scene" onclick="this.querySelector('.card').classList.toggle('flip')">
      <div class="card">
        <div class="face front">
          <div class="qlabel">Question</div>
          <div class="qtext">{S(question)}</div>
          <div class="hint">click to reveal →</div>
        </div>
        <div class="face back">
          <div class="badge">Answer</div>
          <div class="atext">{S(answer)}</div>
        </div>
      </div>
    </div>
    </body></html>"""
    components.html(html_str, height=290, scrolling=False)

def render_score_chart(sh):
    if len(sh)<2: return
    recent=sh[-12:]
    labels=json.dumps([s["date"] for s in recent])
    data=json.dumps([s["pct"] for s in recent])
    colors=json.dumps(["rgba(0,255,135,.9)" if s["pct"]>=80 else "rgba(245,197,24,.9)" if s["pct"]>=60 else "rgba(255,107,53,.9)" for s in recent])
    html_str=f"""<!DOCTYPE html><html><head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
    <style>*{{margin:0;padding:0;box-sizing:border-box;}}body{{background:#12121f;padding:16px;font-family:system-ui;}}</style>
    </head><body>
    <canvas id="c" height="140"></canvas>
    <script>
    new Chart(document.getElementById('c').getContext('2d'),{{
      type:'line',data:{{labels:{labels},datasets:[{{
        label:'Score %',data:{data},borderColor:'#f5c518',
        backgroundColor:'rgba(245,197,24,.06)',borderWidth:2,
        pointBackgroundColor:{colors},pointBorderColor:'#0a0a14',
        pointBorderWidth:2,pointRadius:5,fill:true,tension:0.4
      }}]}},
      options:{{responsive:true,plugins:{{legend:{{display:false}},
        tooltip:{{backgroundColor:'#1e1e3a',padding:10,cornerRadius:8,titleColor:'#e8e8ff',bodyColor:'#8892a4'}}}},
        scales:{{y:{{min:0,max:100,grid:{{color:'rgba(255,255,255,.04)'}},
          ticks:{{callback:v=>v+'%',color:'#5a5a7a',font:{{size:10}}}}}},
          x:{{grid:{{display:false}},ticks:{{color:'#5a5a7a',font:{{size:9}},maxRotation:30}}}}}}}}
    }});
    </script></body></html>"""
    components.html(html_str, height=210, scrolling=False)

def render_result(pct, correct, total):
    verdict="Outstanding 🌟" if pct>=80 else "Good Work 👍" if pct>=60 else "Keep Pushing 📚"
    sub=("You've mastered this material." if pct>=80
         else "Solid — a bit more review and you'll nail it." if pct>=60
         else "Review flashcards to reinforce concepts.")
    ring="#00ff87" if pct>=80 else "#f5c518" if pct>=60 else "#ff6b35"
    confetti="true" if pct>=80 else "false"
    html_str=f"""<!DOCTYPE html><html><head>
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.2/dist/confetti.browser.min.js"></script>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Clash+Display:wght@700;800&family=Cabinet+Grotesk:wght@400;600;700&display=swap');
    *{{margin:0;padding:0;box-sizing:border-box;}}
    body{{font-family:'Cabinet Grotesk',sans-serif;background:linear-gradient(160deg,#0a0a14,#0d1a0d);border-radius:20px;overflow:hidden;}}
    .hero{{padding:2.5rem 2rem 2rem;text-align:center;position:relative;}}
    .grid{{position:absolute;inset:0;background-image:linear-gradient(rgba(245,197,24,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(245,197,24,.03) 1px,transparent 1px);background-size:32px 32px;pointer-events:none;}}
    .ring{{width:96px;height:96px;border-radius:50%;border:2px solid {ring};background:rgba(255,255,255,.02);
      display:flex;align-items:center;justify-content:center;margin:0 auto 1.25rem;
      box-shadow:0 0 32px {ring}33,inset 0 0 32px {ring}11;}}
    .pct{{font-family:'Clash Display',sans-serif;font-size:1.875rem;font-weight:800;color:{ring};line-height:1;}}
    .verdict{{font-family:'Clash Display',sans-serif;font-size:1.5rem;font-weight:700;color:#e8e8ff;margin-bottom:.375rem;}}
    .sub{{color:#5a5a7a;font-size:.85rem;line-height:1.7;max-width:340px;margin:0 auto 1.25rem;}}
    .chips{{display:flex;gap:.625rem;justify-content:center;flex-wrap:wrap;padding-bottom:2rem;}}
    .chip{{padding:.35rem .875rem;border-radius:999px;font-size:.72rem;font-weight:700;}}
    .ok{{background:rgba(0,255,135,.1);color:#00ff87;border:1px solid rgba(0,255,135,.2);}}
    .ng{{background:rgba(255,107,53,.1);color:#ff6b35;border:1px solid rgba(255,107,53,.2);}}
    .info{{background:rgba(255,255,255,.05);color:#8892a4;border:1px solid rgba(255,255,255,.08);}}
    </style></head><body>
    <div class="hero">
      <div class="grid"></div>
      <div class="ring"><div class="pct"><span id="ctr">0</span>%</div></div>
      <div class="verdict">{S(verdict)}</div>
      <p class="sub">{S(sub)}</p>
      <div class="chips">
        <span class="chip ok">✓ {correct} Correct</span>
        <span class="chip ng">✗ {total-correct} Wrong</span>
        <span class="chip info">↗ {total} Total</span>
      </div>
    </div>
    <script>
    var t={pct:.0f},e=document.getElementById('ctr'),c=0,s=Math.max(1,Math.ceil(t/60));
    setInterval(function(){{c=Math.min(c+s,t);e.textContent=Math.round(c);}},16);
    if({confetti}){{setTimeout(function(){{
      confetti({{particleCount:100,spread:80,origin:{{y:0.4}},colors:['#f5c518','#00ff87','#ff6b35','#e8e8ff']}});
      setTimeout(function(){{confetti({{particleCount:50,angle:60,spread:55,origin:{{x:0,y:0.5}}}});}},300);
      setTimeout(function(){{confetti({{particleCount:50,angle:120,spread:55,origin:{{x:1,y:0.5}}}});}},600);
    }},200);}}
    </script></body></html>"""
    components.html(html_str, height=330, scrolling=False)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  — Dark Editorial Theme
# Fonts: Clash Display (headings) + Cabinet Grotesk (body)
# Colors: deep navy bg · amber/gold accent · electric green success
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Clash+Display:wght@600;700;800&family=Cabinet+Grotesk:wght@400;500;600;700;800&display=swap');

#MainMenu,footer,header,.stDeployButton{visibility:hidden!important;display:none!important;}
[data-testid="collapsedControl"],section[data-testid="stSidebar"]{display:none!important;}
.stApp>header{display:none!important;height:0!important;}
html,body,.stApp,.main,.block-container,[data-testid="stAppViewBlockContainer"],
[data-testid="stAppViewContainer"],[data-testid="stMainBlockContainer"],
section.main>div:first-child,div[class*="block-container"]{padding-top:0!important;margin-top:0!important;}
.block-container{max-width:100%!important;padding-bottom:0!important;}

::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:#2a2a3e;border-radius:999px;}

:root{
  --bg:#0a0a14;
  --s1:#0f0f1e;
  --s2:#12121f;
  --s3:#1a1a2e;
  --bd:#2a2a3e;
  --bd2:#3a3a5c;
  --tx:#e8e8ff;
  --tx2:#8892a4;
  --tx3:#5a5a7a;
  --gold:#f5c518;
  --golds:#ff6b35;
  --green:#00ff87;
  --red:#ff4757;
  --blue:#4a9eff;
  --grad-gold:linear-gradient(135deg,#f5c518,#ff6b35);
  --grad-green:linear-gradient(135deg,#00ff87,#00c96b);
  --grad-dark:linear-gradient(135deg,#0f0f1e,#1a1a2e);
  --sh:0 2px 16px rgba(0,0,0,.4);
  --sh2:0 8px 40px rgba(0,0,0,.5);
  --sh3:0 20px 60px rgba(0,0,0,.6);
  --glow-gold:0 0 24px rgba(245,197,24,.2);
  --glow-green:0 0 24px rgba(0,255,135,.15);
}

body,.stApp{font-family:'Cabinet Grotesk',sans-serif!important;background:var(--bg)!important;color:var(--tx)!important;}

/* BUTTONS */
.stButton>button{
  font-family:'Cabinet Grotesk',sans-serif!important;font-weight:700!important;
  border-radius:8px!important;border:none!important;
  transition:all .2s cubic-bezier(.4,0,.2,1)!important;}
.stButton>button[kind="primary"]{
  background:var(--grad-gold)!important;color:#0a0a14!important;
  box-shadow:0 4px 20px rgba(245,197,24,.3)!important;}
.stButton>button[kind="primary"]:hover{
  transform:translateY(-2px)!important;box-shadow:var(--glow-gold)!important;}
.stButton>button[kind="secondary"]{
  background:var(--s2)!important;color:var(--tx2)!important;
  border:1px solid var(--bd)!important;}
.stButton>button[kind="secondary"]:hover{
  color:var(--gold)!important;border-color:var(--gold)!important;
  background:rgba(245,197,24,.06)!important;transform:translateY(-1px)!important;}
.stDownloadButton>button{
  background:var(--grad-gold)!important;color:#0a0a14!important;
  font-family:'Cabinet Grotesk',sans-serif!important;font-weight:700!important;
  border-radius:8px!important;border:none!important;
  box-shadow:0 4px 16px rgba(245,197,24,.25)!important;}
.stDownloadButton>button:hover{transform:translateY(-2px)!important;box-shadow:var(--glow-gold)!important;}

/* FILE UPLOADER */
[data-testid="stFileUploader"]{
  background:var(--s2);border:1.5px dashed var(--bd);border-radius:14px;
  transition:border-color .2s;}
[data-testid="stFileUploader"]:hover{border-color:var(--gold);}

/* INPUTS */
.stTextInput input{
  border-radius:8px!important;border:1px solid var(--bd)!important;
  background:var(--s2)!important;color:var(--tx)!important;
  padding:.625rem .875rem!important;font-family:'Cabinet Grotesk',sans-serif!important;
  transition:border-color .18s,box-shadow .18s!important;}
.stTextInput input:focus{border-color:var(--gold)!important;box-shadow:0 0 0 3px rgba(245,197,24,.1)!important;}
.stTextInput input::placeholder{color:var(--tx3)!important;}

/* RADIO */
div[data-testid="stRadio"]>div{gap:.4rem!important;flex-direction:column!important;display:flex!important;}
div[data-testid="stRadio"]>div>label{
  background:var(--s2)!important;padding:.875rem 1.25rem!important;border-radius:10px!important;
  border:1px solid var(--bd)!important;font-size:.9rem!important;font-weight:600!important;
  color:var(--tx2)!important;width:100%!important;margin:0!important;
  transition:all .18s!important;cursor:pointer!important;}
div[data-testid="stRadio"]>div>label:hover{
  border-color:var(--gold)!important;color:var(--tx)!important;
  background:rgba(245,197,24,.05)!important;}
div[data-testid="stRadio"]>div>label:has(input:checked){
  border-color:var(--gold)!important;background:rgba(245,197,24,.08)!important;
  color:var(--gold)!important;box-shadow:0 0 0 2px rgba(245,197,24,.1)!important;}
div[data-testid="stRadio"]>div>label>div:first-child{display:none!important;}

/* FORM */
[data-testid="stForm"]{
  border-radius:0 0 12px 12px!important;border:1px solid var(--bd)!important;border-top:none!important;
  box-shadow:var(--sh2)!important;padding:1.25rem 1.5rem 1.5rem!important;background:var(--s2)!important;}

/* SELECT BOX */
.stSelectbox>div>div{
  background:var(--s2)!important;border:1px solid var(--bd)!important;
  border-radius:8px!important;color:var(--tx)!important;}

/* AUTH */
.auth-label{font-size:.75rem;font-weight:700;color:var(--tx2);margin-bottom:.25rem;margin-top:.75rem;letter-spacing:.04em;}
.auth-div{text-align:center;font-size:.7rem;color:var(--tx3);margin:.75rem 0;position:relative;}
.auth-div::before,.auth-div::after{content:'';position:absolute;top:50%;width:38%;height:1px;background:var(--bd);}
.auth-div::before{left:0;}.auth-div::after{right:0;}

/* PAGE WRAPPERS */
.page-wrap{max-width:840px;margin:0 auto;padding:2.5rem 1.5rem 4rem;}
.page-bc{font-size:.58rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--tx3);margin-bottom:.375rem;}
.page-h1{font-family:'Clash Display',sans-serif;font-size:2.25rem;font-weight:700;color:var(--tx);letter-spacing:-.02em;margin-bottom:.35rem;}
.page-sub{font-size:.9rem;color:var(--tx2);margin-bottom:2.5rem;line-height:1.8;}

/* STAT CARDS */
.stat4{display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;margin-bottom:1.5rem;}
.stat4-card{background:var(--s2);border:1px solid var(--bd);border-radius:12px;padding:1.125rem;transition:all .2s;}
.stat4-card:hover{border-color:var(--bd2);transform:translateY(-2px);}
.stat4-label{font-size:.56rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--tx3);}
.stat4-val{font-family:'Clash Display',sans-serif;font-size:1.5rem;font-weight:700;color:var(--tx);margin-top:.2rem;}
.stat4-card.acc{border-left:2px solid var(--gold);}
.stat4-card.acc .stat4-val{color:var(--gold);}

/* CFG CARDS */
.cfg-card{background:var(--s2);border:1px solid var(--bd);border-radius:12px;padding:1.375rem;margin-bottom:.875rem;}
.cfg-title{font-size:.875rem;font-weight:700;color:var(--tx);}
.cfg-hint{font-size:.75rem;color:var(--tx3);margin-top:.15rem;}
.cfg-pill{background:rgba(245,197,24,.06);border:1px solid rgba(245,197,24,.15);border-radius:10px;padding:.5rem .875rem;text-align:center;min-width:72px;}
.cfg-num{font-family:'Clash Display',sans-serif;font-size:1.625rem;font-weight:700;color:var(--gold);line-height:1;display:block;}
.cfg-lbl{font-size:.5rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:rgba(245,197,24,.4);}

/* QUESTION TYPE CARDS */
.qt-card{border:1px solid var(--bd);border-radius:11px;padding:1rem;text-align:center;background:var(--s2);transition:all .2s;cursor:pointer;}
.qt-card:hover{border-color:var(--bd2);transform:translateY(-2px);}
.qt-card.sel{border-color:var(--gold);background:rgba(245,197,24,.06);box-shadow:0 0 0 2px rgba(245,197,24,.1);}
.qt-ico{font-size:1.5rem;margin-bottom:.375rem;}
.qt-t{font-size:.78rem;font-weight:700;color:var(--tx);}
.qt-s{font-size:.62rem;color:var(--tx3);margin-top:.15rem;}

/* PROGRESS */
.prog-wrap{background:var(--s2);border:1px solid var(--bd);border-radius:12px;padding:1.25rem 1.375rem;margin-top:.875rem;}
.prog-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:.625rem;}
.prog-label{font-size:.75rem;font-weight:700;color:var(--gold);}
.prog-pct{font-size:.75rem;font-weight:700;color:var(--tx3);}
.prog-bar{width:100%;height:5px;background:var(--s3);border-radius:999px;overflow:hidden;}
.prog-fill{height:100%;background:var(--grad-gold);border-radius:999px;transition:width .4s cubic-bezier(.4,0,.2,1);}
.prog-note{text-align:center;font-size:.65rem;color:var(--tx3);margin-top:.75rem;font-style:italic;}

/* PDF BANNER */
.pdf-banner{background:var(--s2);border:1px solid var(--bd);border-radius:12px;padding:1rem 1.375rem;
  display:flex;align-items:center;gap:.875rem;margin-bottom:1.25rem;}
.pdf-icon{width:44px;height:44px;border-radius:10px;background:linear-gradient(135deg,#2a0a0a,#4a1010);
  display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;border:1px solid #3a1515;}
.pdf-name{font-size:.875rem;font-weight:700;color:var(--tx);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pdf-meta{font-size:.7rem;color:var(--tx3);margin-top:.15rem;}
.diff-badge{display:inline-flex;font-size:.58rem;font-weight:700;padding:2px 7px;border-radius:5px;margin-left:.375rem;}
.diff-badge.Easy{background:rgba(0,255,135,.1);color:#00ff87;}.diff-badge.Medium{background:rgba(245,197,24,.1);color:#f5c518;}.diff-badge.Hard{background:rgba(255,71,87,.1);color:#ff4757;}

/* PREV CARD */
.prev-card{background:var(--s2);border:1px solid var(--gold);border-radius:12px;padding:1.375rem;margin-top:1rem;margin-bottom:1.25rem;box-shadow:0 0 0 3px rgba(245,197,24,.06);}
.prev-badge{background:rgba(245,197,24,.1);color:var(--gold);font-size:.56rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;padding:2px 9px;border-radius:5px;display:inline-block;margin-bottom:.75rem;}
.prev-q{font-size:.9rem;font-weight:700;color:var(--tx);margin-bottom:.75rem;line-height:1.6;}

/* STUDY PAGE */
.study-wrap{max-width:940px;margin:0 auto;padding:0 1.5rem 4rem;}
.study-title{font-family:'Clash Display',sans-serif;font-size:1.375rem;font-weight:700;color:var(--tx);}
.study-badge{background:rgba(245,197,24,.08);color:var(--gold);font-size:.56rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;padding:3px 10px;border-radius:999px;border:1px solid rgba(245,197,24,.15);}

/* MCQ CARDS */
.mcq-card{background:var(--s2);border:1px solid var(--bd);border-radius:14px;overflow:hidden;margin-bottom:1rem;transition:all .22s;}
.mcq-card:hover{border-color:var(--bd2);box-shadow:var(--sh2);}
.mcq-head{padding:1.125rem 1.25rem .625rem;display:flex;align-items:center;justify-content:space-between;}
.mcq-q-num{font-size:.56rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:var(--tx3);background:var(--s3);border:1px solid var(--bd);border-radius:5px;padding:2px 9px;}
.mcq-type{font-size:.56rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;padding:2px 9px;border-radius:5px;background:rgba(245,197,24,.08);color:var(--gold);}
.mcq-q{padding:.25rem 1.25rem 1rem;font-size:.925rem;font-weight:600;color:var(--tx);line-height:1.65;}
.opt-ok{display:flex;align-items:center;padding:.625rem .875rem;margin:.2rem 1rem;border-radius:8px;border:1px solid rgba(0,255,135,.3);background:rgba(0,255,135,.06);}
.opt-no{display:flex;align-items:center;padding:.625rem .875rem;margin:.2rem 1rem;border-radius:8px;border:1px solid var(--bd);background:transparent;transition:all .15s;}
.opt-no:hover{border-color:var(--bd2);background:rgba(255,255,255,.02);}
.opt-lt-ok{width:24px;height:24px;border-radius:50%;background:var(--green);color:#0a0a14;display:flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:800;margin-right:.75rem;flex-shrink:0;}
.opt-lt{width:24px;height:24px;border-radius:50%;background:var(--s3);color:var(--tx3);display:flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:600;margin-right:.75rem;flex-shrink:0;border:1px solid var(--bd);}
.opt-tx-ok{font-size:.85rem;font-weight:600;color:var(--green);flex:1;}
.opt-tx{font-size:.85rem;color:var(--tx2);flex:1;}
.mcq-footer{border-top:1px solid var(--bd);padding:.75rem 1.25rem;display:flex;align-items:center;gap:.75rem;background:var(--s1);}

/* TEST PAGE */
.tcard-info{background:rgba(74,158,255,.07);border-left:3px solid var(--blue);border-radius:10px;padding:.875rem 1.25rem;font-size:.85rem;color:#7ab8ff;margin-bottom:1.25rem;}
.td-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.125rem;margin-bottom:1.25rem;}
.td-card{background:var(--s2);border:1px solid var(--bd);border-radius:16px;padding:1.75rem;position:relative;transition:all .25s;text-align:center;}
.td-card:hover{transform:translateY(-4px);border-color:var(--bd2);box-shadow:var(--sh2);}
.td-card.feat{border-color:var(--gold);box-shadow:0 0 0 3px rgba(245,197,24,.08);}
.td-pop{position:absolute;top:.75rem;right:.75rem;background:var(--grad-gold);color:#0a0a14;font-size:.52rem;font-weight:800;padding:2px 8px;border-radius:999px;text-transform:uppercase;}
.td-ico{width:54px;height:54px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.5rem;margin:0 auto .75rem;}
.td-ico.e{background:rgba(0,255,135,.1);border:1px solid rgba(0,255,135,.2);}
.td-ico.m{background:rgba(245,197,24,.1);border:1px solid rgba(245,197,24,.2);}
.td-ico.h{background:rgba(255,71,87,.1);border:1px solid rgba(255,71,87,.2);}
.td-name{font-family:'Clash Display',sans-serif;font-size:1.1rem;font-weight:700;color:var(--tx);margin-bottom:.3rem;}
.td-hint{font-size:.75rem;color:var(--tx3);line-height:1.65;}
.tq-progress{background:var(--s2);border:1px solid var(--bd);border-radius:12px;padding:.875rem 1.25rem;margin-bottom:1.25rem;display:flex;align-items:center;gap:1.125rem;}
.tq-pbar{flex:1;height:4px;background:var(--s3);border-radius:999px;overflow:hidden;}
.tq-pfill{height:100%;background:var(--grad-gold);border-radius:999px;transition:width .4s;}
.tq-timer{color:var(--gold);font-size:.85rem;font-weight:700;display:flex;align-items:center;gap:4px;white-space:nowrap;}
.tq{background:var(--s2);border:1px solid var(--bd);border-radius:14px;padding:1.375rem;margin-bottom:.75rem;transition:border-color .15s;}
.tq:hover{border-color:var(--bd2);}
.tq-top{display:flex;align-items:center;gap:.5rem;margin-bottom:.75rem;flex-wrap:wrap;}
.tq-num{background:var(--grad-gold);color:#0a0a14;padding:2px 10px;border-radius:6px;font-size:.65rem;font-weight:800;}
.tq-de{background:rgba(245,197,24,.1);color:var(--gold);padding:2px 9px;border-radius:999px;font-size:.62rem;font-weight:700;}
.tq-dm{background:rgba(0,255,135,.1);color:var(--green);padding:2px 9px;border-radius:999px;font-size:.62rem;font-weight:700;}
.tq-dh{background:rgba(255,71,87,.1);color:var(--red);padding:2px 9px;border-radius:999px;font-size:.62rem;font-weight:700;}
.tq-q{font-size:.925rem;font-weight:700;color:var(--tx);line-height:1.6;margin-bottom:.75rem;}

/* REVIEW */
.rv{display:flex;align-items:flex-start;gap:.75rem;padding:.75rem 1rem;border-radius:10px;background:var(--s2);border-left:3px solid transparent;margin-bottom:.375rem;transition:all .15s;}
.rv:hover{transform:translateX(2px);}
.rv-c{border-left-color:var(--green);background:rgba(0,255,135,.04);}
.rv-w{border-left-color:var(--red);background:rgba(255,71,87,.04);}
.rv-q{font-size:.85rem;font-weight:700;color:var(--tx);margin-bottom:.2rem;}
.rv-a{font-size:.78rem;color:var(--tx3);}
.rv-lbl{font-size:.6rem;font-weight:700;padding:2px 9px;border-radius:999px;white-space:nowrap;flex-shrink:0;}
.rv-lbl.ok{background:rgba(0,255,135,.1);color:var(--green);}.rv-lbl.ng{background:rgba(255,71,87,.1);color:var(--red);}

/* DASHBOARD */
.dash-wrap{max-width:960px;margin:0 auto;padding:2.5rem 1.5rem 4rem;}
.dash-h{font-family:'Clash Display',sans-serif;font-size:1.875rem;font-weight:700;color:var(--tx);letter-spacing:-.02em;}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;margin-bottom:1rem;}
.stat-card{background:var(--s2);border:1px solid var(--bd);border-radius:13px;padding:1.375rem;text-align:center;transition:all .2s;}
.stat-card:hover{transform:translateY(-2px);border-color:var(--bd2);}
.stat-ico{font-size:1.5rem;margin-bottom:.375rem;display:block;}
.stat-v{font-family:'Clash Display',sans-serif;font-size:1.625rem;font-weight:700;color:var(--gold);letter-spacing:-.02em;line-height:1;}
.stat-l{font-size:.56rem;font-weight:600;color:var(--tx3);text-transform:uppercase;letter-spacing:.09em;margin-top:.3rem;}
.sh-row{display:flex;align-items:center;gap:.75rem;background:var(--s2);border:1px solid var(--bd);border-radius:11px;padding:.75rem 1.125rem;margin-bottom:.375rem;transition:all .18s;}
.sh-row:hover{border-color:var(--bd2);transform:translateX(3px);}
.sh-diff{font-size:.56rem;font-weight:700;text-transform:uppercase;padding:2px 9px;border-radius:999px;white-space:nowrap;flex-shrink:0;}
.sh-diff.Easy{background:rgba(0,255,135,.1);color:var(--green);}.sh-diff.Medium{background:rgba(245,197,24,.1);color:var(--gold);}.sh-diff.Hard{background:rgba(255,71,87,.1);color:var(--red);}
.sh-score{font-family:'Clash Display',sans-serif;font-size:1.1rem;font-weight:700;color:var(--tx);min-width:48px;text-align:center;}
.sh-meta{flex:1;min-width:0;}
.sh-pdf{font-weight:600;color:var(--tx);font-size:.8rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.sh-date{color:var(--tx3);font-size:.66rem;margin-top:.1rem;}
.sh-pbar{width:80px;height:4px;background:var(--s3);border-radius:999px;overflow:hidden;flex-shrink:0;}
.sh-pfill{height:100%;border-radius:999px;transition:width .4s;}
.sh-pct{font-size:.85rem;font-weight:700;min-width:42px;text-align:right;}
.ds-title{font-family:'Clash Display',sans-serif;font-size:.95rem;font-weight:700;color:var(--tx);margin:1.75rem 0 .875rem;display:flex;align-items:center;gap:.5rem;}
.ds-title::after{content:'';flex:1;height:1px;background:var(--bd);}
.chart-wrap{background:var(--s2);border:1px solid var(--bd);border-radius:13px;padding:1.25rem;margin-bottom:1rem;}

/* HOME */
.hero-section{padding:4.5rem 2rem 2rem;position:relative;}
.hero-badge{display:inline-flex;align-items:center;gap:.5rem;background:rgba(245,197,24,.07);border:1px solid rgba(245,197,24,.2);color:var(--gold);font-size:.58rem;font-weight:700;padding:.3rem .875rem;border-radius:999px;letter-spacing:.07em;text-transform:uppercase;margin-bottom:1.5rem;}
.hero-h1{font-family:'Clash Display',sans-serif;font-size:3.75rem;font-weight:700;color:var(--tx);letter-spacing:-.03em;line-height:1.02;margin-bottom:1.125rem;}
.hero-h1 .acc{color:var(--gold);}
.hero-p{font-size:.9rem;color:var(--tx2);line-height:1.85;margin-bottom:1.75rem;max-width:440px;}
.pill-row{display:flex;flex-wrap:nowrap;gap:.5rem;margin-bottom:2rem;}
.pill{background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:.5rem .875rem;font-size:.7rem;font-weight:600;color:var(--tx2);display:flex;align-items:center;gap:.35rem;transition:all .18s;}
.pill:hover{border-color:var(--gold);color:var(--gold);}
.stats-bar{background:var(--s1);border-top:1px solid var(--bd);border-bottom:1px solid var(--bd);}
.stats-inner{max-width:1200px;margin:0 auto;display:grid;grid-template-columns:repeat(4,1fr);}
.sc{text-align:center;padding:2rem 1.25rem;border-right:1px solid var(--bd);}
.sc:last-child{border-right:none;}
.sc-n{font-family:'Clash Display',sans-serif;font-size:2.1rem;font-weight:700;color:var(--gold);display:block;letter-spacing:-.03em;}
.sc-l{font-size:.56rem;color:var(--tx3);font-weight:600;text-transform:uppercase;letter-spacing:.12em;margin-top:.3rem;display:block;}
.section{padding:5rem 2rem;}.section-inner{max-width:1200px;margin:0 auto;}
.sec-eyebrow{font-size:.58rem;font-weight:700;text-transform:uppercase;letter-spacing:.14em;color:var(--gold);margin-bottom:.5rem;}
.sec-title{font-family:'Clash Display',sans-serif;font-size:2.1rem;font-weight:700;color:var(--tx);letter-spacing:-.03em;margin-bottom:.625rem;}
.sec-sub{font-size:.875rem;color:var(--tx2);max-width:420px;line-height:1.85;}
.hw-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.25rem;margin-top:2.5rem;}
.hw-card{background:var(--s2);border:1px solid var(--bd);border-radius:18px;padding:1.875rem;transition:all .28s;position:relative;overflow:hidden;}
.hw-card:hover{transform:translateY(-5px);border-color:var(--gold);box-shadow:0 0 0 1px rgba(245,197,24,.1),var(--sh2);}
.hw-card::before{content:attr(data-n);position:absolute;top:-10px;right:.875rem;font-family:'Clash Display',sans-serif;font-size:4.5rem;font-weight:700;color:rgba(245,197,24,.04);line-height:1;pointer-events:none;}
.hw-ico{width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;margin-bottom:.875rem;border:1px solid var(--bd);}
.hw-t{font-family:'Clash Display',sans-serif;font-size:.95rem;font-weight:700;color:var(--tx);margin-bottom:.375rem;}
.hw-p{font-size:.825rem;color:var(--tx2);line-height:1.75;margin:0;}
.feat-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;}
.feat-card{background:var(--s2);border:1px solid var(--bd);border-radius:13px;padding:1.25rem;display:flex;align-items:flex-start;gap:.75rem;transition:all .2s;}
.feat-card:hover{border-color:var(--bd2);transform:translateY(-2px);}
.feat-ico{width:38px;height:38px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0;border:1px solid var(--bd);}
.feat-t{font-size:.78rem;font-weight:700;color:var(--tx);margin-bottom:.15rem;}
.feat-s{font-size:.68rem;color:var(--tx3);line-height:1.6;}
.df-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1.25rem;margin-top:2.5rem;}
.df-card{background:var(--s2);border:1px solid var(--bd);border-radius:18px;padding:1.875rem;transition:all .28s;}
.df-card:hover{transform:translateY(-5px);box-shadow:var(--sh2);}
.df-card.e{border-top:2px solid var(--green);}.df-card.m{border-top:2px solid var(--gold);}.df-card.h{border-top:2px solid var(--red);}
.df-pill{display:inline-flex;font-size:.56rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;padding:2px 9px;border-radius:999px;margin-bottom:.5rem;}
.df-pill.e{background:rgba(0,255,135,.1);color:var(--green);}.df-pill.m{background:rgba(245,197,24,.1);color:var(--gold);}.df-pill.h{background:rgba(255,71,87,.1);color:var(--red);}
.df-name{font-family:'Clash Display',sans-serif;font-size:1.375rem;font-weight:700;color:var(--tx);margin-bottom:.4rem;}
.df-desc{font-size:.825rem;color:var(--tx2);line-height:1.75;}
.site-footer{background:var(--s1);border-top:1px solid var(--bd);padding:3.5rem 2rem 1.5rem;}
.ft-inner{max-width:1200px;margin:0 auto;}
.ft-top{display:grid;grid-template-columns:2fr 1fr 1fr 1.5fr;gap:3rem;margin-bottom:2.5rem;}
.ft-brand{display:flex;align-items:center;gap:.625rem;margin-bottom:.875rem;}
.ft-logo{width:32px;height:32px;border-radius:8px;background:var(--grad-gold);display:flex;align-items:center;justify-content:center;font-size:1rem;}
.ft-name{font-family:'Clash Display',sans-serif;font-size:.875rem;font-weight:700;color:var(--tx);}
.ft-desc{font-size:.8rem;color:var(--tx3);line-height:1.8;}
.ft-hd{font-size:.56rem;font-weight:700;color:var(--tx3);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.75rem;}
.ft-lk{display:block;font-size:.8rem;color:var(--tx3);margin-bottom:.4rem;cursor:pointer;transition:color .15s;}
.ft-lk:hover{color:var(--gold);}
.ft-bot{border-top:1px solid var(--bd);padding-top:1.25rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.875rem;font-size:.62rem;color:var(--tx3);text-transform:uppercase;letter-spacing:.07em;}
.about-wrap{max-width:960px;margin:0 auto;padding:3rem 1.5rem 4rem;}
@media(max-width:768px){
  .hw-grid,.df-grid,.ft-top,.td-grid,.feat-grid,.stat-grid,.stat4{grid-template-columns:1fr;}
  .hero-h1{font-size:2.25rem;}
  .stats-inner{grid-template-columns:repeat(2,1fr);}
}
</style>""", unsafe_allow_html=True)

# ─── NAVBAR CSS INJECTION ─────────────────────────────────────────────────────
components.html("""<script>
(function(){
  var old=window.parent.document.getElementById('qg-css');
  if(old)old.remove();
  var s=window.parent.document.createElement('style');
  s.id='qg-css';
  s.textContent=`
    @import url('https://fonts.googleapis.com/css2?family=Clash+Display:wght@700&family=Cabinet+Grotesk:wght@500;700&display=swap');
    body,.stApp{background:#0a0a14!important;}
    section[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"]{display:none!important;}
    [data-testid="stAppViewBlockContainer"]>div:first-child{
      position:sticky!important;top:0!important;z-index:9999!important;
      background:#0f0f1e!important;border-bottom:1px solid #2a2a3e!important;
      box-shadow:0 1px 20px rgba(0,0,0,.5)!important;}
    [data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="stHorizontalBlock"]{
      align-items:center!important;padding:0 2.5rem!important;min-height:60px!important;
      gap:0!important;max-width:1440px!important;margin:0 auto!important;}
    [data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="column"]{
      display:flex!important;align-items:center!important;padding-top:0!important;padding-bottom:0!important;}
    .nb-brand{display:flex;align-items:center;gap:9px;white-space:nowrap;}
    .nb-logo{width:32px;height:32px;border-radius:7px;background:linear-gradient(135deg,#f5c518,#ff6b35);
      display:flex;align-items:center;justify-content:center;font-size:16px;box-shadow:0 4px 12px rgba(245,197,24,.3);}
    .nb-title{font-family:'Clash Display',sans-serif;font-size:.875rem;font-weight:700;color:#e8e8ff;letter-spacing:-.01em;}
    .nb-title span{color:#f5c518;}
    [data-testid="stAppViewBlockContainer"]>div:first-child button[kind="secondary"],
    [data-testid="stAppViewBlockContainer"]>div:first-child button[kind="primary"]{
      font-family:'Cabinet Grotesk',sans-serif!important;font-size:.82rem!important;font-weight:600!important;
      background:transparent!important;border:none!important;border-radius:0!important;
      box-shadow:none!important;height:60px!important;padding:0 13px!important;
      white-space:nowrap!important;width:auto!important;
      border-bottom:2px solid transparent!important;
      transition:color .15s,border-color .15s!important;color:#5a5a7a!important;}
    [data-testid="stAppViewBlockContainer"]>div:first-child button[kind="secondary"]:hover{
      color:#e8e8ff!important;background:transparent!important;
      border-bottom:2px solid #2a2a3e!important;transform:none!important;}
    [data-testid="stAppViewBlockContainer"]>div:first-child button[kind="primary"]{
      color:#f5c518!important;font-weight:700!important;border-bottom:2px solid #f5c518!important;}
    [data-testid="stAppViewBlockContainer"]>div:first-child button:disabled{
      opacity:.25!important;background:transparent!important;border-bottom:2px solid transparent!important;}
    .nb-user{display:flex;align-items:center;gap:8px;justify-content:flex-end;width:100%;}
    .nb-sep{width:1px;height:18px;background:#2a2a3e;flex-shrink:0;margin:0 4px;}
    .nb-avatar{width:30px;height:30px;border-radius:50%;
      background:linear-gradient(135deg,#f5c518,#ff6b35);
      display:flex;align-items:center;justify-content:center;
      font-size:.68rem;font-weight:800;color:#0a0a14;font-family:'Clash Display',sans-serif;flex-shrink:0;}
    .nb-out{width:28px;height:28px;border-radius:7px;background:#12121f;border:1px solid #2a2a3e;
      display:flex;align-items:center;justify-content:center;cursor:pointer;color:#5a5a7a;
      font-size:.8rem;transition:all .18s;margin-left:2px;}
    .nb-out:hover{background:rgba(255,71,87,.1);border-color:rgba(255,71,87,.3);color:#ff4757;}
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
      background:#0a0a14!important;}
    </style>""", unsafe_allow_html=True)

    st.markdown("<div style='height:3rem;'></div>", unsafe_allow_html=True)
    _, ac, _ = st.columns([1, 1.05, 1])
    with ac:
        st.markdown("""
        <div style="background:#0f0f1e;border:1px solid #2a2a3e;border-radius:20px;
          padding:2rem 2rem 1.5rem;box-shadow:0 16px 64px rgba(0,0,0,.6);margin-bottom:.75rem;">
          <div style="display:flex;flex-direction:column;align-items:center;">
            <div style="width:64px;height:64px;border-radius:16px;
              background:linear-gradient(135deg,#f5c518,#ff6b35);
              display:flex;align-items:center;justify-content:center;font-size:2rem;
              box-shadow:0 12px 32px rgba(245,197,24,.4);margin-bottom:1.25rem;
              animation:fl 4s ease-in-out infinite;">🧠</div>
            <style>@keyframes fl{0%,100%{transform:translateY(0);}50%{transform:translateY(-5px);}}</style>
            <div style="font-family:'Clash Display',sans-serif;font-size:1.5rem;font-weight:700;color:#e8e8ff;letter-spacing:-.03em;margin-bottom:.375rem;">QuizGenius AI</div>
            <div style="font-size:.82rem;color:#5a5a7a;text-align:center;line-height:1.75;max-width:270px;">
              Your AI-powered study partner.<br>Sign in to continue.</div>
            <div style="display:flex;gap:.5rem;margin-top:1.125rem;flex-wrap:wrap;justify-content:center;">
              <span style="background:rgba(245,197,24,.07);color:#f5c518;border:1px solid rgba(245,197,24,.15);font-size:.6rem;font-weight:700;padding:3px 11px;border-radius:999px;">📄 PDF Upload</span>
              <span style="background:rgba(0,255,135,.07);color:#00ff87;border:1px solid rgba(0,255,135,.15);font-size:.6rem;font-weight:700;padding:3px 11px;border-radius:999px;">⚡ Groq AI</span>
              <span style="background:rgba(74,158,255,.07);color:#4a9eff;border:1px solid rgba(74,158,255,.15);font-size:.6rem;font-weight:700;padding:3px 11px;border-radius:999px;">🎴 Flashcards</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        t1, t2 = st.columns(2)
        with t1:
            if st.button("Sign In", key="tab_login",
                         type="primary" if st.session_state.auth_mode=="login" else "secondary",
                         use_container_width=True):
                st.session_state.auth_mode="login"; st.rerun()
        with t2:
            if st.button("Create Account", key="tab_signup",
                         type="primary" if st.session_state.auth_mode=="signup" else "secondary",
                         use_container_width=True):
                st.session_state.auth_mode="signup"; st.rerun()

        if st.session_state.auth_mode == "login":
            with st.form("lf"):
                st.markdown('<div class="auth-label">USERNAME</div>', unsafe_allow_html=True)
                lu = st.text_input("u", placeholder="your username", label_visibility="collapsed")
                st.markdown('<div class="auth-label">PASSWORD</div>', unsafe_allow_html=True)
                lp = st.text_input("p", type="password", placeholder="••••••••", label_visibility="collapsed")
                if st.form_submit_button("Sign In →", use_container_width=True, type="primary"):
                    if not lu.strip() or not lp.strip(): st.error("Fill in all fields.")
                    else:
                        ok, msg = do_login(lu.strip(), lp.strip())
                        if ok: st.rerun()
                        else: st.error(msg)
            st.markdown('<div class="auth-div">or</div>', unsafe_allow_html=True)
            if st.button("Continue as Guest →", type="secondary", use_container_width=True, key="guest_btn"):
                st.session_state.logged_in=True; st.session_state.current_user="__guest__"; st.rerun()
        else:
            with st.form("sf"):
                st.markdown('<div class="auth-label">DISPLAY NAME</div>', unsafe_allow_html=True)
                sdn = st.text_input("dn", placeholder="Optional", label_visibility="collapsed")
                st.markdown('<div class="auth-label">USERNAME</div>', unsafe_allow_html=True)
                su  = st.text_input("su", placeholder="At least 3 characters", label_visibility="collapsed", key="su")
                st.markdown('<div class="auth-label">PASSWORD</div>', unsafe_allow_html=True)
                sp  = st.text_input("sp", type="password", placeholder="At least 6 characters", label_visibility="collapsed", key="sp")
                st.markdown('<div class="auth-label">CONFIRM PASSWORD</div>', unsafe_allow_html=True)
                sp2 = st.text_input("sp2", type="password", placeholder="Repeat password", label_visibility="collapsed", key="sp2")
                if st.form_submit_button("Create Account →", use_container_width=True, type="primary"):
                    if sp != sp2: st.error("Passwords don't match.")
                    else:
                        ok, msg = do_signup(su.strip(), sp.strip(), sdn.strip())
                        if ok: st.rerun()
                        else: st.error(msg)

        st.markdown('<div style="text-align:center;font-size:.66rem;color:#3a3a5c;margin-top:1.25rem;">© 2025 QuizGenius AI · Design by Vishwas Patel</div>', unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════════════════════
cp = st.session_state.current_page
gen_ok = st.session_state.has_generated
uname  = st.session_state.current_user or "User"
is_guest = uname == "__guest__"
display_name = "Guest" if is_guest else uname.capitalize()
init = display_name[0].upper()

c0,c1,c2,c3,c4,c5,c6 = st.columns([2, .72, .82, .62, .88, .62, 1.7])
with c0:
    st.markdown('<div class="nb-brand"><div class="nb-logo">💡</div><div class="nb-title">QuizGenius <span>AI</span></div></div>', unsafe_allow_html=True)
with c1:
    if st.button("Home",      key="n_home",  type="primary" if cp=="Home"      else "secondary"): go("Home")
with c2:
    if st.button("Generate",  key="n_gen",   type="primary" if cp=="Generate"  else "secondary"): go("Generate")
with c3:
    if st.button("Study",     key="n_study", type="primary" if cp in ("Study","Flashcard","Test") else "secondary", disabled=not gen_ok): go("Study")
with c4:
    if st.button("Dashboard", key="n_dash",  type="primary" if cp=="Dashboard" else "secondary"): go("Dashboard")
with c5:
    if st.button("About",     key="n_about", type="primary" if cp=="About"     else "secondary"): go("About")
with c6:
    st.markdown(f'''<div class="nb-user">
      <div class="nb-sep"></div>
      <div class="nb-avatar">{init}</div>
      <button class="nb-out" onclick="var b=window.parent.document.querySelectorAll('button');for(var i=0;i<b.length;i++){{if(b[i].innerText.trim()=='\U0001f6aa'){{b[i].click();break;}}}}" title="Sign out">↪</button>
    </div>''', unsafe_allow_html=True)
    if st.button("🚪", key="n_logout", type="secondary"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

# ─── API KEY BANNER ────────────────────────────────────────────────────────────
if not get_key() and cp == "Generate":
    st.markdown("""<div style="background:rgba(255,71,87,.07);border:1px solid rgba(255,71,87,.2);border-radius:10px;
      padding:.875rem 1.375rem;margin:1rem 2rem;font-size:.85rem;color:#ff6b6b;">
      ⚠️ <strong>Groq API key not set.</strong> Set the <code>GROQ_API_KEY</code> env var on Render, or enter it below.
    </div>""", unsafe_allow_html=True)
    with st.expander("🔑 Enter Groq API Key (session only)"):
        ki = st.text_input("Key", type="password", placeholder="gsk_...", label_visibility="collapsed")
        if ki: st.session_state.groq_key_input = ki; st.success("Key saved for this session.")

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if cp == "Home":
    sh=st.session_state.score_history; tst=len(sh)
    avg=round(sum(s["pct"] for s in sh)/max(tst,1),1) if sh else 0
    best=max((s["pct"] for s in sh),default=0); tqg=len(st.session_state.questions)

    mockup="""<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:16px;padding:1.25rem;box-shadow:0 8px 40px rgba(0,0,0,.5);">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.75rem;">
        <span style="font-size:.52rem;font-weight:700;color:#5a5a7a;text-transform:uppercase;letter-spacing:.08em;background:#0f0f1e;padding:2px 9px;border-radius:5px;border:1px solid #2a2a3e;">Q3 / 10 · Medium</span>
        <span style="font-size:.58rem;color:#5a5a7a;">⏱ 38s</span>
      </div>
      <div style="font-size:.78rem;font-weight:700;color:#e8e8ff;line-height:1.6;margin-bottom:.75rem;">What is the primary function of mitochondria in a cell?</div>
      <div style="border:1px solid rgba(0,255,135,.3);border-radius:8px;padding:.4rem .75rem;margin-bottom:.3rem;font-size:.68rem;color:#00ff87;font-weight:600;display:flex;align-items:center;gap:.5rem;background:rgba(0,255,135,.05);">
        <span style="width:15px;height:15px;border-radius:50%;background:#00ff87;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:.5rem;color:#0a0a14;font-weight:800;">✓</span>A) Produce ATP through cellular respiration
      </div>
      <div style="padding:.4rem .75rem;margin-bottom:.3rem;font-size:.68rem;color:#3a3a5c;">B) Synthesise proteins for the nucleus</div>
      <div style="padding:.4rem .75rem;margin-bottom:.3rem;font-size:.68rem;color:#3a3a5c;">C) Control cell division and growth</div>
      <div style="padding:.4rem .75rem;font-size:.68rem;color:#3a3a5c;">D) Break down waste via autophagy</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.4rem;margin-top:.75rem;margin-bottom:.625rem;">
        <div style="background:#1a1a2e;color:#8892a4;border-radius:7px;padding:.45rem;text-align:center;font-size:.56rem;font-weight:700;border:1px solid #2a2a3e;">📖 Study</div>
        <div style="background:linear-gradient(135deg,#f5c518,#ff6b35);color:#0a0a14;border-radius:7px;padding:.45rem;text-align:center;font-size:.56rem;font-weight:700;">🎴 Cards</div>
        <div style="background:rgba(0,255,135,.1);color:#00ff87;border-radius:7px;padding:.45rem;text-align:center;font-size:.56rem;font-weight:700;border:1px solid rgba(0,255,135,.2);">🎯 Test</div>
      </div>
      <div style="background:rgba(245,197,24,.06);border:1px solid rgba(245,197,24,.15);border-radius:8px;padding:.5rem .875rem;display:flex;align-items:center;gap:.5rem;">
        <span style="font-size:1rem;">🏆</span>
        <div><div style="font-size:.6rem;font-weight:700;color:#f5c518;">Score: 9/10 · 90% — Outstanding!</div><div style="font-size:.54rem;color:#ff6b35;margin-top:.1rem;">Streak: 5 sessions</div></div>
      </div>
    </div>"""

    st.markdown(f"""<div class="hero-section">
      <div style="max-width:1240px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:4rem;align-items:center;padding-bottom:2rem;">
        <div>
          <div class="hero-badge">✦ Next-Gen AI Study Platform</div>
          <h1 class="hero-h1">Turn any PDF into<br><span class="acc">exam-ready</span><br>quizzes instantly</h1>
          <p class="hero-p">QuizGenius AI transforms study material into adaptive flashcards, MCQs, True/False, and fill-in-the-blank quizzes — powered by Groq's Llama 3.1.</p>
          <div class="pill-row">
            <div class="pill">📄 PDF & OCR</div>
            <div class="pill">🎴 Flashcards</div>
            <div class="pill">⏱ Timed Tests</div>
            <div class="pill">📊 Analytics</div>
          </div>
        </div>
        <div>{mockup}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    l,b1,b2,r = st.columns([.5,1.3,1.1,.5])
    with b1:
        if st.button("⚡  Start Generating Now", key="hero_cta", type="primary", use_container_width=True): go("Generate")
    with b2:
        if st.button("📊  My Dashboard", key="hero_dash", type="secondary", use_container_width=True): go("Dashboard")

    st.markdown(f'<div class="stats-bar"><div class="stats-inner"><div class="sc"><span class="sc-n">{tqg if tqg else "10K+"}</span><span class="sc-l">Questions Generated</span></div><div class="sc"><span class="sc-n">{tst if tst else "500+"}</span><span class="sc-l">Tests Taken</span></div><div class="sc"><span class="sc-n">{f"{avg}%" if sh else "95%"}</span><span class="sc-l">Average Score</span></div><div class="sc"><span class="sc-n">{f"{best}%" if sh else "Top"}</span><span class="sc-l">Best Score</span></div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="section-inner"><div class="sec-eyebrow">HOW IT WORKS</div><div class="sec-title">Three steps to smarter learning</div><p class="sec-sub">Upload any PDF, generate AI questions, then study and test yourself with full progress tracking.</p><div class="hw-grid"><div class="hw-card" data-n="1"><div class="hw-ico" style="background:rgba(74,158,255,.1);border-color:rgba(74,158,255,.2);">📄</div><div class="hw-t">1. Upload PDF</div><p class="hw-p">Drag and drop any PDF. OCR handles scanned documents automatically via Tesseract.</p></div><div class="hw-card" data-n="2"><div class="hw-ico" style="background:rgba(0,255,135,.1);border-color:rgba(0,255,135,.2);">🧠</div><div class="hw-t">2. AI Generates</div><p class="hw-p">Groq-powered Llama 3.1 extracts key concepts and creates questions at lightning speed.</p></div><div class="hw-card" data-n="3"><div class="hw-ico" style="background:rgba(245,197,24,.1);border-color:rgba(245,197,24,.2);">🎯</div><div class="hw-t">3. Study & Excel</div><p class="hw-p">Flip flashcards, bookmark tough questions, take timed tests, track your improvement.</p></div></div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section" style="background:var(--s1);border-top:1px solid var(--bd);border-bottom:1px solid var(--bd);"><div class="section-inner"><div class="sec-eyebrow">FEATURES</div><div class="sec-title">Everything you need to excel</div><p class="sec-sub">Built for serious learners and exam preparation.</p><br><div class="feat-grid"><div class="feat-card"><div class="feat-ico" style="background:rgba(245,197,24,.08);border-color:rgba(245,197,24,.15);">🎴</div><div><div class="feat-t">Flashcard Mode</div><div class="feat-s">3D flip cards with progress tracking and bookmarks.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:rgba(245,197,24,.08);border-color:rgba(245,197,24,.15);">⭐</div><div><div class="feat-t">Smart Bookmarks</div><div class="feat-s">Star any question and filter sessions to bookmarked items.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:rgba(0,255,135,.08);border-color:rgba(0,255,135,.15);">🎯</div><div><div class="feat-t">Auto Difficulty Detection</div><div class="feat-s">AI analyses PDF complexity and recommends Easy, Medium, or Hard.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:rgba(255,71,87,.08);border-color:rgba(255,71,87,.15);">❌</div><div><div class="feat-t">Wrong Answer Tracker</div><div class="feat-s">Test mistakes saved so you can revisit and improve.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:rgba(74,158,255,.08);border-color:rgba(74,158,255,.15);">⏱</div><div><div class="feat-t">Per-Question Timer</div><div class="feat-s">Countdown per question to simulate real exam pressure.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:rgba(0,255,135,.08);border-color:rgba(0,255,135,.15);">📊</div><div><div class="feat-t">Score History & Chart</div><div class="feat-s">Track every test and visualise improvement over time.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:rgba(74,158,255,.08);border-color:rgba(74,158,255,.15);">🗂️</div><div><div class="feat-t">Topic Filter</div><div class="feat-s">Extract chapters and focus generation on specific topics.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:rgba(245,197,24,.08);border-color:rgba(245,197,24,.15);">🔀</div><div><div class="feat-t">3 Question Types</div><div class="feat-s">MCQ, True/False, or Fill-in-the-blank.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:rgba(0,255,135,.08);border-color:rgba(0,255,135,.15);">👁</div><div><div class="feat-t">Preview First</div><div class="feat-s">Preview a sample question before generating the full set.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:rgba(74,158,255,.08);border-color:rgba(74,158,255,.15);">📤</div><div><div class="feat-t">Export as HTML</div><div class="feat-s">Download a formatted quiz sheet for offline study.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:rgba(255,71,87,.08);border-color:rgba(255,71,87,.15);">👤</div><div><div class="feat-t">User Accounts</div><div class="feat-s">Sign in to keep score history and bookmarks in session.</div></div></div><div class="feat-card"><div class="feat-ico" style="background:rgba(245,197,24,.08);border-color:rgba(245,197,24,.15);">⚡</div><div><div class="feat-t">Groq-Powered Speed</div><div class="feat-s">Llama 3.1 via Groq API — faster than local Ollama.</div></div></div></div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="section-inner"><div class="sec-eyebrow">DIFFICULTY LEVELS</div><div class="sec-title">Choose your challenge</div><p class="sec-sub">Three adaptive difficulty levels engineered to match your learning stage.</p><div class="df-grid"><div class="df-card e"><div style="font-size:1.75rem;margin-bottom:.625rem;">🌱</div><span class="df-pill e">Foundational</span><div class="df-name">Easy</div><p class="df-desc">Direct recall, key terminology, and fundamental concepts. 5 test questions.</p></div><div class="df-card m"><div style="font-size:1.75rem;margin-bottom:.625rem;">📈</div><span class="df-pill m">Intermediate</span><div class="df-name">Medium</div><p class="df-desc">Applied comprehension, connecting concepts and understanding causality. 7 questions.</p></div><div class="df-card h"><div style="font-size:1.75rem;margin-bottom:.625rem;">🔥</div><span class="df-pill h">Mastery</span><div class="df-name">Hard</div><p class="df-desc">Advanced synthesis and critical analysis. Designed for top-tier exam prep. 10 questions.</p></div></div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="site-footer"><div class="ft-inner"><div class="ft-top"><div><div class="ft-brand"><div class="ft-logo">🧠</div><span class="ft-name">QuizGenius AI</span></div><p class="ft-desc">Transform any PDF into an adaptive learning experience. Powered by Groq + Llama 3.1.</p></div><div><div class="ft-hd">Product</div><span class="ft-lk">Quiz Generator</span><span class="ft-lk">Flashcard Mode</span><span class="ft-lk">Adaptive Testing</span></div><div><div class="ft-hd">Resources</div><span class="ft-lk">Documentation</span><span class="ft-lk">Help Center</span><span class="ft-lk">About</span></div><div><div class="ft-hd">Connect</div><p style="font-size:.8rem;color:#3a3a5c;line-height:1.8;">patelvishwas702@gmail.com</p></div></div><div class="ft-bot"><span>© 2025 QuizGenius AI. All rights reserved.</span><span>Design by Vishwas Patel</span></div></div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# GENERATE
# ══════════════════════════════════════════════════════════════════════════════
elif cp == "Generate":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="page-bc">Workspace › AI Creation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-h1">Generate Study Material</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Upload your PDF to create AI-powered quizzes instantly via Groq + Llama 3.1.</p>', unsafe_allow_html=True)

    if not st.session_state.pdf_text.strip():
        uf = st.file_uploader("Upload your PDF", type="pdf")
        if uf:
            with st.spinner("Reading PDF…"):
                text = get_pdf_text(uf)
            if text.strip():
                clear_pdf()
                st.session_state.pdf_text     = text
                st.session_state.pdf_filename = uf.name
                st.session_state.pdf_size     = uf.size
                st.session_state.pdf_hash     = hashlib.md5(text.encode()).hexdigest()
                st.session_state.detected_difficulty = detect_diff(text)
                st.session_state.topics       = extract_topics(text)
                st.rerun()
            else: st.error("Could not extract text. Try another PDF.")
    else:
        pdf_text=st.session_state.pdf_text; wc=len(pdf_text.split()); max_q=calc_max_q(wc)
        fn=st.session_state.pdf_filename or "Uploaded PDF"; sz=st.session_state.pdf_size/1024; dd=st.session_state.detected_difficulty

        pb1,pb2=st.columns([5,1])
        with pb1:
            st.markdown(f'<div class="pdf-banner"><div class="pdf-icon">📄</div><div style="flex:1;min-width:0;"><div class="pdf-name">{S(fn)}</div><div class="pdf-meta">{wc:,} words · {sz:.1f} KB · AI suggests: <span class="diff-badge {dd}">{dd}</span></div></div></div>', unsafe_allow_html=True)
        with pb2:
            if st.button("Change", key="chg", type="secondary", use_container_width=True): clear_pdf(); st.rerun()

        st.markdown(f'<div class="stat4"><div class="stat4-card"><div class="stat4-label">Words</div><div class="stat4-val">{wc:,}</div></div><div class="stat4-card"><div class="stat4-label">Characters</div><div class="stat4-val">{len(pdf_text):,}</div></div><div class="stat4-card"><div class="stat4-label">File Size</div><div class="stat4-val">{sz:.0f} KB</div></div><div class="stat4-card acc"><div class="stat4-label">Max Questions</div><div class="stat4-val">{max_q}</div></div></div>', unsafe_allow_html=True)

        if st.session_state.topics:
            with st.expander(f"🗂️ Topic Filter — {len(st.session_state.topics)} sections detected"):
                st.caption("Select sections to focus on. Leave empty for full document.")
                sel=st.multiselect("Topics",st.session_state.topics,default=st.session_state.selected_topics,label_visibility="collapsed")
                st.session_state.selected_topics=sel
                if sel: st.success(f"Focusing on {len(sel)} topic(s).")

        st.markdown('<div class="cfg-card">', unsafe_allow_html=True)
        num_q=st.slider("Questions",1,max_q,min(10,max_q),label_visibility="collapsed")
        st.markdown(f'<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;margin-top:.5rem;"><div><div class="cfg-title">Question Count</div><div class="cfg-hint">How many questions to generate.</div></div><div class="cfg-pill"><span class="cfg-num">{num_q}</span><div class="cfg-lbl">Questions</div></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cfg-card"><div style="font-size:.875rem;font-weight:700;color:var(--tx);margin-bottom:.875rem;">Question Type</div>', unsafe_allow_html=True)
        cqt=st.session_state.q_type
        qa1,qa2,qa3=st.columns(3)
        for col,qt,ico,lbl,sub in [(qa1,"MCQ","📝","Multiple Choice","4 options, one correct"),(qa2,"TF","✅","True / False","Fact-based statements"),(qa3,"FIB","✏️","Fill in Blank","Complete the sentence")]:
            with col:
                st.markdown(f'<div class="qt-card {"sel" if cqt==qt else ""}"><div class="qt-ico">{ico}</div><div class="qt-t">{lbl}</div><div class="qt-s">{sub}</div></div>', unsafe_allow_html=True)
                if st.button(f"Select {lbl}",key=f"qt_{qt}",type="primary" if cqt==qt else "secondary",use_container_width=True):
                    st.session_state.q_type=qt; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        _,gc,_=st.columns([1,2,1])
        with gc:
            prev_click=st.button("👁  Preview Sample",key="prev_btn",type="secondary",use_container_width=True)
            gen_click =st.button("⚡  Generate Questions",type="primary",use_container_width=True,key="gen_btn")
            if st.session_state.generation_done:
                if st.button("📖  Open Study Mode →",type="primary",use_container_width=True,key="goto_s2"): go("Study")

        if prev_click:
            if not get_key(): st.error("Enter your Groq API key first.")
            else:
                with st.spinner("Generating preview…"):
                    try:
                        chunks=[pdf_text[i:i+1200] for i in range(0,len(pdf_text),1100)] if not LC_AVAILABLE else RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=100).split_text(pdf_text)
                        pq=llm_gen(chunks[0],st.session_state.q_type,dd)
                        st.session_state.preview_question=pq; st.session_state.show_preview=True
                    except Exception as e: st.error(f"Preview failed: {e}")

        if st.session_state.show_preview and st.session_state.preview_question:
            pq=st.session_state.preview_question
            popts="".join(
                f'<div style="padding:.5rem .875rem;border-radius:8px;margin:.25rem 0;'
                f'background:{"rgba(0,255,135,.06)" if o[0]==pq["correct"] else "transparent"};'
                f'border-left:2px solid {"var(--green)" if o[0]==pq["correct"] else "transparent"};'
                f'font-size:.85rem;font-weight:{"700" if o[0]==pq["correct"] else "400"};'
                f'color:{"var(--green)" if o[0]==pq["correct"] else "var(--tx3)"};">{S(o)}</div>'
                for o in pq["options"] if o.strip()
            )
            st.markdown(f'<div class="prev-card"><div class="prev-badge">Preview · {S(pq["type"])} · {S(pq["difficulty"])}</div><div class="prev-q">{S(pq["question"])}</div>{popts}</div>', unsafe_allow_html=True)

        prog_ph=st.empty()

        if gen_click:
            if not get_key(): st.error("Enter your Groq API key first.")
            else:
                cur_hash=st.session_state.pdf_hash
                for k,v in [("questions",[]),("test_questions",[]),("has_generated",False),("has_test_generated",False),("generation_done",False),("questions_pdf_hash",""),("selected_difficulty",None),("user_answers",{}),("test_submitted",False),("chunks",[]),("bookmarks",[]),("wrong_answers",[])]:
                    st.session_state[k]=v
                st.session_state.quiz_key+=1

                def upd(pct,msg):
                    prog_ph.markdown(f'<div class="prog-wrap"><div class="prog-top"><span class="prog-label">⚡ {msg}</span><span class="prog-pct">{pct}%</span></div><div class="prog-bar"><div class="prog-fill" style="width:{pct}%"></div></div><p class="prog-note">Groq + Llama 3.1 is working. Usually under 30s.</p></div>', unsafe_allow_html=True)

                upd(10,"Splitting document…")
                use_text=pdf_text
                if st.session_state.selected_topics:
                    filt=""
                    for t in st.session_state.selected_topics:
                        idx=pdf_text.lower().find(t.lower())
                        if idx>=0: filt+=pdf_text[idx:idx+3000]+"\n\n"
                    if filt.strip(): use_text=filt

                if LC_AVAILABLE:
                    chunks=RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=100).split_text(use_text)
                else:
                    chunks=[use_text[i:i+1200] for i in range(0,len(use_text),1100)]
                st.session_state.chunks=chunks

                upd(20,"Building search index…"); qt=st.session_state.q_type; temp_qs=[]
                for i in range(num_q):
                    upd(20+int((i+1)*75/num_q),f"Generating question {i+1} of {num_q}…")
                    rel=keyword_search("key concepts important facts",chunks,k=3)
                    ctx="\n".join(rel[:2]) if rel else chunks[i%len(chunks)]
                    try: temp_qs.append(llm_gen(ctx,qt,dd))
                    except Exception as e:
                        temp_qs.append({"question":f"[Error: {str(e)[:80]}]","options":["A) --","B) --","C) --","D) --"],"correct":"A","context":"","type":qt,"difficulty":dd})

                upd(100,"Done!")
                st.session_state.questions=temp_qs; st.session_state.questions_pdf_hash=cur_hash
                st.session_state.has_generated=True; st.session_state.generation_done=True; st.rerun()

        if st.session_state.generation_done and st.session_state.has_generated:
            qt_lbl={"MCQ":"Multiple Choice","TF":"True/False","FIB":"Fill-in-the-Blank"}.get(st.session_state.q_type,"")
            st.success(f"✅ Generated {len(st.session_state.questions)} {qt_lbl} questions!")
            _,sc2,_=st.columns([1,2,1])
            with sc2:
                if st.button("📖 Open Study Mode",type="primary",use_container_width=True,key="goto_study"): go("Study")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STUDY / FLASHCARD / TEST
# ══════════════════════════════════════════════════════════════════════════════
elif cp in ("Study","Flashcard","Test"):
    if not gen_ok:
        st.markdown('<div style="max-width:480px;margin:5rem auto;text-align:center;padding:2rem;"><div style="font-size:3rem;margin-bottom:1rem;">📄</div><div style="font-family:Clash Display,sans-serif;font-size:1.375rem;font-weight:700;color:var(--tx);margin-bottom:.5rem;">No Questions Yet</div><div style="font-size:.875rem;color:var(--tx3);margin-bottom:2rem;">Upload a PDF and generate questions first.</div></div>', unsafe_allow_html=True)
        _,c2,_=st.columns([1,1,1])
        with c2:
            if st.button("⚡ Go Generate",type="primary",use_container_width=True): go("Generate")
        st.stop()

    if (st.session_state.questions_pdf_hash and st.session_state.pdf_hash
            and st.session_state.questions_pdf_hash!=st.session_state.pdf_hash):
        st.warning("New PDF loaded — please regenerate questions.")
        if st.button("⚡ Regenerate",type="primary"): go("Generate")
        st.stop()

    st.markdown('<div class="study-wrap">', unsafe_allow_html=True)
    tc1,tc2,tc3=st.columns(3)
    with tc1:
        if st.button("📖 Study",     key="tab_s", type="primary" if cp=="Study"     else "secondary",use_container_width=True): go("Study")
    with tc2:
        if st.button("🎴 Flashcards",key="tab_f", type="primary" if cp=="Flashcard" else "secondary",use_container_width=True): go("Flashcard")
    with tc3:
        if st.button("🎯 Test",      key="tab_t", type="primary" if cp=="Test"      else "secondary",use_container_width=True):
            st.session_state.selected_difficulty=None; st.session_state.has_test_generated=False
            st.session_state.user_answers={}; st.session_state.test_submitted=False; go("Test")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── STUDY ──────────────────────────────────────────────────────────────────
    if cp=="Study":
        qs=st.session_state.questions; bms=st.session_state.bookmarks; fn=st.session_state.pdf_filename or "PDF"
        sh1,sh2,sh3=st.columns([2.5,1,1])
        with sh1: st.markdown(f'<div style="display:flex;align-items:center;gap:.75rem;margin-bottom:1.5rem;"><span class="study-title">Study Questions</span><span class="study-badge">{len(qs)} Qs · {st.session_state.q_type}</span></div>', unsafe_allow_html=True)
        with sh2: fm=st.selectbox("Filter",["All","Bookmarked","Wrong Answers"],label_visibility="collapsed",key="sf")
        with sh3: st.download_button("📤 Export",export_html(qs,f"QuizGenius — {fn}").encode(),f"quiz_{fn.replace('.pdf','')}.html","text/html",use_container_width=True)

        with st.expander("📋 Copy all as text"):
            st.code("\n\n".join(f"Q{i+1}: {q['question']}\n"+"\n".join(q['options'])+f"\nAnswer: {q['correct']}" for i,q in enumerate(qs)),language=None)

        if fm=="Bookmarked":
            dqs=[(i,q) for i,q in enumerate(qs) if i in bms]
            if not dqs: st.info("No bookmarks yet.")
        elif fm=="Wrong Answers":
            wa={q["question"] for q in st.session_state.wrong_answers}
            dqs=[(i,q) for i,q in enumerate(qs) if q["question"] in wa]
            if not dqs: st.info("No wrong answers yet — take a test first.")
        else: dqs=list(enumerate(qs))

        for idx,q in dqs:
            is_bm=idx in bms
            st.markdown(f'<div class="mcq-card"><div class="mcq-head"><span class="mcq-q-num">Q{idx+1}</span><span class="mcq-type">{S(q.get("type","MCQ"))}</span></div><div class="mcq-q">{S(q["question"])}</div>', unsafe_allow_html=True)
            seen=set()
            for opt in q["options"]:
                raw=opt.strip()
                if not raw: continue
                lt=raw[0]
                if lt not in "ABCD" or lt in seen: continue
                seen.add(lt); txt=clean_opt(raw)
                if lt==q["correct"]: st.markdown(f'<div class="opt-ok"><div class="opt-lt-ok">{S(lt)}</div><div class="opt-tx-ok">{S(txt)}</div><span style="margin-left:auto;color:var(--green);">✓</span></div>', unsafe_allow_html=True)
                else: st.markdown(f'<div class="opt-no"><div class="opt-lt">{S(lt)}</div><div class="opt-tx">{S(txt)}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="mcq-footer">', unsafe_allow_html=True)
            bc,cc=st.columns([1,4])
            with bc:
                if st.button("⭐ Saved" if is_bm else "☆ Bookmark",key=f"bm_{idx}_{st.session_state.quiz_key}",type="primary" if is_bm else "secondary",use_container_width=True):
                    if idx in st.session_state.bookmarks: st.session_state.bookmarks.remove(idx)
                    else: st.session_state.bookmarks.append(idx)
                    persist(); st.rerun()
            with cc:
                with st.expander(f"Source context — Q{idx+1}"):
                    st.markdown(f'<div style="background:var(--s3);border-left:3px solid var(--gold);padding:.75rem 1rem;border-radius:7px;font-size:.825rem;color:var(--tx2);line-height:1.7;">{S(q["context"])}</div>', unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

    # ── FLASHCARD ──────────────────────────────────────────────────────────────
    elif cp=="Flashcard":
        fc_qs=get_fc_qs(); total=len(fc_qs)
        st.markdown('<div style="max-width:660px;margin:0 auto;">', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;margin-bottom:1.5rem;"><div style="font-family:Clash Display,sans-serif;font-size:1.75rem;font-weight:700;color:var(--tx);margin-bottom:.375rem;">Flashcards</div><div style="font-size:.85rem;color:var(--tx3);">Click the card to reveal the answer.</div></div>', unsafe_allow_html=True)

        f1,f2,f3=st.columns(3)
        for fi,flt in zip([f1,f2,f3],["All","Bookmarked","Mistakes"]):
            with fi:
                if st.button(flt,key=f"fcf_{flt}",type="primary" if st.session_state.fc_filter==flt else "secondary",use_container_width=True):
                    st.session_state.fc_filter=flt; st.session_state.fc_idx=0; st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

        if not fc_qs:
            st.info("No cards match this filter." if st.session_state.fc_filter!="All" else "No questions yet.")
        else:
            idx=min(st.session_state.fc_idx,total-1); oi,q=fc_qs[idx]
            is_bm2=oi in st.session_state.bookmarks
            ans=clean_opt(next((o for o in q["options"] if o.startswith(q["correct"])),q["correct"]))
            render_flashcard(q["question"],ans,idx+1,total,is_bm2)
            st.markdown("<br>", unsafe_allow_html=True)
            n1,n2,n3,n4=st.columns(4)
            with n1:
                if st.button("← Prev",  key="fp",type="secondary",use_container_width=True): st.session_state.fc_idx=(idx-1)%total; st.rerun()
            with n2:
                if st.button("🔀 Random",key="fr",type="secondary",use_container_width=True):
                    import random; st.session_state.fc_idx=random.randint(0,total-1); st.rerun()
            with n3:
                if st.button("Next →",  key="fn",type="secondary",use_container_width=True): st.session_state.fc_idx=(idx+1)%total; st.rerun()
            with n4:
                if st.button("⭐ Saved" if is_bm2 else "☆ Save",key=f"fb_{idx}",type="primary" if is_bm2 else "secondary",use_container_width=True):
                    if oi in st.session_state.bookmarks: st.session_state.bookmarks.remove(oi)
                    else: st.session_state.bookmarks.append(oi)
                    persist(); st.rerun()
            st.markdown(f'<div style="text-align:center;margin-top:.75rem;font-size:.82rem;font-weight:600;color:var(--tx3);">{idx+1} / {total} · {st.session_state.fc_filter}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── TEST ───────────────────────────────────────────────────────────────────
    elif cp=="Test":
        if not st.session_state.selected_difficulty:
            st.markdown('<div class="tcard-info">📝 Test questions are generated fresh via Groq for each difficulty level — separate from your study questions.</div>', unsafe_allow_html=True)
            tm1,tm2=st.columns([2,2])
            with tm1:
                timed=st.checkbox("⏱ Enable per-question timer",value=st.session_state.timed_mode)
                st.session_state.timed_mode=timed
            if timed:
                with tm2:
                    secs=st.slider("Seconds",15,120,st.session_state.per_q_time,5,label_visibility="collapsed")
                    st.session_state.per_q_time=secs; st.caption(f"⏱ {secs}s per question")

            st.markdown('<br><div style="text-align:center;"><div style="font-family:Clash Display,sans-serif;font-size:1.5rem;font-weight:700;color:var(--tx);margin-bottom:.375rem;">Select Your Challenge Level</div><div style="font-size:.875rem;color:var(--tx3);margin-bottom:1.5rem;">Fresh questions generated for each level.</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="td-grid"><div class="td-card"><div class="td-ico e">🌱</div><div class="td-name">Beginner</div><div class="td-hint">Foundational concepts. 5 questions.</div></div><div class="td-card feat"><span class="td-pop">POPULAR</span><div class="td-ico m">🎓</div><div class="td-name">Standard</div><div class="td-hint">Applied scenarios. 7 questions.</div></div><div class="td-card"><div class="td-ico h">🔥</div><div class="td-name">Expert</div><div class="td-hint">Complex synthesis. 10 questions.</div></div></div>', unsafe_allow_html=True)
            d1,d2,d3=st.columns(3)
            with d1:
                if st.button("🌱 Start Easy",  key="easy",use_container_width=True,type="secondary"):
                    st.session_state.selected_difficulty="Easy"; st.session_state.test_started_at=int(time.time()*1000); st.rerun()
            with d2:
                if st.button("🎓 Start Medium",key="med", use_container_width=True,type="primary"):
                    st.session_state.selected_difficulty="Medium"; st.session_state.test_started_at=int(time.time()*1000); st.rerun()
            with d3:
                if st.button("🔥 Start Hard",  key="hard",use_container_width=True,type="secondary"):
                    st.session_state.selected_difficulty="Hard"; st.session_state.test_started_at=int(time.time()*1000); st.rerun()

        elif not st.session_state.has_test_generated:
            diff=st.session_state.selected_difficulty; n={"Easy":5,"Medium":7,"Hard":10}[diff]
            if not get_key():
                st.error("Groq API key required.")
                if st.button("← Back"): st.session_state.selected_difficulty=None; st.rerun()
            else:
                st.markdown(f'<div style="background:var(--s2);border:1px solid var(--bd);border-radius:14px;padding:2.5rem;text-align:center;"><div style="width:56px;height:56px;border-radius:50%;background:rgba(245,197,24,.1);border:1px solid rgba(245,197,24,.2);display:flex;align-items:center;justify-content:center;margin:0 auto 1.125rem;font-size:1.5rem;">🧠</div><div style="font-family:Clash Display,sans-serif;font-size:1.125rem;font-weight:700;color:var(--tx);margin-bottom:.375rem;">Generating {diff} Test</div><div style="font-size:.875rem;color:var(--tx3);">Creating {n} fresh questions via Groq…</div></div>', unsafe_allow_html=True)
                prog=st.progress(0,text=f"Generating {diff} test…")
                st.session_state.test_questions=[]; st.session_state.user_answers={}; st.session_state.test_submitted=False
                chunks=st.session_state.chunks; qt=st.session_state.q_type; temp=[]
                for i in range(n):
                    prog.progress(int((i+1)*100/n),text=f"Question {i+1}/{n}…")
                    rel=keyword_search(f"{diff} level concepts",chunks,k=3)
                    ctx="\n".join(rel[:2]) if rel else (chunks[i%len(chunks)] if chunks else st.session_state.pdf_text[:1500])
                    try: temp.append(llm_gen(ctx,qt,diff))
                    except Exception as e:
                        temp.append({"question":f"[Error: {str(e)[:80]}]","options":["A) --","B) --","C) --","D) --"],"correct":"A","context":"","type":qt,"difficulty":diff})
                prog.progress(100,text="Ready!")
                st.session_state.test_questions=temp; st.session_state.has_test_generated=True; prog.empty(); st.rerun()

        elif not st.session_state.test_submitted:
            diff=st.session_state.selected_difficulty; tq=len(st.session_state.test_questions)
            ans=len(st.session_state.user_answers); pct_d=int(ans/tq*100) if tq else 0
            started=st.session_state.test_started_at or int(time.time()*1000)

            st.markdown(f"""<div class="tq-progress">
              <div style="display:flex;flex-direction:column;gap:4px;flex:1;">
                <div style="display:flex;justify-content:space-between;">
                  <span style="font-size:.7rem;font-weight:700;color:var(--gold);">{ans}/{tq} answered</span>
                  <span style="font-size:.7rem;font-weight:700;color:var(--tx3);">{pct_d}%</span>
                </div>
                <div class="tq-pbar"><div class="tq-pfill" style="width:{pct_d}%"></div></div>
              </div>
              <div class="tq-timer">⏱ <span id="tf">00:00</span></div>
            </div>
            <script>(function(){{var s={started};setInterval(function(){{var e=document.getElementById("tf");if(e){{var t=Math.floor((Date.now()-s)/1000);e.innerText=(Math.floor(t/60)<10?"0":"")+Math.floor(t/60)+":"+(t%60<10?"0":"")+t%60;}}}},500);}})();</script>
            """, unsafe_allow_html=True)

            for i,q in enumerate(st.session_state.test_questions):
                d=q.get("difficulty",diff); dc={"Easy":"tq-dm","Medium":"tq-de","Hard":"tq-dh"}.get(d,"tq-de")
                st.markdown(f'<div class="tq"><div class="tq-top"><span class="tq-num">Q{i+1}</span><span class="{dc}">{S(d)}</span></div><div class="tq-q">{S(q["question"])}</div></div>', unsafe_allow_html=True)
                copts=[]; seen=set()
                for opt in q["options"]:
                    raw=opt.strip()
                    if not raw: continue
                    lt=raw[0]
                    if lt not in "ABCD" or lt in seen: continue
                    seen.add(lt); copts.append(f"{lt}) {clean_opt(raw)}")
                a=st.radio(f"q{i}",copts,index=None,key=f"tq_{i}_{st.session_state.quiz_key}",label_visibility="collapsed")
                if a: st.session_state.user_answers[i]=a[0]
                st.markdown("<br>", unsafe_allow_html=True)

            s1,s2,s3=st.columns([1,2,1])
            with s2:
                if ans==tq:
                    if st.button("✅ Submit Test",use_container_width=True,type="primary"):
                        save_score(diff,sum(1 for i,q in enumerate(st.session_state.test_questions) if st.session_state.user_answers.get(i)==q["correct"]),tq,st.session_state.pdf_filename or "Unknown")
                        st.session_state.test_submitted=True; st.rerun()
                else: st.warning(f"Answer all {tq} questions first. ({ans}/{tq})")

        else:
            diff=st.session_state.selected_difficulty
            corr=sum(1 for i,q in enumerate(st.session_state.test_questions) if st.session_state.user_answers.get(i)==q["correct"])
            total=len(st.session_state.test_questions); pct=corr/total*100 if total else 0; wc2=total-corr
            render_result(pct,corr,total)
            if wc2>0: st.markdown(f'<div style="background:rgba(255,71,87,.07);border:1px solid rgba(255,71,87,.2);border-radius:10px;padding:.75rem 1.125rem;font-size:.85rem;color:#ff6b6b;margin:.75rem 0;">⚠️ {wc2} wrong answer(s) saved — review via Study → Wrong Answers or Flashcards → Mistakes.</div>', unsafe_allow_html=True)

            st.markdown('<div style="background:var(--s2);border:1px solid var(--bd);border-radius:16px;overflow:hidden;margin-top:.75rem;">', unsafe_allow_html=True)
            st.markdown('<div style="padding:1.375rem;"><div style="font-size:.875rem;font-weight:700;color:var(--tx);margin-bottom:.875rem;">📋 Detailed Review</div>', unsafe_allow_html=True)
            for i,q in enumerate(st.session_state.test_questions):
                ua=st.session_state.user_answers.get(i); ok=ua==q["correct"]
                lc="var(--green)" if ok else "var(--red)"; yt=ua or "--"; ct=q["correct"]
                for opt in q["options"]:
                    if opt and len(opt)>1:
                        if opt[0]==ua: yt=clean_opt(opt)
                        if opt[0]==q["correct"]: ct=clean_opt(opt)
                wrong=f'<br>Correct: <span style="color:var(--green);font-weight:700;">{S(ct)}</span>' if not ok else ""
                st.markdown(f'<div class="rv {"rv-c" if ok else "rv-w"}"><div style="font-size:.9rem;flex-shrink:0;color:{lc};">{"✓" if ok else "✗"}</div><div style="flex:1;"><div class="rv-q">Q{i+1}: {S(q["question"])}</div><div class="rv-a">Your answer: <span style="color:{lc};font-weight:600;">{S(str(yt))}</span>{wrong}</div></div><span class="rv-lbl {"ok" if ok else "ng"}">{"Correct" if ok else "Wrong"}</span></div>', unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

            r1,r2,r3,r4=st.columns(4)
            with r1:
                if st.button("🔄 Retake",    use_container_width=True,type="secondary"):
                    st.session_state.user_answers={}; st.session_state.test_submitted=False; st.session_state.quiz_key+=1; st.rerun()
            with r2:
                if st.button("🎯 New Level", use_container_width=True,type="primary"):
                    st.session_state.selected_difficulty=None; st.session_state.has_test_generated=False
                    st.session_state.user_answers={}; st.session_state.test_submitted=False; st.rerun()
            with r3:
                if st.button("🎴 Cards",     use_container_width=True,type="secondary"): go("Flashcard")
            with r4:
                if st.button("📊 Dashboard", use_container_width=True,type="secondary"): go("Dashboard")

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif cp=="Dashboard":
    sh=st.session_state.score_history; qs=st.session_state.questions
    bms=st.session_state.bookmarks; was=st.session_state.wrong_answers
    tt=len(sh); avg=round(sum(s["pct"] for s in sh)/max(tt,1),1) if sh else 0
    best=max((s["pct"] for s in sh),default=0)
    un=st.session_state.current_user or "User"
    greet=f"Welcome back, {un.capitalize()}!" if un!="__guest__" else "Welcome, Guest!"

    st.markdown('<div class="dash-wrap">', unsafe_allow_html=True)
    st.markdown(f'<div class="page-bc">Your Progress › Dashboard</div><div class="dash-h">Learning Dashboard</div><div style="font-size:.875rem;color:var(--tx3);margin-top:.375rem;margin-bottom:2rem;line-height:1.75;">{S(greet)} Track your progress and study patterns.</div>', unsafe_allow_html=True)

    st.markdown(f'''<div class="stat-grid">
      <div class="stat-card"><span class="stat-ico">📊</span><div class="stat-v">{tt}</div><div class="stat-l">Tests Taken</div></div>
      <div class="stat-card"><span class="stat-ico">🎯</span><div class="stat-v">{avg}%</div><div class="stat-l">Avg Score</div></div>
      <div class="stat-card"><span class="stat-ico">🏆</span><div class="stat-v">{best}%</div><div class="stat-l">Best Score</div></div>
      <div class="stat-card"><span class="stat-ico">⭐</span><div class="stat-v">{len(bms)}</div><div class="stat-l">Bookmarks</div></div>
    </div>''', unsafe_allow_html=True)
    d1,d2=st.columns(2)
    with d1: st.markdown(f'<div class="stat-card" style="margin-bottom:1rem;"><span class="stat-ico">📝</span><div class="stat-v">{len(qs)}</div><div class="stat-l">Questions Generated</div></div>', unsafe_allow_html=True)
    with d2: st.markdown(f'<div class="stat-card" style="margin-bottom:1rem;"><span class="stat-ico">❌</span><div class="stat-v">{len(was)}</div><div class="stat-l">Mistakes Tracked</div></div>', unsafe_allow_html=True)

    if len(sh)>=2:
        st.markdown('<div class="ds-title">📈 Score Trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        render_score_chart(sh)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="ds-title">🗂️ Score History</div>', unsafe_allow_html=True)
    if not sh:
        st.markdown('<div style="background:var(--s2);border:1px solid var(--bd);border-radius:13px;padding:2.5rem;text-align:center;"><div style="font-size:2.25rem;margin-bottom:.75rem;">🧪</div><div style="font-family:Clash Display,sans-serif;font-size:.95rem;font-weight:700;color:var(--tx);">No tests taken yet</div><div style="font-size:.82rem;color:var(--tx3);margin-top:.375rem;">Complete a test to see your history here.</div></div>', unsafe_allow_html=True)
        _,bc,_=st.columns([1,1,1])
        with bc:
            if st.button("🎯 Take a Test Now",type="primary",use_container_width=True): go("Test")
    else:
        for e in reversed(sh):
            pc=e["pct"]; bc2="var(--green)" if pc>=80 else "var(--gold)" if pc>=60 else "var(--red)"
            st.markdown(f'<div class="sh-row"><span class="sh-diff {e["diff"]}">{e["diff"]}</span><div class="sh-score">{e["score"]}/{e["total"]}</div><div class="sh-meta"><div class="sh-pdf">{S(e["pdf"])}</div><div class="sh-date">{e["date"]}</div></div><div class="sh-pbar"><div class="sh-pfill" style="width:{pc}%;background:{bc2};"></div></div><span class="sh-pct" style="color:{bc2};">{pc}%</span></div>', unsafe_allow_html=True)
        _,cl,_=st.columns([1,1,1])
        with cl:
            if st.button("🗑 Clear History",type="secondary",use_container_width=True):
                st.session_state.score_history=[]; persist(); st.rerun()

    if was:
        st.markdown(f'<div class="ds-title">❌ Wrong Answers ({len(was)})</div>', unsafe_allow_html=True)
        for wq in was[:4]:
            ct=clean_opt(next((o for o in wq["options"] if o.startswith(wq["correct"])),wq["correct"]))
            st.markdown(f'<div class="rv rv-w"><div style="color:var(--red);font-size:.9rem;flex-shrink:0;">✗</div><div style="flex:1;"><div class="rv-q">{S(wq["question"])}</div><div class="rv-a">Correct: <span style="color:var(--green);font-weight:600;">{S(ct)}</span></div></div></div>', unsafe_allow_html=True)
        if len(was)>4: st.caption(f"+ {len(was)-4} more. See all in Study → Wrong Answers.")
        w1,w2=st.columns(2)
        with w1:
            if st.button("📖 Study Wrong Answers",type="secondary",use_container_width=True): go("Study")
        with w2:
            if st.button("🎴 Flashcard Mistakes",type="secondary",use_container_width=True):
                st.session_state.fc_filter="Mistakes"; go("Flashcard")

    if bms:
        st.markdown(f'<div class="ds-title">⭐ Bookmarks ({len(bms)})</div>', unsafe_allow_html=True)
        for bi in bms[:3]:
            if bi<len(qs): st.markdown(f'<div class="rv" style="border-left:3px solid var(--gold);background:rgba(245,197,24,.04);"><div style="flex-shrink:0;">⭐</div><div class="rv-q">{S(qs[bi]["question"])}</div></div>', unsafe_allow_html=True)
        if len(bms)>3: st.caption(f"+ {len(bms)-3} more bookmarks.")
        if st.button("📖 Study Bookmarks",type="secondary"): go("Study")

    if un=="__guest__":
        st.markdown('<div style="background:rgba(245,197,24,.05);border:1px solid rgba(245,197,24,.15);border-radius:11px;padding:1rem 1.375rem;font-size:.85rem;color:#f5c518;margin-top:1.25rem;">💡 Guest mode — data resets on page refresh. Create an account to keep session data.</div>', unsafe_allow_html=True)
        _,bc3,_=st.columns([1,1,1])
        with bc3:
            if st.button("Create Account",type="primary",use_container_width=True):
                for k in list(st.session_state.keys()): del st.session_state[k]; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif cp=="About":
    st.markdown('<div class="about-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="page-bc">QuizGenius AI › About</div><div class="page-h1">About QuizGenius AI</div><p class="page-sub">Revolutionising learning through adaptive AI-powered assessments.</p>', unsafe_allow_html=True)
    cs ="background:var(--s2);border:1px solid var(--bd);border-radius:15px;padding:1.875rem;margin-bottom:1.25rem;"
    ct2="font-family:'Clash Display',sans-serif;font-size:1rem;font-weight:700;color:var(--tx);margin-bottom:.75rem;"
    cp2="font-size:.85rem;color:var(--tx2);line-height:1.85;"
    a1,a2=st.columns(2)
    with a1:
        st.markdown(f'<div style="{cs}"><div style="{ct2}">🧠 Our Mission</div><p style="{cp2}">QuizGenius AI empowers students and professionals to learn smarter. Groq-powered Llama 3.1 generates adaptive quizzes with flashcards and real-time progress tracking. Fast, free, and cloud-deployable on Render.</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="{cs}"><div style="{ct2}">🚀 Technology Stack</div><p style="{cp2}"><strong style="color:var(--gold);">Llama 3.1</strong> — via Groq API<br><strong style="color:var(--gold);">LangChain</strong> — text splitting<br><strong style="color:var(--gold);">Tesseract OCR</strong> — scanned PDFs<br><strong style="color:var(--gold);">Streamlit</strong> — web framework<br><strong style="color:var(--gold);">Render</strong> — cloud deployment</p></div>', unsafe_allow_html=True)
    with a2:
        st.markdown(f'<div style="{cs}"><div style="{ct2}">✨ Key Features</div><p style="{cp2}">🎴 Flashcard mode with 3D flip<br>⭐ Bookmark questions<br>🎯 Auto difficulty detection<br>❌ Wrong answer tracker<br>⏱ Per-question timed mode<br>📊 Score history + chart<br>🗂️ Topic / chapter filter<br>🔀 MCQ / True-False / Fill-in-Blank<br>👁 Preview before generation<br>📤 Export as formatted HTML<br>⚡ Groq-powered speed</p></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="{cs}">
          <div style="{ct2}">📞 Connect</div>
          <a href="https://www.linkedin.com/in/vishwas-patel-ba91a2288/" target="_blank"
            style="display:flex;align-items:center;gap:.75rem;padding:.625rem .875rem;
            background:rgba(0,119,181,.08);border:1px solid rgba(0,119,181,.2);border-radius:10px;
            text-decoration:none;margin-bottom:.5rem;">
            <div style="width:34px;height:34px;border-radius:8px;background:#0077b5;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="white"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
            </div>
            <div><div style="font-size:.8rem;font-weight:700;color:#4da6d9;">LinkedIn</div><div style="font-size:.68rem;color:var(--tx3);">Vishwas Patel</div></div>
            <div style="margin-left:auto;color:var(--tx3);">↗</div>
          </a>
          <a href="mailto:patelvishwas702@gmail.com"
            style="display:flex;align-items:center;gap:.75rem;padding:.625rem .875rem;
            background:rgba(234,67,53,.08);border:1px solid rgba(234,67,53,.2);border-radius:10px;text-decoration:none;">
            <div style="width:34px;height:34px;border-radius:8px;background:#ea4335;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="white"><path d="M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 0 1 0 19.366V5.457c0-2.023 2.309-3.178 3.927-1.964L5.455 4.64 12 9.548l6.545-4.91 1.528-1.145C21.69 2.28 24 3.434 24 5.457z"/></svg>
            </div>
            <div><div style="font-size:.8rem;font-weight:700;color:#f87171;">Email</div><div style="font-size:.68rem;color:var(--tx3);">patelvishwas702@gmail.com</div></div>
            <div style="margin-left:auto;color:var(--tx3);">↗</div>
          </a>
          <div style="padding-top:.75rem;border-top:1px solid var(--bd);margin-top:.625rem;">
            <p style="font-size:.75rem;color:var(--tx3);margin:0;">Design & Dev by <strong style="color:var(--tx);">Vishwas Patel</strong></p>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
        
