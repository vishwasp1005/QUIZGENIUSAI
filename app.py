import streamlit as st
import streamlit.components.v1 as components
import hashlib, re, html as html_module, time, os, json, datetime
import PyPDF2

st.set_page_config(page_title="QuizGenius AI", page_icon="🧠",
                   layout="wide", initial_sidebar_state="collapsed")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"

try:
    import pytesseract; from pdf2image import convert_from_bytes; OCR_AVAILABLE = True
except ImportError: OCR_AVAILABLE = False
try:
    from groq import Groq; GROQ_AVAILABLE = True
except ImportError: GROQ_AVAILABLE = False
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter; LC_AVAILABLE = True
except ImportError: LC_AVAILABLE = False

_D = {
    "logged_in":False,"current_user":None,"auth_mode":"login","users":{},"user_data":{},
    "questions":[],"test_questions":[],"has_generated":False,"has_test_generated":False,
    "quiz_key":0,"pdf_text":"","pdf_filename":"","pdf_size":0,"pdf_hash":"",
    "questions_pdf_hash":"","user_answers":{},"test_submitted":False,
    "current_page":"Home","selected_difficulty":None,"chunks":[],"test_started_at":None,
    "generation_done":False,"bookmarks":[],"wrong_answers":[],"score_history":[],
    "detected_difficulty":"Medium","topics":[],"selected_topics":[],"q_type":"MCQ",
    "fc_idx":0,"fc_filter":"All","timed_mode":False,"per_q_time":45,
    "preview_question":None,"show_preview":False,"groq_key_input":"",
}
for k,v in _D.items():
    if k not in st.session_state: st.session_state[k] = v

S  = lambda t: html_module.escape(str(t))
def go(p): st.session_state.current_page=p; st.rerun()
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()
def get_key(): return GROQ_API_KEY or st.session_state.get("groq_key_input","")

# (logout handled inline by checkbox after navbar — no top-level handler needed)

def do_login(u,pw):
    users=st.session_state.users
    if u not in users: return False,"User not found."
    if users[u]["pw"]!=hash_pw(pw): return False,"Incorrect password."
    st.session_state.logged_in=True; st.session_state.current_user=u
    ud=st.session_state.user_data.get(u,{})
    st.session_state.score_history=ud.get("score_history",[])
    st.session_state.bookmarks=ud.get("bookmarks",[])
    st.session_state.wrong_answers=ud.get("wrong_answers",[])
    return True,"ok"

def do_signup(u,pw,name=""):
    if len(u)<3: return False,"Username needs at least 3 characters."
    if len(pw)<6: return False,"Password needs at least 6 characters."
    if u in st.session_state.users: return False,"Username already taken."
    st.session_state.users[u]={"pw":hash_pw(pw),"name":name or u,
        "created":datetime.datetime.now().strftime("%Y-%m-%d")}
    return do_login(u,pw)

def persist():
    u=st.session_state.current_user
    if not u or u=="__guest__": return
    st.session_state.user_data[u]={
        "score_history":st.session_state.score_history,
        "bookmarks":st.session_state.bookmarks,
        "wrong_answers":st.session_state.wrong_answers,
    }

def get_pdf_text(f):
    try:
        reader=PyPDF2.PdfReader(f)
        text="".join(p.extract_text() or "" for p in reader.pages)
        if len(text.strip())<100 and OCR_AVAILABLE:
            f.seek(0)
            try:
                imgs=convert_from_bytes(f.read())
                return "".join(pytesseract.image_to_string(i)+"\n" for i in imgs)
            except Exception as e: st.error(f"OCR error: {e}"); return text
        return text
    except Exception as e: st.error(f"PDF error: {e}"); return ""

def calc_max_q(wc):
    if wc<500: return 10
    if wc<1000: return 25
    if wc<2000: return 50
    if wc<5000: return 100
    return 150

def detect_diff(text):
    w=text.split()
    if not w: return "Medium"
    avg=sum(len(x) for x in w)/len(w)
    s=[x for x in re.split(r"[.!?]+",text) if x.strip()]
    cpx=sum(1 for x in w if len(x)>9)/len(w)*100
    sc=avg*1.5+(len(w)/max(len(s),1))*0.15+cpx*0.4
    return "Easy" if sc<12 else "Hard" if sc>20 else "Medium"

def extract_topics(text):
    topics=[]
    for line in text.split("\n"):
        line=line.strip()
        if not (3<len(line)<90): continue
        if re.match(r"^(chapter|section|unit|topic|part|module)\s+\d+",line,re.I): topics.append(line)
        elif re.match(r"^\d+[\.\)]\s+[A-Z]",line): topics.append(line)
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

def call_groq(prompt):
    k=get_key()
    if not k: raise ValueError("No Groq API key.")
    if not GROQ_AVAILABLE: raise ImportError("groq package missing.")
    client=Groq(api_key=k)
    r=client.chat.completions.create(model=GROQ_MODEL,
        messages=[{"role":"user","content":prompt}],temperature=0.7,max_tokens=600)
    return r.choices[0].message.content

def keyword_search(query,texts,k=3):
    qw=set(query.lower().split())
    return sorted(texts,key=lambda t:len(qw&set(t.lower().split()))/max(len(qw),1),reverse=True)[:k]

def build_prompt(ctx,q_type,diff="Medium"):
    lvl={"Easy":"BASIC recall","Medium":"COMPREHENSION","Hard":"ANALYSIS"}.get(diff,"COMPREHENSION")
    ctx=ctx[:1400]
    if q_type=="TF":
        return f"Create ONE True/False question ({lvl}).\nOUTPUT ONLY:\nQuestion: [statement]\nAnswer: True\n\nContext:\n{ctx}"
    if q_type=="FIB":
        return f"Create ONE fill-in-blank MCQ ({lvl}). Use ___ for blank.\nOUTPUT ONLY:\nQuestion: [sentence with ___]\nA) [correct]\nB) [wrong]\nC) [wrong]\nD) [wrong]\nCorrect Answer: A\n\nContext:\n{ctx}"
    return f"Generate ONE MCQ ({lvl}).\nOUTPUT ONLY:\nQuestion: [question]\nA) [option]\nB) [option]\nC) [option]\nD) [option]\nCorrect Answer: [A/B/C/D]\n\nContext:\n{ctx}"

def clean_opt(opt):
    raw=opt.strip(); txt=raw[2:].strip() if len(raw)>2 else raw
    for s in ("Context:","Correct Answer:","Question:"):
        if s.lower() in txt.lower(): txt=txt[:txt.lower().index(s.lower())].strip()
    return txt.splitlines()[0].strip() if txt else "--"

def parse_mcq(raw):
    q=""; opts={}; c="A"
    for line in raw.splitlines():
        line=line.strip()
        if line.lower().startswith("question:"): q=line[line.index(":")+1:].strip()
        elif re.match(r"^[A-D]\)",line):
            lt=line[0]; txt=line[2:].strip()
            for s in ("Context:","Correct Answer:","Question:"):
                if s.lower() in txt.lower(): txt=txt[:txt.lower().index(s.lower())].strip()
            if txt: opts[lt]=f"{lt}) {txt.splitlines()[0].strip()}"
        elif re.match(r"^Correct Answer\s*:",line,re.I):
            m=re.search(r"[A-D]",line)
            if m: c=m.group(0)
    return q or "Parsing error",[opts.get(l,f"{l}) --") for l in "ABCD"],c

def parse_tf(raw):
    q=""; c="True"
    for line in raw.splitlines():
        line=line.strip()
        if line.lower().startswith("question:"): q=line[line.index(":")+1:].strip()
        elif re.match(r"^answer\s*:",line,re.I): c="True" if "true" in line.lower() else "False"
    return q or "Parsing error",["A) True","B) False"],"A" if c=="True" else "B"

def parse_fib(raw):
    q=""; opts={}; c="A"
    for line in raw.splitlines():
        line=line.strip()
        if line.lower().startswith("question:"): q=line[line.index(":")+1:].strip()
        elif re.match(r"^[A-D]\)",line):
            lt=line[0]; txt=line[2:].strip().splitlines()[0].strip()
            if txt: opts[lt]=f"{lt}) {txt}"
        elif re.match(r"^correct\s*answer\s*:",line,re.I):
            m=re.search(r"[A-D]",line)
            if m: c=m.group(0)
    return q or "Parsing error",[opts.get(l,f"{l}) --") for l in "ABCD"],c

def llm_gen(ctx,q_type,diff="Medium"):
    raw=call_groq(build_prompt(ctx,q_type,diff))
    if q_type=="TF": q,opts,c=parse_tf(raw)
    elif q_type=="FIB": q,opts,c=parse_fib(raw)
    else: q,opts,c=parse_mcq(raw)
    return {"question":q,"options":opts,"correct":c,"context":ctx[:300],"type":q_type,"difficulty":diff}

def save_score(diff,correct,total,pdf_name):
    pct=round(correct/max(total,1)*100,1)
    st.session_state.score_history.append({
        "date":datetime.datetime.now().strftime("%b %d %H:%M"),
        "diff":diff,"score":correct,"total":total,"pct":pct,"pdf":pdf_name})
    st.session_state.wrong_answers=[
        q for i,q in enumerate(st.session_state.test_questions)
        if st.session_state.user_answers.get(i)!=q["correct"]]
    persist()

def export_html(questions,title):
    rows=""
    for i,q in enumerate(questions):
        opts="".join(
            f'<li style="padding:.5rem 1rem;border-radius:6px;margin:.3rem 0;'
            f'background:{"#fff7ed" if o[0]==q["correct"] else "#fafafa"};'
            f'border-left:3px solid {"#e84c1e" if o[0]==q["correct"] else "transparent"};'
            f'color:{"#e84c1e" if o[0]==q["correct"] else "#555"};font-size:.9rem;">'
            f'{S(o)}</li>' for o in q["options"] if o.strip())
        rows+=(f'<div style="margin-bottom:1.5rem;padding:1.5rem;border:1px solid #e5e7eb;'
               f'border-radius:12px;background:#fff;">'
               f'<div style="font-size:.65rem;font-weight:700;text-transform:uppercase;'
               f'color:#9ca3af;margin-bottom:.5rem;">Q{i+1} · {q.get("type","MCQ")}</div>'
               f'<div style="font-size:1rem;font-weight:700;color:#111;margin-bottom:.875rem;">'
               f'{S(q["question"])}</div>'
               f'<ul style="list-style:none;padding:0;margin:0;">{opts}</ul></div>')
    return (f'<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{S(title)}</title>'
            f'<style>body{{font-family:system-ui;max-width:800px;margin:3rem auto;'
            f'padding:0 1.5rem;background:#f8f5f0;}}</style></head>'
            f'<body><h1>{S(title)}</h1>'
            f'<p style="color:#9ca3af;">{datetime.datetime.now().strftime("%B %d, %Y")} · QuizGenius AI</p>'
            f'{rows}</body></html>')

def render_score_chart(sh):
    if len(sh)<2: return
    recent=sh[-12:]
    labels=json.dumps([s["date"] for s in recent])
    data=json.dumps([s["pct"] for s in recent])
    colors=json.dumps(["#22c55e" if s["pct"]>=80 else "#f59e0b" if s["pct"]>=60 else "#ef4444" for s in recent])
    components.html(f"""<!DOCTYPE html><html><head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
    <style>*{{margin:0;padding:0;box-sizing:border-box;}}body{{background:#fff;padding:16px;font-family:system-ui;}}</style>
    </head><body><canvas id="c" height="120"></canvas>
    <script>new Chart(document.getElementById("c").getContext("2d"),{{
      type:"line",data:{{labels:{labels},datasets:[{{label:"Score %",data:{data},
        borderColor:"#e84c1e",backgroundColor:"rgba(232,76,30,.06)",borderWidth:2,
        pointBackgroundColor:{colors},pointBorderColor:"#fff",pointBorderWidth:2,
        pointRadius:5,fill:true,tension:0.4}}]}},
      options:{{responsive:true,plugins:{{legend:{{display:false}},
        tooltip:{{backgroundColor:"#111",padding:10,cornerRadius:8,titleColor:"#fff",bodyColor:"#ccc"}}}},
        scales:{{y:{{min:0,max:100,grid:{{color:"rgba(0,0,0,.06)"}},
          ticks:{{callback:v=>v+"%",color:"#9ca3af",font:{{size:10}}}}}},
          x:{{grid:{{display:false}},ticks:{{color:"#9ca3af",font:{{size:9}},maxRotation:30}}}}}}}}
    }});</script></body></html>""",height=190,scrolling=False)

def render_flashcard(question,answer,num,total,bookmarked=False):
    pct=int(num/total*100); bm="#e84c1e" if bookmarked else "#d1d5db"
    components.html(f"""<!DOCTYPE html><html><head><style>
    *{{margin:0;padding:0;box-sizing:border-box;}}
    body{{font-family:"Segoe UI",system-ui,sans-serif;background:transparent;overflow:hidden;}}
    .bar{{width:100%;height:4px;background:#f3f4f6;border-radius:999px;margin-bottom:14px;}}
    .fill{{height:100%;background:#e84c1e;border-radius:999px;width:{pct}%;}}
    .meta{{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;}}
    .num{{font-size:.6rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
      color:#6b7280;background:#f9fafb;padding:3px 10px;border-radius:6px;border:1px solid #e5e7eb;}}
    .bm{{font-size:1.2rem;color:{bm};}}
    .scene{{perspective:1200px;width:100%;height:220px;cursor:pointer;}}
    .card{{width:100%;height:100%;position:relative;transform-style:preserve-3d;
      transition:transform .6s cubic-bezier(.175,.885,.32,1.1);}}
    .card.flip{{transform:rotateY(180deg);}}
    .face{{position:absolute;inset:0;backface-visibility:hidden;border-radius:16px;
      display:flex;flex-direction:column;align-items:center;justify-content:center;padding:2rem;}}
    .front{{background:#fff;border:1.5px solid #e5e7eb;box-shadow:0 4px 24px rgba(0,0,0,.08);}}
    .back{{background:linear-gradient(135deg,#fff7ed,#fef3c7);border:1.5px solid #fed7aa;
      transform:rotateY(180deg);box-shadow:0 8px 32px rgba(232,76,30,.12);}}
    .qlbl{{font-size:.55rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
      color:#9ca3af;margin-bottom:.75rem;}}
    .qtext{{font-size:.95rem;font-weight:700;color:#111;line-height:1.6;text-align:center;max-width:480px;}}
    .abadge{{background:#fff7ed;border:1px solid #fed7aa;color:#e84c1e;font-size:.58rem;
      font-weight:700;padding:3px 12px;border-radius:999px;margin-bottom:.875rem;
      text-transform:uppercase;letter-spacing:.08em;}}
    .atext{{font-size:.95rem;font-weight:700;color:#e84c1e;line-height:1.6;text-align:center;max-width:480px;}}
    .hint{{position:absolute;bottom:14px;font-size:.6rem;color:#9ca3af;font-style:italic;}}
    </style></head><body>
    <div class="bar"><div class="fill"></div></div>
    <div class="meta"><span class="num">Card {num} / {total}</span>
      <span class="bm">{"★" if bookmarked else "☆"}</span></div>
    <div class="scene" onclick="this.querySelector(".card").classList.toggle("flip")">
      <div class="card">
        <div class="face front">
          <div class="qlbl">Question</div>
          <div class="qtext">{S(question)}</div>
          <div class="hint">click to flip →</div>
        </div>
        <div class="face back">
          <div class="abadge">Answer</div>
          <div class="atext">{S(answer)}</div>
        </div>
      </div>
    </div></body></html>""",height=285,scrolling=False)

def render_result(pct,correct,total):
    verdict="Outstanding! 🎉" if pct>=80 else "Good Work! 👍" if pct>=60 else "Keep Practicing 📚"
    sub=("You mastered this material." if pct>=80 else
         "Almost there — a little more review." if pct>=60 else
         "Keep reviewing flashcards to improve.")
    col="#22c55e" if pct>=80 else "#f59e0b" if pct>=60 else "#ef4444"
    confetti="true" if pct>=80 else "false"
    components.html(f"""<!DOCTYPE html><html><head>
    <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.2/dist/confetti.browser.min.js"></script>
    <style>*{{margin:0;padding:0;box-sizing:border-box;}}
    body{{font-family:"Segoe UI",system-ui,sans-serif;background:#fff;
      border:1.5px solid #e5e7eb;border-radius:20px;overflow:hidden;}}
    .hero{{padding:2.5rem 2rem;text-align:center;}}
    .ring{{width:100px;height:100px;border-radius:50%;border:3px solid {col};
      background:{col}18;display:flex;align-items:center;justify-content:center;
      margin:0 auto 1.25rem;box-shadow:0 0 0 8px {col}11;}}
    .pct{{font-size:2rem;font-weight:800;color:{col};line-height:1;}}
    .verdict{{font-size:1.375rem;font-weight:800;color:#111;margin-bottom:.375rem;}}
    .sub{{color:#6b7280;font-size:.875rem;line-height:1.75;max-width:340px;margin:0 auto 1.5rem;}}
    .chips{{display:flex;gap:.625rem;justify-content:center;flex-wrap:wrap;}}
    .chip{{padding:.375rem 1rem;border-radius:999px;font-size:.75rem;font-weight:700;}}
    .ok{{background:#f0fdf4;color:#22c55e;border:1px solid #bbf7d0;}}
    .ng{{background:#fef2f2;color:#ef4444;border:1px solid #fecaca;}}
    .info{{background:#f9fafb;color:#6b7280;border:1px solid #e5e7eb;}}
    </style></head><body><div class="hero">
    <div class="ring"><div class="pct"><span id="ctr">0</span>%</div></div>
    <div class="verdict">{S(verdict)}</div>
    <p class="sub">{S(sub)}</p>
    <div class="chips">
      <span class="chip ok">✓ {correct} Correct</span>
      <span class="chip ng">✗ {total-correct} Wrong</span>
      <span class="chip info">↗ {total} Total</span>
    </div></div>
    <script>
    var t={pct:.0f},e=document.getElementById("ctr"),c=0,s=Math.max(1,Math.ceil(t/60));
    setInterval(function(){{c=Math.min(c+s,t);e.textContent=Math.round(c);}},16);
    if({confetti}){{setTimeout(function(){{confetti({{particleCount:120,spread:80,
      origin:{{y:0.4}},colors:["#e84c1e","#f59e0b","#22c55e"]}});}},200);}}
    </script></body></html>""",height=310,scrolling=False)

# ════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
#MainMenu,footer,header,.stDeployButton{visibility:hidden!important;display:none!important;}
[data-testid="collapsedControl"],section[data-testid="stSidebar"]{display:none!important;}
.stApp>header{display:none!important;height:0!important;}
html,body,.stApp,.main,.block-container,
[data-testid="stAppViewBlockContainer"],[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],section.main>div:first-child,
div[class*="block-container"]{padding-top:0!important;margin-top:0!important;}
.block-container{max-width:100%!important;padding-bottom:0!important;}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:#f3f4f6;}
::-webkit-scrollbar-thumb{background:#d1d5db;border-radius:999px;}
:root{
  --bg:#f8f5f0;--wh:#fff;--bd:#e5e7eb;--bd2:#d1d5db;
  --tx:#111;--tx2:#374151;--tx3:#6b7280;--tx4:#9ca3af;
  --or:#e84c1e;--ol:#fff7ed;--om:#fed7aa;--od:#c2410c;
  --gr:#22c55e;--re:#ef4444;--ye:#f59e0b;
  --sh:0 1px 4px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.06);
  --sh2:0 4px 24px rgba(0,0,0,.1);--sh3:0 8px 40px rgba(0,0,0,.12);
  --r:12px;--r2:8px;}
body,.stApp{font-family:'Inter',system-ui,sans-serif!important;
  background:var(--bg)!important;color:var(--tx)!important;}
/* BUTTONS */
.stButton>button{font-family:'Inter',sans-serif!important;font-weight:600!important;
  border-radius:var(--r2)!important;transition:all .18s!important;border:none!important;}
.stButton>button[kind="primary"]{background:#e84c1e!important;color:#fff!important;
  box-shadow:0 2px 8px rgba(232,76,30,.3)!important;}
.stButton>button[kind="primary"]:hover{background:#c2410c!important;
  transform:translateY(-1px)!important;box-shadow:0 4px 16px rgba(232,76,30,.4)!important;}
.stButton>button[kind="secondary"]{background:#fff!important;color:var(--tx2)!important;
  border:1.5px solid var(--bd)!important;}
.stButton>button[kind="secondary"]:hover{border-color:var(--or)!important;
  color:var(--or)!important;transform:translateY(-1px)!important;}
.stButton>button:disabled{opacity:.4!important;transform:none!important;}
.stDownloadButton>button{font-family:'Inter',sans-serif!important;font-weight:600!important;
  background:#e84c1e!important;color:#fff!important;border:none!important;
  border-radius:var(--r2)!important;}
.stDownloadButton>button:hover{background:#c2410c!important;transform:translateY(-1px)!important;}
/* INPUTS */
.stTextInput input{font-family:'Inter',sans-serif!important;border-radius:var(--r2)!important;
  border:1.5px solid var(--bd)!important;background:#fff!important;
  padding:.625rem .875rem!important;font-size:.9rem!important;
  transition:border-color .18s,box-shadow .18s!important;}
.stTextInput input:focus{border-color:var(--or)!important;
  box-shadow:0 0 0 3px rgba(232,76,30,.1)!important;outline:none!important;}
.stTextInput input::placeholder{color:var(--tx4)!important;}
.stTextInput>label{font-weight:600!important;font-size:.8rem!important;color:var(--tx3)!important;}
/* RADIO */
div[data-testid="stRadio"]>div{gap:.375rem!important;flex-direction:column!important;display:flex!important;}
div[data-testid="stRadio"]>div>label{background:#fff!important;padding:.8rem 1.125rem!important;
  border-radius:var(--r2)!important;border:1.5px solid var(--bd)!important;
  font-size:.875rem!important;font-weight:500!important;width:100%!important;
  margin:0!important;transition:all .15s!important;cursor:pointer!important;
  color:#374151!important;}
div[data-testid="stRadio"]>div>label p,
div[data-testid="stRadio"]>div>label div,
div[data-testid="stRadio"]>div>label span{color:#374151!important;}
div[data-testid="stRadio"]>div>label:hover{border-color:var(--or)!important;
  background:var(--ol)!important;color:var(--or)!important;}
div[data-testid="stRadio"]>div>label:hover p,
div[data-testid="stRadio"]>div>label:hover span{color:var(--or)!important;}
div[data-testid="stRadio"]>div>label:has(input:checked){border-color:var(--or)!important;
  background:var(--ol)!important;color:var(--or)!important;font-weight:700!important;
  box-shadow:0 0 0 3px rgba(232,76,30,.08)!important;}
div[data-testid="stRadio"]>div>label:has(input:checked) p,
div[data-testid="stRadio"]>div>label:has(input:checked) span{color:var(--or)!important;}
div[data-testid="stRadio"]>div>label>div:first-child{display:none!important;}
/* SMOOTH PAGE FADE TRANSITION — 0.1s barely noticeable */
@keyframes fadeIn{from{opacity:0;}to{opacity:1;}}
[data-testid="stMainBlockContainer"]>div>div{animation:fadeIn .1s ease both;}

/* GLOBAL TEXT VISIBILITY — fix any invisible text across the app */
.stApp *{color:inherit;}
p,span,div,label,li,td,th,h1,h2,h3,h4,h5,h6{color:var(--tx2);}
.stMarkdown p,.stMarkdown span{color:var(--tx2)!important;}
/* Slider */
[data-testid="stSlider"] label{color:var(--tx2)!important;font-weight:600!important;}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"]{color:var(--tx4)!important;}
/* Caption / small text */
.stCaption,.stCaption p{color:var(--tx3)!important;font-size:.75rem!important;}
/* Expander */
[data-testid="stExpander"] summary span{color:var(--tx)!important;font-weight:600!important;}
[data-testid="stExpander"] summary:hover span{color:var(--or)!important;}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p{color:var(--tx2)!important;}
/* Multiselect */
[data-testid="stMultiSelect"] label{color:var(--tx2)!important;font-weight:600!important;}
[data-testid="stMultiSelect"] [data-baseweb="tag"]{background:var(--ol)!important;
  border:1px solid var(--om)!important;}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span{color:var(--or)!important;}
/* Checkbox */
[data-testid="stCheckbox"] label p{color:var(--tx2)!important;}
/* Progress bar text */
[data-testid="stProgress"] p{color:var(--tx3)!important;}
/* Spinner text */
[data-testid="stSpinner"] p{color:var(--tx2)!important;}
/* Code blocks */
.stCodeBlock code{color:#d63031!important;}

/* MISC */
.stSelectbox>div>div{background:#fff!important;border:1.5px solid var(--bd)!important;
  border-radius:var(--r2)!important;color:var(--tx)!important;}
.stSelectbox label{color:var(--tx2)!important;font-size:.8rem!important;font-weight:600!important;}
/* FILE UPLOADER — fix dark bg, ensure text visibility */
[data-testid="stFileUploader"]{background:#fff!important;border:2px dashed var(--bd2)!important;
  border-radius:var(--r)!important;padding:.5rem!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--or)!important;}
[data-testid="stFileUploaderDropzone"]{background:#fff!important;}
[data-testid="stFileUploaderDropzone"] *{color:var(--tx2)!important;}
[data-testid="stFileUploaderDropzone"] small{color:var(--tx3)!important;}
[data-testid="stFileUploader"] button{background:var(--or)!important;color:#fff!important;
  border:none!important;border-radius:var(--r2)!important;font-weight:600!important;}
section[data-testid="stFileUploadDropzone"]{background:#fff!important;}
section[data-testid="stFileUploadDropzone"] div,
section[data-testid="stFileUploadDropzone"] span,
section[data-testid="stFileUploadDropzone"] p{color:var(--tx2)!important;}
[data-testid="stForm"]{background:#fff!important;border:1.5px solid var(--bd)!important;
  border-radius:var(--r)!important;box-shadow:var(--sh)!important;padding:1.5rem!important;}
.stSuccess>div{background:#f0fdf4!important;border:1px solid #bbf7d0!important;
  border-radius:var(--r2)!important;color:#166534!important;}
.stError>div{background:#fef2f2!important;border:1px solid #fecaca!important;
  border-radius:var(--r2)!important;color:#991b1b!important;}
.stWarning>div{background:#fffbeb!important;border:1px solid #fde68a!important;
  border-radius:var(--r2)!important;color:#92400e!important;}
.stInfo>div{background:#eff6ff!important;border:1px solid #bfdbfe!important;
  border-radius:var(--r2)!important;color:#1e40af!important;}
.stProgress>div>div{background:#e84c1e!important;border-radius:999px!important;}
.stProgress>div{background:#f3f4f6!important;border-radius:999px!important;}
[data-testid="stExpander"]{background:#fff!important;border:1.5px solid var(--bd)!important;
  border-radius:var(--r)!important;}
/* ── NAVBAR ── */
[data-testid="stAppViewBlockContainer"]>div:first-child{
  position:sticky!important;top:0!important;z-index:9999!important;
  background:#fff!important;border-bottom:1px solid var(--bd)!important;
  box-shadow:0 1px 8px rgba(0,0,0,.06)!important;
  height:64px!important;overflow:visible!important;}
/* Force EVERY intermediate Streamlit wrapper inside navbar to be a flex row centered at 64px */
[data-testid="stAppViewBlockContainer"]>div:first-child>div,
[data-testid="stAppViewBlockContainer"]>div:first-child>div>div,
[data-testid="stAppViewBlockContainer"]>div:first-child>div>div>div{
  height:64px!important;padding:0!important;margin:0!important;}
[data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="stHorizontalBlock"]{
  display:flex!important;flex-direction:row!important;flex-wrap:nowrap!important;
  align-items:center!important;justify-content:flex-start!important;
  padding:0 1.5rem!important;height:64px!important;min-height:64px!important;
  gap:0!important;margin:0 auto!important;max-width:1400px!important;
  overflow:visible!important;}
/* Each column: full height flex, vertically centred */
[data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="column"]{
  display:flex!important;flex-direction:row!important;align-items:center!important;
  height:64px!important;min-height:64px!important;
  padding:0!important;margin:0!important;flex-shrink:0!important;overflow:visible!important;}
/* Every Streamlit wrapper INSIDE columns: propagate height + centre */
[data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="column"]>div,
[data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="column"]>div>div,
[data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="stElementContainer"],
[data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="stMarkdownContainer"],
[data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="stButton"]{
  display:flex!important;align-items:center!important;justify-content:center!important;
  height:64px!important;width:100%!important;padding:0!important;margin:0!important;}
/* Nav link buttons — transparent tab underline style */
[data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="stHorizontalBlock"]
  [data-testid="column"]:not(:first-child):not(:last-child) button{
  font-family:'Inter',sans-serif!important;font-size:.82rem!important;font-weight:500!important;
  background:transparent!important;color:var(--tx3)!important;border:none!important;
  border-radius:0!important;box-shadow:none!important;height:64px!important;
  padding:0 10px!important;white-space:nowrap!important;width:auto!important;
  border-bottom:2px solid transparent!important;transition:color .15s,border-color .15s!important;}
[data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="stHorizontalBlock"]
  [data-testid="column"]:not(:first-child):not(:last-child) button:hover{
  color:var(--tx)!important;background:transparent!important;transform:none!important;
  box-shadow:none!important;border-bottom:2px solid var(--bd2)!important;}
[data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="stHorizontalBlock"]
  [data-testid="column"]:not(:first-child):not(:last-child) button[kind="primary"]{
  color:var(--or)!important;font-weight:700!important;
  border-bottom:2px solid var(--or)!important;background:transparent!important;box-shadow:none!important;}
[data-testid="stAppViewBlockContainer"]>div:first-child [data-testid="stHorizontalBlock"]
  [data-testid="column"]:not(:first-child):not(:last-child) button[kind="primary"]:hover{
  transform:none!important;background:transparent!important;box-shadow:none!important;}
/* Hide logout block — display:none kills layout space; JS .click() still works on hidden elements */
[data-testid="stAppViewBlockContainer"]>div:nth-child(2){
  display:none!important;}
/* Generate page columns — align tops */
.gl>[data-testid="column"]{align-self:start!important;}
/* PDF banner Change button — vertically centered */
[data-testid="stButton"]>button{vertical-align:middle!important;}
/* Kill ALL Streamlit-injected top spacing */
[data-testid="stAppViewBlockContainer"]>div:not(:first-child){
  margin-top:0!important;padding-top:0!important;}
[data-testid="stMainBlockContainer"]{padding-top:0!important;margin-top:0!important;}
section[data-testid="stMain"]>div{padding-top:0!important;}
/* LAYOUTS */
.pw{max-width:900px;margin:0 auto;padding:.5rem 1.5rem 5rem;}
.fw{max-width:1280px;margin:0 auto;padding:0 2rem 5rem;}
.badge{display:inline-flex;align-items:center;font-size:.6rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.07em;padding:2px 9px;border-radius:999px;}
.b-or{background:var(--ol);color:var(--or);border:1px solid var(--om);}
.b-gr{background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0;}
.b-re{background:#fef2f2;color:var(--re);border:1px solid #fecaca;}
.b-ye{background:#fffbeb;color:#d97706;border:1px solid #fde68a;}
.b-gy{background:#f9fafb;color:var(--tx3);border:1px solid var(--bd);}
/* STUDY CARDS */
.qc{background:#fff;border:1.5px solid var(--bd);border-radius:var(--r);
  overflow:hidden;margin-bottom:.875rem;transition:box-shadow .18s;}
.qc:hover{box-shadow:var(--sh2);}
.qc-head{padding:1rem 1.25rem .625rem;display:flex;align-items:center;
  justify-content:space-between;border-bottom:1px solid #f9fafb;}
.qc-num{font-size:.6rem;font-weight:700;text-transform:uppercase;color:var(--tx4);
  background:#f9fafb;border:1px solid var(--bd);border-radius:5px;padding:2px 8px;}
.qc-q{padding:.75rem 1.25rem .875rem;font-size:.925rem;font-weight:600;line-height:1.65;}
.oc{display:flex;align-items:center;gap:.75rem;padding:.625rem 1rem;margin:.2rem 1.25rem;border-radius:var(--r2);}
.oc-ok{border:1.5px solid #bbf7d0;background:#f0fdf4;}
.oc-no{border:1.5px solid var(--bd);background:transparent;}
.ol-ok{width:24px;height:24px;border-radius:50%;background:var(--gr);color:#fff;
  display:flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:800;flex-shrink:0;}
.ol-no{width:24px;height:24px;border-radius:50%;background:#f3f4f6;color:var(--tx3);
  display:flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:600;
  flex-shrink:0;border:1px solid var(--bd);}
.ot-ok{font-size:.875rem;font-weight:600;color:#16a34a;flex:1;}
.ot-no{font-size:.875rem;color:var(--tx3);flex:1;}
.qc-foot{border-top:1px solid #f3f4f6;padding:.75rem 1.25rem;
  display:flex;align-items:center;gap:.75rem;background:#fafafa;}
/* PROGRESS BAR */
.pb-wrap{background:#fff;border:1.5px solid var(--bd);border-radius:var(--r);
  padding:1.25rem 1.375rem;margin-top:1rem;}
.pb-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:.625rem;}
.pb-lbl{font-size:.78rem;font-weight:700;color:var(--or);}
.pb-pct{font-size:.78rem;font-weight:700;color:var(--tx3);}
.pb-bar{width:100%;height:6px;background:#f3f4f6;border-radius:999px;overflow:hidden;}
.pb-fill{height:100%;background:#e84c1e;border-radius:999px;transition:width .4s;}
.pb-note{text-align:center;font-size:.68rem;color:var(--tx4);margin-top:.75rem;font-style:italic;}
/* PDF BANNER */
.pdf-bn{background:#fff;border:1.5px solid var(--bd);border-radius:var(--r);
  padding:1rem 1.25rem;display:flex;align-items:center;gap:.875rem;
  margin-bottom:1.25rem;box-shadow:var(--sh);}
.pdf-ic{width:44px;height:44px;border-radius:10px;background:var(--ol);
  display:flex;align-items:center;justify-content:center;
  font-size:1.25rem;flex-shrink:0;border:1px solid var(--om);}
/* TEST CARDS */
.tc{background:#fff;border:1.5px solid var(--bd);border-radius:var(--r);
  padding:1.375rem;margin-bottom:.75rem;}
.tc-top{display:flex;align-items:center;gap:.5rem;margin-bottom:.75rem;flex-wrap:wrap;}
.tc-num{background:var(--or);color:#fff;padding:2px 10px;border-radius:6px;font-size:.65rem;font-weight:800;}
.tc-q{font-size:.925rem;font-weight:700;line-height:1.6;}
.tpb{background:#fff;border:1.5px solid var(--bd);border-radius:var(--r);
  padding:.875rem 1.25rem;margin-bottom:1.25rem;display:flex;align-items:center;gap:1rem;}
.tpb-bar{flex:1;height:5px;background:#f3f4f6;border-radius:999px;overflow:hidden;}
.tpb-fill{height:100%;background:#e84c1e;border-radius:999px;transition:width .4s;}
.tpb-timer{color:var(--or);font-size:.85rem;font-weight:700;white-space:nowrap;}
/* REVIEW */
.rv{display:flex;align-items:flex-start;gap:.75rem;padding:.75rem 1rem;
  border-radius:var(--r2);background:#fff;margin-bottom:.375rem;border:1.5px solid var(--bd);}
.rv-c{border-left:3px solid var(--gr);background:#f0fdf4;}
.rv-w{border-left:3px solid var(--re);background:#fef2f2;}
.rv-q{font-size:.85rem;font-weight:700;margin-bottom:.2rem;}
.rv-a{font-size:.78rem;color:var(--tx3);}
/* DASHBOARD */
.sg{display:grid;grid-template-columns:repeat(4,1fr);gap:.875rem;margin-bottom:1.25rem;}
.sc{background:#fff;border:1.5px solid var(--bd);border-radius:var(--r);
  padding:1.375rem;text-align:center;box-shadow:var(--sh);transition:all .18s;}
.sc:hover{box-shadow:var(--sh2);transform:translateY(-2px);}
.sc-ico{font-size:1.5rem;display:block;margin-bottom:.375rem;}
.sc-val{font-size:1.75rem;font-weight:800;color:var(--or);letter-spacing:-.03em;line-height:1;}
.sc-lbl{font-size:.6rem;font-weight:600;color:var(--tx4);text-transform:uppercase;
  letter-spacing:.1em;margin-top:.3rem;}
.shr{display:flex;align-items:center;gap:.75rem;background:#fff;border:1.5px solid var(--bd);
  border-radius:var(--r2);padding:.75rem 1.125rem;margin-bottom:.375rem;
  transition:box-shadow .15s;box-shadow:var(--sh);}
.shr:hover{box-shadow:var(--sh2);}
.shr-d{font-size:.58rem;font-weight:700;text-transform:uppercase;
  padding:2px 9px;border-radius:999px;white-space:nowrap;flex-shrink:0;}
.shr-s{font-size:1rem;font-weight:800;min-width:48px;text-align:center;}
.shr-m{flex:1;min-width:0;}
.shr-pdf{font-weight:600;font-size:.8rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.shr-dt{color:var(--tx4);font-size:.68rem;margin-top:.1rem;}
.shr-pb{width:72px;height:4px;background:#f3f4f6;border-radius:999px;overflow:hidden;flex-shrink:0;}
.shr-pf{height:100%;border-radius:999px;}
.shr-pct{font-size:.875rem;font-weight:700;min-width:40px;text-align:right;}
.ds-t{font-size:.95rem;font-weight:700;margin:1.75rem 0 .875rem;
  display:flex;align-items:center;gap:.5rem;}
.ds-t::after{content:'';flex:1;height:1px;background:var(--bd);}
.ch-w{background:#fff;border:1.5px solid var(--bd);border-radius:var(--r);
  padding:1.25rem;margin-bottom:1rem;box-shadow:var(--sh);}
/* HOME */
.hs{background:var(--wh);border-bottom:1px solid var(--bd);padding:5rem 2rem 4rem;}
.hi{max-width:1280px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:4rem;align-items:center;}
.htag{display:inline-flex;align-items:center;gap:.375rem;background:var(--ol);
  border:1px solid var(--om);color:var(--or);font-size:.62rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.08em;padding:.3rem .875rem;
  border-radius:999px;margin-bottom:1.25rem;}
.hh1{font-size:3.25rem;font-weight:800;letter-spacing:-.04em;line-height:1.05;margin-bottom:1rem;}
.hh1 .ac{color:var(--or);}
.hp{font-size:.9rem;color:var(--tx3);line-height:1.85;margin-bottom:1.75rem;}
.pr{display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:2rem;}
.pi{background:#f9fafb;border:1.5px solid var(--bd);border-radius:var(--r2);
  padding:.4rem .875rem;font-size:.72rem;font-weight:600;color:var(--tx3);
  display:flex;align-items:center;gap:.35rem;transition:all .15s;}
.pi:hover{border-color:var(--or);color:var(--or);background:var(--ol);}
.sb{background:var(--tx);padding:2.25rem 0;}
.si{max-width:1280px;margin:0 auto;display:grid;grid-template-columns:repeat(4,1fr);}
.si-c{text-align:center;padding:1rem;border-right:1px solid #333;}
.si-c:last-child{border-right:none;}
.si-n{font-size:2rem;font-weight:800;color:var(--or);display:block;letter-spacing:-.04em;}
.si-l{font-size:.6rem;color:#9ca3af;font-weight:600;text-transform:uppercase;
  letter-spacing:.12em;margin-top:.25rem;display:block;}
.sec{padding:5rem 2rem;}
.sec-alt{padding:5rem 2rem;background:#fff;border-top:1px solid var(--bd);border-bottom:1px solid var(--bd);}
.sec-in{max-width:1280px;margin:0 auto;}
.sec-ey{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;
  color:var(--or);margin-bottom:.25rem;}
.sec-ti{font-size:1.875rem;font-weight:800;letter-spacing:-.03em;line-height:1.15;margin-bottom:.5rem;}
.sec-su{font-size:.9rem;color:var(--tx3);line-height:1.75;max-width:460px;}
.hwg{display:grid;grid-template-columns:repeat(3,1fr);gap:1.25rem;margin-top:2.5rem;}
.hwc{background:#fff;border:1.5px solid var(--bd);border-radius:16px;padding:1.875rem;
  transition:all .25s;position:relative;overflow:hidden;box-shadow:var(--sh);}
.hwc:hover{transform:translateY(-4px);box-shadow:var(--sh3);border-color:var(--or);}
.hwc::before{content:attr(data-n);position:absolute;top:-8px;right:1rem;
  font-size:5rem;font-weight:800;color:rgba(232,76,30,.06);line-height:1;pointer-events:none;}
.hwi{width:48px;height:48px;border-radius:12px;display:flex;align-items:center;
  justify-content:center;font-size:1.375rem;margin-bottom:1rem;background:var(--ol);border:1px solid var(--om);}
.hwt{font-size:.925rem;font-weight:700;margin-bottom:.375rem;}
.hwp{font-size:.825rem;color:var(--tx3);line-height:1.75;margin:0;}
.fg{display:grid;grid-template-columns:repeat(2,1fr);gap:.875rem;}
.fcc{background:#fff;border:1.5px solid var(--bd);border-radius:var(--r);padding:1.125rem;
  display:flex;align-items:flex-start;gap:.75rem;transition:all .18s;box-shadow:var(--sh);}
.fcc:hover{box-shadow:var(--sh2);border-color:var(--or);transform:translateY(-2px);}
.fci{width:36px;height:36px;border-radius:9px;flex-shrink:0;display:flex;align-items:center;
  justify-content:center;font-size:.95rem;background:var(--ol);border:1px solid var(--om);}
.fct{font-size:.8rem;font-weight:700;margin-bottom:.1rem;}
.fcs{font-size:.7rem;color:var(--tx4);line-height:1.6;}
.dfg{display:grid;grid-template-columns:repeat(3,1fr);gap:1.25rem;margin-top:2.5rem;}
.dfc{background:#fff;border:1.5px solid var(--bd);border-radius:16px;padding:1.875rem;
  transition:all .25s;box-shadow:var(--sh);}
.dfc:hover{transform:translateY(-4px);box-shadow:var(--sh3);}
.dfc.e{border-top:3px solid var(--gr);}
.dfc.m{border-top:3px solid var(--or);}
.dfc.h{border-top:3px solid var(--re);}
.dfn{font-size:1.25rem;font-weight:800;margin:.5rem 0 .375rem;}
.dfd{font-size:.825rem;color:var(--tx3);line-height:1.75;}
.ft{background:var(--tx);padding:3.5rem 2rem 1.5rem;}
.fti{max-width:1280px;margin:0 auto;}
.ft-top{display:grid;grid-template-columns:2fr 1fr 1fr 1.5fr;gap:3rem;margin-bottom:2.5rem;}
.ft-brand{display:flex;align-items:center;gap:.625rem;margin-bottom:.875rem;}
.ft-logo{width:30px;height:30px;border-radius:7px;background:var(--or);
  display:flex;align-items:center;justify-content:center;font-size:.9rem;}
.ft-name{font-size:.875rem;font-weight:700;color:#fff;}
.ft-desc{font-size:.8rem;color:#6b7280;line-height:1.8;}
.ft-hd{font-size:.6rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
  letter-spacing:.1em;margin-bottom:.75rem;}
.ft-lk{display:block;font-size:.8rem;color:#6b7280;margin-bottom:.4rem;cursor:pointer;transition:color .15s;}
.ft-lk:hover{color:var(--or);}
.ft-bot{border-top:1px solid #1f2937;padding-top:1.25rem;display:flex;
  justify-content:space-between;align-items:center;flex-wrap:wrap;
  gap:.875rem;font-size:.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:.07em;}
/* GENERATE */
.gl{display:grid;grid-template-columns:1fr 380px;gap:1.5rem;align-items:start;}
.gp{background:#fff;border:1.5px solid var(--bd);border-radius:var(--r);box-shadow:var(--sh);overflow:hidden;}
.gp-hd{padding:.875rem 1.25rem;border-bottom:1px solid var(--bd);
  display:flex;align-items:center;justify-content:space-between;}
.gp-ti{font-size:.8rem;font-weight:700;}
.gp-em{display:flex;flex-direction:column;align-items:center;
  justify-content:center;padding:3rem 1.5rem;text-align:center;}
.gp-em-ic{width:56px;height:56px;border-radius:50%;background:#f3f4f6;
  display:flex;align-items:center;justify-content:center;font-size:1.5rem;margin-bottom:1rem;}
.gp-em-t{font-size:.85rem;font-weight:600;}
.gp-em-s{font-size:.75rem;color:var(--tx4);margin-top:.25rem;line-height:1.6;}
.cfg{background:#fff;border:1.5px solid var(--bd);border-radius:var(--r);
  padding:1.25rem 1.375rem;margin-bottom:.875rem;box-shadow:var(--sh);}
.cfg-lbl{font-size:.78rem;font-weight:700;margin-bottom:.125rem;}
.cfg-hint{font-size:.7rem;color:var(--tx4);margin-bottom:.75rem;}
.qtg{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;}
.qtc{border:1.5px solid var(--bd);border-radius:var(--r2);padding:.875rem;
  text-align:center;background:#fff;cursor:pointer;transition:all .18s;}
.qtc:hover{border-color:var(--or);background:var(--ol);}
.qtc.sel{border-color:var(--or);background:var(--ol);box-shadow:0 0 0 3px rgba(232,76,30,.1);}
.qtc-ico{font-size:1.375rem;margin-bottom:.375rem;}
.qtc-n{font-size:.75rem;font-weight:700;}
.qtc-d{font-size:.62rem;color:var(--tx4);margin-top:.125rem;}
.po-ok{display:flex;align-items:center;gap:.625rem;padding:.5rem .875rem;
  margin:.25rem 0;border-radius:var(--r2);border:1.5px solid #bbf7d0;background:#f0fdf4;}
.po{display:flex;align-items:center;gap:.625rem;padding:.5rem .875rem;
  margin:.25rem 0;border-radius:var(--r2);border:1.5px solid var(--bd);background:transparent;}
.pl-ok{width:22px;height:22px;border-radius:50%;background:var(--gr);
  color:#fff;display:flex;align-items:center;justify-content:center;
  font-size:.6rem;font-weight:800;flex-shrink:0;}
.pl{width:22px;height:22px;border-radius:50%;background:#f3f4f6;color:var(--tx3);
  display:flex;align-items:center;justify-content:center;font-size:.6rem;font-weight:600;
  flex-shrink:0;border:1px solid var(--bd);}
.pt-ok{font-size:.82rem;font-weight:600;color:#16a34a;flex:1;}
.pt{font-size:.82rem;color:var(--tx3);flex:1;}
/* TEST DIFFICULTY */
.tdg{display:grid;grid-template-columns:repeat(3,1fr);gap:1.125rem;margin-bottom:1.25rem;}
.tdc{background:#fff;border:1.5px solid var(--bd);border-radius:16px;padding:1.75rem;
  text-align:center;box-shadow:var(--sh);transition:all .22s;position:relative;}
.tdc:hover{transform:translateY(-4px);box-shadow:var(--sh3);border-color:var(--or);}
.tdc.feat{border-color:var(--or);box-shadow:0 0 0 3px rgba(232,76,30,.08);}
.tdc-pop{position:absolute;top:.75rem;right:.75rem;background:var(--or);
  color:#fff;font-size:.52rem;font-weight:800;padding:2px 8px;border-radius:999px;}
.tdc-ico{width:52px;height:52px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:1.5rem;margin:0 auto .75rem;}
.tdc-ico.e{background:#f0fdf4;border:1px solid #bbf7d0;}
.tdc-ico.m{background:var(--ol);border:1px solid var(--om);}
.tdc-ico.h{background:#fef2f2;border:1px solid #fecaca;}
.tdc-n{font-size:1rem;font-weight:700;margin-bottom:.25rem;}
.tdc-h{font-size:.75rem;color:var(--tx3);line-height:1.65;}
.aw{max-width:960px;margin:0 auto;padding:3rem 1.5rem 5rem;}
@media(max-width:900px){
  .gl{grid-template-columns:1fr;}.hi{grid-template-columns:1fr;}
  .hh1{font-size:2.25rem;}.hwg,.dfg,.ft-top,.tdg,.fg,.sg{grid-template-columns:1fr;}
  .si{grid-template-columns:repeat(2,1fr);}.hs{padding:3rem 1.5rem;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# AUTH PAGE
# ════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("""<style>
    body,.stApp,[data-testid="stAppViewContainer"],
    section[data-testid="stMain"],[data-testid="stMainBlockContainer"],
    .block-container{background:#f8f5f0!important;}
    </style>""", unsafe_allow_html=True)
    st.markdown("<div style='height:3.5rem'></div>", unsafe_allow_html=True)
    _, mc, _ = st.columns([1, 1.05, 1])
    with mc:
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:1.75rem;">
          <div style="display:inline-flex;align-items:center;gap:10px;margin-bottom:1rem;">
            <div style="width:52px;height:52px;border-radius:14px;background:#e84c1e;
              display:flex;align-items:center;justify-content:center;
              box-shadow:0 8px 24px rgba(232,76,30,.35);">
              <span style="font-size:1.6rem;font-weight:900;color:#fff;
                letter-spacing:-.04em;font-family:'Inter',system-ui,sans-serif;
                line-height:1;">Q</span>
            </div>
            <div style="text-align:left;">
              <div style="font-size:1.35rem;font-weight:800;color:#111;
                letter-spacing:-.04em;line-height:1.1;">
                QuizGenius <span style="color:#e84c1e;">AI</span></div>
              <div style="font-size:.6rem;font-weight:600;color:#9ca3af;
                text-transform:uppercase;letter-spacing:.1em;margin-top:1px;">
                AI Study Platform</div>
            </div>
          </div>
          <div style="font-size:.875rem;color:#6b7280;line-height:1.7;
            max-width:280px;margin:0 auto;">
            Transform any PDF into exam-ready quizzes instantly.</div>
        </div>""", unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        with t1:
            if st.button("Sign In", key="tab_login",
                         type="primary" if st.session_state.auth_mode=="login" else "secondary",
                         use_container_width=True):
                st.session_state.auth_mode = "login"; st.rerun()
        with t2:
            if st.button("Create Account", key="tab_signup",
                         type="primary" if st.session_state.auth_mode=="signup" else "secondary",
                         use_container_width=True):
                st.session_state.auth_mode = "signup"; st.rerun()
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        if st.session_state.auth_mode == "login":
            with st.form("lf"):
                lu = st.text_input("Username", placeholder="Enter your username")
                lp = st.text_input("Password", type="password", placeholder="Enter your password")
                if st.form_submit_button("Sign In →", use_container_width=True, type="primary"):
                    if not lu.strip() or not lp.strip(): st.error("Please fill in all fields.")
                    else:
                        ok, msg = do_login(lu.strip(), lp.strip())
                        if ok: st.rerun()
                        else:  st.error(msg)
            st.markdown("""<div style="display:flex;align-items:center;gap:.75rem;margin:.875rem 0;">
              <div style="flex:1;height:1px;background:#e5e7eb;"></div>
              <span style="font-size:.72rem;color:#9ca3af;font-weight:500;">or</span>
              <div style="flex:1;height:1px;background:#e5e7eb;"></div>
            </div>""", unsafe_allow_html=True)
            if st.button("Continue as Guest →", type="secondary", use_container_width=True, key="gb"):
                st.session_state.logged_in = True
                st.session_state.current_user = "__guest__"
                st.rerun()
        else:
            with st.form("sf"):
                sdn = st.text_input("Display Name (optional)", placeholder="e.g. Vishwas Patel")
                su  = st.text_input("Username", placeholder="At least 3 characters")
                sp  = st.text_input("Password", type="password", placeholder="At least 6 characters")
                sp2 = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
                if st.form_submit_button("Create Account →", use_container_width=True, type="primary"):
                    if sp != sp2: st.error("Passwords don't match.")
                    else:
                        ok, msg = do_signup(su.strip(), sp.strip(), sdn.strip())
                        if ok: st.rerun()
                        else:  st.error(msg)
        st.markdown("""<div style="text-align:center;font-size:.7rem;color:#9ca3af;margin-top:1.5rem;">
          © 2025 QuizGenius AI · All rights reserved
        </div>""", unsafe_allow_html=True)
    st.stop()

# ════════════════════════════════════════════════════════════════
# NAVBAR
# ════════════════════════════════════════════════════════════════
cp       = st.session_state.current_page
gen_ok   = st.session_state.has_generated
uname    = st.session_state.current_user or "User"
is_guest = uname == "__guest__"
dname    = "Guest" if is_guest else uname.capitalize()
init     = dname[0].upper()

# ── shared logo HTML ─────────────────────────────────────────────
Q_LOGO = """<div style="width:{sz}px;height:{sz}px;border-radius:{r}px;background:#e84c1e;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
  box-shadow:0 4px 14px rgba(232,76,30,.35);">
  <span style="font-size:{fs};font-weight:900;color:#fff;letter-spacing:-.04em;
    font-family:'Inter',system-ui,sans-serif;line-height:1;">{ltr}</span>
</div>"""

def q_logo(sz=32, r=9, fs="1rem", ltr="Q"):
    return Q_LOGO.format(sz=sz, r=r, fs=fs, ltr=ltr)

# ── NAVBAR — 6 columns, all items on one horizontal line ─────────
nb0,nb1,nb2,nb3,nb4,nb5,nb6 = st.columns([2.0, .58, .76, .54, .88, .54, 1.9])
with nb0:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:9px;
      height:64px;white-space:nowrap;flex-shrink:0;">
      {q_logo(32, 9, "1rem", "Q")}
      <div style="line-height:1.15;">
        <div style="font-size:.88rem;font-weight:800;color:#111;letter-spacing:-.02em;
          white-space:nowrap;">QuizGenius <span style="color:#e84c1e;">AI</span></div>
        <div style="font-size:.5rem;font-weight:600;color:#9ca3af;
          text-transform:uppercase;letter-spacing:.1em;white-space:nowrap;">AI Study Platform</div>
      </div>
    </div>""", unsafe_allow_html=True)
with nb1:
    if st.button("Home",      key="n_home",  type="primary" if cp=="Home"                        else "secondary"): go("Home")
with nb2:
    if st.button("Generate",  key="n_gen",   type="primary" if cp=="Generate"                    else "secondary"): go("Generate")
with nb3:
    if st.button("Study",     key="n_study", type="primary" if cp in("Study","Flashcard","Test") else "secondary", disabled=not gen_ok): go("Study")
with nb4:
    if st.button("Dashboard", key="n_dash",  type="primary" if cp=="Dashboard"                   else "secondary"): go("Dashboard")
with nb5:
    if st.button("About",     key="n_about", type="primary" if cp=="About"                       else "secondary"): go("About")
with nb6:
    st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;
      justify-content:flex-end;height:64px;width:100%;padding-right:4px;">
      <div style="width:32px;height:32px;border-radius:50%;background:#e84c1e;
        display:flex;align-items:center;justify-content:center;
        font-size:.75rem;font-weight:800;color:#fff;flex-shrink:0;">{init}</div>
      <span style="font-size:.82rem;font-weight:600;color:#374151;
        white-space:nowrap;flex-shrink:0;">{S(dname)}</span>
      <div title="Sign out"
        onclick="(function(){{
          var c=window.parent.document;
          var b=c.querySelector('[data-testid=stAppViewBlockContainer]>div:nth-child(2) button');
          if(b){{b.click();return;}}
          var all=c.querySelectorAll('button');
          for(var i=0;i<all.length;i++){{if(all[i].innerText.trim()==='__LO__'){{all[i].click();return;}}}}
        }})()"
        style="width:34px;height:34px;border-radius:8px;border:1.5px solid #e5e7eb;
          background:#fff;display:flex;align-items:center;justify-content:center;
          flex-shrink:0;cursor:pointer;color:#6b7280;
          transition:background .15s,border-color .15s,color .15s;"
        onmouseenter="this.style.background='#fff7ed';this.style.borderColor='#e84c1e';this.style.color='#e84c1e'"
        onmouseleave="this.style.background='#fff';this.style.borderColor='#e5e7eb';this.style.color='#6b7280'">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
          <polyline points="16 17 21 12 16 7"/>
          <line x1="21" y1="12" x2="9" y2="12"/>
        </svg>
      </div>
    </div>""", unsafe_allow_html=True)

# Hidden logout button — CSS display:none hides it + kills layout space
# Native JS .click() still fires on display:none elements (confirmed browser behaviour)
if st.button("__LO__", key="n_logout"):
    for k in list(st.session_state.keys()): del st.session_state[k]
    st.rerun()

# API key warning (Generate page only)
if not get_key() and cp == "Generate":
    st.markdown("""<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:10px;
      padding:.875rem 1.375rem;margin:1rem 2rem;font-size:.875rem;color:#92400e;">
      ⚠️ <strong>Groq API key not set.</strong> Add <code>GROQ_API_KEY</code> in Render environment.
    </div>""", unsafe_allow_html=True)
    with st.expander("🔑 Enter Groq API Key (session only)"):
        ki = st.text_input("Key", type="password", placeholder="gsk_...", label_visibility="collapsed")
        if ki: st.session_state.groq_key_input = ki; st.success("Saved for this session.")

# ════════════════════════════════════════════════════════════════
# HOME PAGE
# ════════════════════════════════════════════════════════════════
if cp == "Home":
    sh   = st.session_state.score_history
    tst  = len(sh)
    avg  = round(sum(s["pct"] for s in sh)/max(tst,1),1) if sh else 0
    best = max((s["pct"] for s in sh),default=0)
    tqg  = len(st.session_state.questions)

    mockup = """<div style="background:#fff;border:1.5px solid #e5e7eb;border-radius:16px;
      padding:1.25rem;box-shadow:0 8px 32px rgba(0,0,0,.1);">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.75rem;">
        <span style="font-size:.55rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
          letter-spacing:.08em;background:#f9fafb;padding:2px 9px;border-radius:5px;
          border:1px solid #e5e7eb;">Q3 / 10 · Medium</span>
        <span style="font-size:.6rem;color:#9ca3af;">⏱ 38s</span></div>
      <div style="font-size:.8rem;font-weight:700;color:#111;line-height:1.6;margin-bottom:.75rem;">
        What is the primary function of mitochondria in a cell?</div>
      <div style="border:1.5px solid #bbf7d0;border-radius:8px;padding:.4rem .75rem;
        margin-bottom:.3rem;font-size:.7rem;color:#16a34a;font-weight:600;
        display:flex;align-items:center;gap:.5rem;background:#f0fdf4;">
        <span style="width:16px;height:16px;border-radius:50%;background:#22c55e;
          display:flex;align-items:center;justify-content:center;
          font-size:.5rem;color:#fff;font-weight:800;flex-shrink:0;">A</span>
        Produce ATP through cellular respiration</div>
      <div style="padding:.4rem .75rem;margin-bottom:.3rem;font-size:.7rem;color:#9ca3af;
        border:1.5px solid #f3f4f6;border-radius:8px;">B) Synthesise proteins for the nucleus</div>
      <div style="padding:.4rem .75rem;margin-bottom:.3rem;font-size:.7rem;color:#9ca3af;
        border:1.5px solid #f3f4f6;border-radius:8px;">C) Control cell division and growth</div>
      <div style="padding:.4rem .75rem;font-size:.7rem;color:#9ca3af;
        border:1.5px solid #f3f4f6;border-radius:8px;">D) Break down waste via autophagy</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.4rem;margin-top:.875rem;">
        <div style="background:#f9fafb;color:#6b7280;border-radius:7px;padding:.45rem;
          text-align:center;font-size:.58rem;font-weight:700;border:1.5px solid #e5e7eb;">📖 Study</div>
        <div style="background:#e84c1e;color:#fff;border-radius:7px;padding:.45rem;
          text-align:center;font-size:.58rem;font-weight:700;">🎴 Cards</div>
        <div style="background:#f0fdf4;color:#16a34a;border-radius:7px;padding:.45rem;
          text-align:center;font-size:.58rem;font-weight:700;border:1.5px solid #bbf7d0;">🎯 Test</div>
      </div></div>"""

    st.markdown(f"""<div class="hs"><div class="hi">
      <div>
        <div class="htag">✦ Next-Gen AI Study Platform</div>
        <h1 class="hh1">Turn any PDF into<br><span class="ac">exam-ready</span><br>quizzes instantly</h1>
        <p class="hp">QuizGenius AI transforms your study material into adaptive flashcards,
        MCQs, True/False, and fill-in-the-blank quizzes — powered by Groq's Llama 3.1.</p>
        <div class="pr">
          <div class="pi">📄 PDF &amp; OCR</div><div class="pi">🎴 Flashcards</div>
          <div class="pi">⏱ Timed Tests</div><div class="pi">📊 Analytics</div>
        </div>
      </div><div>{mockup}</div>
    </div></div>""", unsafe_allow_html=True)

    # CTA buttons — full width, centered, equal
    st.markdown("""<div style="max-width:1280px;margin:0 auto;padding:0 2rem;">""",
                unsafe_allow_html=True)
    hb1, hb2 = st.columns(2, gap="medium")
    with hb1:
        if st.button("⚡  Start Generating Now", key="hcta",  type="primary",  use_container_width=True): go("Generate")
    with hb2:
        if st.button("📊  My Dashboard",          key="hdash", type="secondary", use_container_width=True): go("Dashboard")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""<div class="sb"><div class="si">
      <div class="si-c"><span class="si-n">{tqg if tqg else "10K+"}</span><span class="si-l">Questions Generated</span></div>
      <div class="si-c"><span class="si-n">{tst if tst else "500+"}</span><span class="si-l">Tests Taken</span></div>
      <div class="si-c"><span class="si-n">{f"{avg}%" if sh else "95%"}</span><span class="si-l">Average Score</span></div>
      <div class="si-c"><span class="si-n">{f"{best}%" if sh else "Top"}</span><span class="si-l">Best Score</span></div>
    </div></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sec"><div class="sec-in">
      <div class="sec-ey">HOW IT WORKS</div>
      <div class="sec-ti">Three steps to smarter study</div>
      <p class="sec-su">Upload any PDF, let AI generate questions, then study and test yourself.</p>
      <div class="hwg">
        <div class="hwc" data-n="1"><div class="hwi">📄</div>
          <div class="hwt">1. Upload Your PDF</div>
          <p class="hwp">Drag and drop any study material. OCR handles scanned documents automatically.</p></div>
        <div class="hwc" data-n="2"><div class="hwi">🧠</div>
          <div class="hwt">2. AI Generates Questions</div>
          <p class="hwp">Groq-powered Llama 3.1 extracts key concepts and creates questions in seconds.</p></div>
        <div class="hwc" data-n="3"><div class="hwi">🎯</div>
          <div class="hwt">3. Study &amp; Excel</div>
          <p class="hwp">Flip flashcards, bookmark tough spots, take timed tests, and track improvement.</p></div>
      </div></div></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sec-alt"><div class="sec-in">
      <div class="sec-ey">FEATURES</div>
      <div class="sec-ti">Everything you need to excel</div>
      <p class="sec-su">Built for serious learners and exam preparation.</p>
      <div class="fg" style="margin-top:2rem;">
        <div class="fcc"><div class="fci">🎴</div><div><div class="fct">Flashcard Mode</div>
          <div class="fcs">3D flip cards with progress tracking and bookmarks.</div></div></div>
        <div class="fcc"><div class="fci">⭐</div><div><div class="fct">Smart Bookmarks</div>
          <div class="fcs">Star tough questions and revisit them anytime.</div></div></div>
        <div class="fcc"><div class="fci">🎯</div><div><div class="fct">Auto Difficulty Detection</div>
          <div class="fcs">AI analyses your PDF and recommends the right level.</div></div></div>
        <div class="fcc"><div class="fci">❌</div><div><div class="fct">Wrong Answer Tracker</div>
          <div class="fcs">Mistakes are saved so you can target weak spots.</div></div></div>
        <div class="fcc"><div class="fci">⏱</div><div><div class="fct">Per-Question Timer</div>
          <div class="fcs">Countdown per question to simulate real exam pressure.</div></div></div>
        <div class="fcc"><div class="fci">📊</div><div><div class="fct">Score History &amp; Chart</div>
          <div class="fcs">Track every test and visualise improvement over time.</div></div></div>
        <div class="fcc"><div class="fci">🗂️</div><div><div class="fct">Topic Filter</div>
          <div class="fcs">Focus generation on specific chapters or sections.</div></div></div>
        <div class="fcc"><div class="fci">🔀</div><div><div class="fct">3 Question Types</div>
          <div class="fcs">MCQ, True/False, or Fill-in-the-blank.</div></div></div>
        <div class="fcc"><div class="fci">👁</div><div><div class="fct">Live Preview</div>
          <div class="fcs">Preview a real question before generating the full set.</div></div></div>
        <div class="fcc"><div class="fci">📤</div><div><div class="fct">Export as HTML</div>
          <div class="fcs">Download a formatted quiz sheet for offline study.</div></div></div>
      </div></div></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sec"><div class="sec-in">
      <div class="sec-ey">DIFFICULTY LEVELS</div>
      <div class="sec-ti">Choose your challenge</div>
      <p class="sec-su">Three levels designed to match your current learning stage.</p>
      <div class="dfg">
        <div class="dfc e">
          <span class="badge b-gr">Foundational</span>
          <div class="dfn">🌱 Easy</div>
          <p class="dfd">Direct recall, key terminology, foundational concepts. 5 test questions.</p></div>
        <div class="dfc m">
          <span class="badge b-or">Intermediate</span>
          <div class="dfn">📈 Medium</div>
          <p class="dfd">Applied comprehension, connecting concepts. 7 test questions.</p></div>
        <div class="dfc h">
          <span class="badge b-re">Mastery</span>
          <div class="dfn">🔥 Hard</div>
          <p class="dfd">Advanced synthesis and critical analysis. 10 questions.</p></div>
      </div></div></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="ft"><div class="fti">
      <div class="ft-top">
        <div>
          <div class="ft-brand">
            <div class="ft-logo" style="display:flex;align-items:center;justify-content:center;"><span style="font-size:.85rem;font-weight:900;color:#fff;font-family:'Inter',system-ui,sans-serif;line-height:1;">Q</span></div><span class="ft-name">QuizGenius AI</span></div>
          <p class="ft-desc">Transform any PDF into an adaptive learning experience.
            Powered by Groq + Llama 3.1.</p></div>
        <div><div class="ft-hd">Product</div>
          <span class="ft-lk">Quiz Generator</span>
          <span class="ft-lk">Flashcard Mode</span>
          <span class="ft-lk">Adaptive Testing</span></div>
        <div><div class="ft-hd">Resources</div>
          <span class="ft-lk">Documentation</span>
          <span class="ft-lk">Help Center</span>
          <span class="ft-lk">About</span></div>
        <div><div class="ft-hd">Connect</div>
          <p style="font-size:.8rem;color:#6b7280;line-height:1.8;">patelvishwas702@gmail.com</p></div>
      </div>
      <div class="ft-bot">
        <span>© 2025 QuizGenius AI. All rights reserved.</span>
        <span>Design by Vishwas Patel</span>
      </div></div></div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# GENERATE PAGE
# ════════════════════════════════════════════════════════════════
elif cp == "Generate":
    # White header band matching home hero start position
    st.markdown("""<div style="background:#fff;border-bottom:1px solid #e5e7eb;
      padding:1.5rem 2rem 1.25rem;margin-bottom:0;">
      <div style="max-width:1280px;margin:0 auto;">
        <div style="font-size:.6rem;font-weight:700;text-transform:uppercase;
          letter-spacing:.1em;color:#9ca3af;margin-bottom:.2rem;">Workspace › AI Creation</div>
        <div style="font-size:1.6rem;font-weight:800;color:#111;letter-spacing:-.03em;">
          Generate Study Material</div>
        <div style="font-size:.875rem;color:#6b7280;margin-top:.2rem;">
          Transform your documents into high-quality assessments instantly.</div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="fw" style="padding-top:.25rem;">', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 0.52], gap="large")

    with left_col:
        if not st.session_state.pdf_text.strip():
            st.markdown('<div class="cfg">', unsafe_allow_html=True)
            st.markdown("""<div class="cfg-lbl">📄 Upload PDF or Document</div>
            <div class="cfg-hint">Drag and drop your lecture notes, textbooks, or research papers.</div>
            """, unsafe_allow_html=True)
            uf = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
            if uf:
                with st.spinner("Reading PDF…"):
                    text = get_pdf_text(uf)
                if text.strip():
                    import hashlib as _hm
                    clear_pdf()
                    st.session_state.pdf_text     = text
                    st.session_state.pdf_filename = uf.name
                    st.session_state.pdf_size     = uf.size
                    st.session_state.pdf_hash     = _hm.md5(text.encode()).hexdigest()
                    st.session_state.detected_difficulty = detect_diff(text)
                    st.session_state.topics       = extract_topics(text)
                    st.rerun()
                else:
                    st.error("Could not extract text. Try another PDF.")
        else:
            pdf_text = st.session_state.pdf_text
            wc       = len(pdf_text.split())
            max_q    = calc_max_q(wc)
            fn       = st.session_state.pdf_filename or "Uploaded PDF"
            sz       = st.session_state.pdf_size / 1024
            dd       = st.session_state.detected_difficulty
            dcol     = {"Easy":"#22c55e","Medium":"#e84c1e","Hard":"#ef4444"}.get(dd,"#e84c1e")

            pb1, pb2 = st.columns([5, 1])
            with pb1:
                st.markdown(f"""<div class="pdf-bn" style="margin-bottom:0;">
                  <div class="pdf-ic">📄</div>
                  <div style="flex:1;min-width:0;">
                    <div style="font-size:.9rem;font-weight:700;color:#111;">{S(fn)}</div>
                    <div style="font-size:.72rem;color:#9ca3af;margin-top:.125rem;">{wc:,} words · {sz:.1f} KB ·
                      AI suggests: <span style="color:{dcol};font-weight:700;">{dd}</span></div>
                  </div></div>""", unsafe_allow_html=True)
            with pb2:
                st.markdown('<div style="padding-top:8px;">', unsafe_allow_html=True)
                if st.button("Change", key="chg_pdf", type="secondary", use_container_width=True):
                    clear_pdf(); st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            s1,s2,s3,s4 = st.columns(4)
            for col,lbl,val in [(s1,"Words",f"{wc:,}"),(s2,"Characters",f"{len(pdf_text):,}"),
                                 (s3,"File Size",f"{sz:.0f} KB"),(s4,"Max Qs",str(max_q))]:
                with col:
                    st.markdown(f"""<div style="background:#fff;border:1.5px solid #e5e7eb;
                      border-radius:10px;padding:.875rem;text-align:center;
                      box-shadow:0 1px 4px rgba(0,0,0,.05);margin-bottom:.875rem;">
                      <div style="font-size:1.25rem;font-weight:800;color:#e84c1e;">{val}</div>
                      <div style="font-size:.58rem;font-weight:600;text-transform:uppercase;
                        letter-spacing:.09em;color:#9ca3af;">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            if st.session_state.topics:
                with st.expander(f"🗂️ Topic Filter — {len(st.session_state.topics)} sections detected"):
                    st.caption("Select sections to focus on. Leave empty for full document.")
                    sel = st.multiselect("Topics", st.session_state.topics,
                                         default=st.session_state.selected_topics,
                                         label_visibility="collapsed")
                    st.session_state.selected_topics = sel
                    if sel: st.success(f"Focusing on {len(sel)} topic(s).")

            st.markdown("""<div class="cfg"><div class="cfg-lbl">Question Count</div>
            <div class="cfg-hint">How many questions to generate from your document.</div></div>""",
                        unsafe_allow_html=True)
            num_q = st.slider("Questions", 1, max_q, min(10, max_q), label_visibility="collapsed")
            st.markdown(f"""<div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;
              padding:.5rem 1rem;margin-bottom:.875rem;
              display:flex;align-items:center;justify-content:space-between;">
              <span style="font-size:.78rem;font-weight:600;color:#92400e;">
                Selected: <strong>{num_q} questions</strong></span>
              <span style="font-size:.72rem;color:#d97706;">Max: {max_q}</span>
            </div>""", unsafe_allow_html=True)

            # Question Type selection
            st.markdown("""<div class="cfg"><div class="cfg-lbl">Question Type</div>
            <div class="cfg-hint">Choose the format of generated questions.</div></div>""",
                        unsafe_allow_html=True)
            cqt = st.session_state.q_type
            qa1,qa2,qa3 = st.columns(3)
            for col,qt,ico,lbl,sub in [
                (qa1,"MCQ","📝","Multiple Choice","4 options, 1 correct"),
                (qa2,"TF","✅","True / False","Fact-based statements"),
                (qa3,"FIB","✏️","Fill in Blank","Complete the sentence")]:
                with col:
                    st.markdown(f"""<div class="qtc {'sel' if cqt==qt else ''}">
                      <div class="qtc-ico">{ico}</div>
                      <div class="qtc-n">{lbl}</div>
                      <div class="qtc-d">{sub}</div>
                    </div>""", unsafe_allow_html=True)
                    if st.button(f"{'✓ ' if cqt==qt else ''}{lbl}", key=f"qt_{qt}",
                                  type="primary" if cqt==qt else "secondary",
                                  use_container_width=True):
                        st.session_state.q_type = qt; st.rerun()

            st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)

            # Difficulty
            st.markdown("""<div class="cfg"><div class="cfg-lbl">Difficulty Level</div>
            <div class="cfg-hint">Adjusts question complexity and cognitive depth.</div></div>""",
                        unsafe_allow_html=True)
            cur_d = st.session_state.detected_difficulty
            d1,d2,d3 = st.columns(3)
            for col,dname,ico in [(d1,"Easy","🌱"),(d2,"Medium","📈"),(d3,"Hard","🔥")]:
                with col:
                    sel_cls = {"Easy":"border:1.5px solid #22c55e;background:#f0fdf4;",
                               "Medium":"border:1.5px solid #e84c1e;background:#fff7ed;",
                               "Hard":"border:1.5px solid #ef4444;background:#fef2f2;"}.get(dname,"")
                    st.markdown(f"""<div style="border-radius:8px;padding:.75rem;text-align:center;
                      background:#fff;cursor:pointer;transition:all .18s;
                      {sel_cls if cur_d==dname else 'border:1.5px solid #e5e7eb;'}">
                      <div style="font-size:.78rem;font-weight:700;">{ico} {dname}</div>
                    </div>""", unsafe_allow_html=True)
                    if st.button(dname, key=f"d_{dname}",
                                  type="primary" if cur_d==dname else "secondary",
                                  use_container_width=True):
                        st.session_state.detected_difficulty = dname; st.rerun()

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

            # GENERATE BUTTON
            gen_btn = st.button("⚡  GENERATE QUESTIONS", type="primary",
                                 use_container_width=True, key="gen_btn")
            prog_ph = st.empty()

            # ONE "Open Study Mode" button — only shown after generation
            if st.session_state.generation_done and st.session_state.has_generated:
                qt_label = {"MCQ":"Multiple Choice","TF":"True/False","FIB":"Fill-in-the-Blank"}.get(
                    st.session_state.q_type,"")
                st.success(f"✅ Generated {len(st.session_state.questions)} {qt_label} questions!")
                if st.button("📖  Open Study Mode →", type="secondary",
                              use_container_width=True, key="open_study_final"):
                    go("Study")

            if gen_btn:
                if not get_key():
                    st.error("Enter your Groq API key first.")
                else:
                    cur_hash = st.session_state.pdf_hash
                    for k2,v2 in [("questions",[]),("test_questions",[]),("has_generated",False),
                                   ("has_test_generated",False),("generation_done",False),
                                   ("questions_pdf_hash",""),("selected_difficulty",None),
                                   ("user_answers",{}),("test_submitted",False),("chunks",[]),
                                   ("bookmarks",[]),("wrong_answers",[]),
                                   ("preview_question",None),("show_preview",False)]:
                        st.session_state[k2] = v2
                    st.session_state.quiz_key += 1

                    def upd(pct, msg):
                        prog_ph.markdown(f"""<div class="pb-wrap">
                          <div class="pb-top"><span class="pb-lbl">⚡ {msg}</span>
                            <span class="pb-pct">{pct}%</span></div>
                          <div class="pb-bar"><div class="pb-fill" style="width:{pct}%"></div></div>
                          <p class="pb-note">Groq + Llama 3.1 is working…</p>
                        </div>""", unsafe_allow_html=True)

                    upd(5,"Splitting document…")
                    use_text = pdf_text
                    if st.session_state.selected_topics:
                        filt = ""
                        for t in st.session_state.selected_topics:
                            idx = pdf_text.lower().find(t.lower())
                            if idx >= 0: filt += pdf_text[idx:idx+3000]+"\n\n"
                        if filt.strip(): use_text = filt

                    if LC_AVAILABLE:
                        chunks = RecursiveCharacterTextSplitter(
                            chunk_size=1200,chunk_overlap=100).split_text(use_text)
                    else:
                        chunks = [use_text[i:i+1200] for i in range(0,len(use_text),1100)]
                    st.session_state.chunks = chunks
                    qt_now = st.session_state.q_type
                    diff_now = st.session_state.detected_difficulty
                    temp_qs = []

                    for i in range(num_q):
                        upd(10+int((i+1)*85/num_q), f"Generating question {i+1} of {num_q}…")
                        rel = keyword_search("key concepts important facts", chunks, k=3)
                        ctx = "\n".join(rel[:2]) if rel else chunks[i%len(chunks)]
                        try:
                            temp_qs.append(llm_gen(ctx, qt_now, diff_now))
                        except Exception as e:
                            temp_qs.append({"question":f"[Error: {str(e)[:80]}]",
                                "options":["A) --","B) --","C) --","D) --"],
                                "correct":"A","context":"","type":qt_now,"difficulty":diff_now})

                    upd(100,"Done!")
                    st.session_state.questions          = temp_qs
                    st.session_state.questions_pdf_hash = cur_hash
                    st.session_state.has_generated      = True
                    st.session_state.generation_done    = True
                    st.rerun()

    # RIGHT PANEL — Live Preview (sticky, aligned with left column top)
    with right_col:
        st.markdown("""<div style="position:sticky;top:72px;">""", unsafe_allow_html=True)
        st.markdown("""<div class="gp">
          <div class="gp-hd">
            <span class="gp-ti">🔍 Live Preview</span>
            <span class="badge b-or">SAMPLE</span>
          </div>""", unsafe_allow_html=True)

        if st.session_state.pdf_text.strip():
            if st.session_state.show_preview and st.session_state.preview_question:
                pq = st.session_state.preview_question
                st.markdown(f"""<div style="padding:1.25rem;">
                  <span class="badge b-or" style="margin-bottom:.625rem;display:inline-block;">
                    {S(pq["type"])} · {S(pq["difficulty"])}</span>
                  <div style="font-size:.9rem;font-weight:700;color:#111;
                    line-height:1.6;margin-bottom:.75rem;">{S(pq["question"])}</div>
                </div>""", unsafe_allow_html=True)
                seen = set()
                for opt in pq["options"]:
                    raw = opt.strip()
                    if not raw: continue
                    lt = raw[0]
                    if lt not in "ABCD" or lt in seen: continue
                    seen.add(lt)
                    txt = clean_opt(raw)
                    if lt == pq["correct"]:
                        st.markdown(f"""<div class="po-ok" style="margin:0 1.25rem .25rem;">
                          <div class="pl-ok">{S(lt)}</div>
                          <div class="pt-ok">{S(txt)}</div>
                          <span style="color:#22c55e;margin-left:auto;">✓</span>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="po" style="margin:0 1.25rem .25rem;">
                          <div class="pl">{S(lt)}</div>
                          <div class="pt">{S(txt)}</div>
                        </div>""", unsafe_allow_html=True)
                st.markdown("""<div style="margin:1rem 1.25rem 1.25rem;padding:.625rem .875rem;
                  background:#f9fafb;border-radius:8px;font-size:.72rem;color:#6b7280;">
                  💡 This is a real sample question from your document.</div>""",
                    unsafe_allow_html=True)
                if st.button("🔄 New Preview", key="new_prev", type="secondary",
                              use_container_width=True):
                    st.session_state.show_preview = False
                    st.session_state.preview_question = None
                    st.rerun()
            else:
                st.markdown("""<div class="gp-em">
                  <div class="gp-em-ic">📋</div>
                  <div class="gp-em-t">No content generated</div>
                  <div class="gp-em-s">Click Preview Sample to see<br>a real question from your document.</div>
                </div>""", unsafe_allow_html=True)
                if st.button("👁  Preview Sample Question", type="secondary",
                              use_container_width=True, key="prev_btn"):
                    if not get_key():
                        st.error("Add your Groq API key first.")
                    else:
                        with st.spinner("Generating preview…"):
                            try:
                                chunks_p = st.session_state.chunks
                                if not chunks_p:
                                    if LC_AVAILABLE:
                                        chunks_p = RecursiveCharacterTextSplitter(
                                            chunk_size=1200,chunk_overlap=100
                                        ).split_text(st.session_state.pdf_text)
                                    else:
                                        chunks_p = [st.session_state.pdf_text[i:i+1200]
                                                    for i in range(0,len(st.session_state.pdf_text),1100)]
                                pq = llm_gen(chunks_p[0], st.session_state.q_type,
                                              st.session_state.detected_difficulty)
                                st.session_state.preview_question = pq
                                st.session_state.show_preview = True
                                st.rerun()
                            except Exception as e:
                                st.error(f"Preview failed: {e}")
        else:
            st.markdown("""<div class="gp-em">
              <div class="gp-em-ic">📄</div>
              <div class="gp-em-t">Upload a PDF first</div>
              <div class="gp-em-s">Upload a document on the left<br>to enable live preview.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div style="padding:.875rem 1.25rem;border-top:1px solid #f3f4f6;
          background:#fffbeb;border-radius:0 0 12px 12px;">
          <div style="font-size:.72rem;font-weight:700;color:#d97706;margin-bottom:.25rem;">💡 Pro Tip</div>
          <div style="font-size:.7rem;color:#92400e;line-height:1.65;">
            For best results, upload text-heavy PDFs. Diagrams and handwritten notes may vary.</div>
        </div></div></div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# STUDY / FLASHCARD / TEST
# ════════════════════════════════════════════════════════════════
elif cp in ("Study","Flashcard","Test"):
    if not gen_ok:
        st.markdown("""<div style="max-width:500px;margin:5rem auto;text-align:center;padding:2rem;">
          <div style="width:72px;height:72px;border-radius:50%;background:#fff7ed;
            border:1.5px solid #fed7aa;display:flex;align-items:center;justify-content:center;
            font-size:2rem;margin:0 auto 1.25rem;">📄</div>
          <div style="font-size:1.375rem;font-weight:800;color:#111;margin-bottom:.5rem;">No Questions Yet</div>
          <div style="font-size:.875rem;color:#6b7280;margin-bottom:2rem;">Upload a PDF and generate questions first.</div>
        </div>""", unsafe_allow_html=True)
        _,c2,_ = st.columns([1,1,1])
        with c2:
            if st.button("⚡ Go to Generate", type="primary", use_container_width=True): go("Generate")
        st.stop()

    if (st.session_state.questions_pdf_hash and st.session_state.pdf_hash and
            st.session_state.questions_pdf_hash != st.session_state.pdf_hash):
        st.warning("New PDF detected — please regenerate questions.")
        if st.button("⚡ Regenerate", type="primary"): go("Generate")
        st.stop()

    st.markdown('<div class="fw">', unsafe_allow_html=True)
    st.markdown('<div style="height:.625rem"></div>', unsafe_allow_html=True)
    tc1,tc2,tc3 = st.columns(3)
    with tc1:
        if st.button("📖  Study",      key="tab_s", type="primary" if cp=="Study"     else "secondary", use_container_width=True): go("Study")
    with tc2:
        if st.button("🎴  Flashcards", key="tab_f", type="primary" if cp=="Flashcard" else "secondary", use_container_width=True): go("Flashcard")
    with tc3:
        if st.button("🎯  Test",       key="tab_t", type="primary" if cp=="Test"      else "secondary", use_container_width=True):
            st.session_state.selected_difficulty = None
            st.session_state.has_test_generated  = False
            st.session_state.user_answers        = {}
            st.session_state.test_submitted      = False
            go("Test")

    st.markdown('<div style="height:1.25rem"></div>', unsafe_allow_html=True)

    # ── STUDY ──
    if cp == "Study":
        qs  = st.session_state.questions
        bms = st.session_state.bookmarks
        fn  = st.session_state.pdf_filename or "PDF"
        sh1,sh2,sh3 = st.columns([3,1,1])
        with sh1:
            st.markdown(f"""<div style="display:flex;align-items:center;gap:.75rem;margin-bottom:1.25rem;">
              <span style="font-size:1.125rem;font-weight:800;color:#111;">Study Questions</span>
              <span class="badge b-or">{len(qs)} Qs · {st.session_state.q_type}</span>
            </div>""", unsafe_allow_html=True)
        with sh2:
            fm = st.selectbox("Filter",["All","Bookmarked","Wrong Answers"],
                               label_visibility="collapsed",key="sf")
        with sh3:
            st.download_button("📤 Export",
                export_html(qs,f"QuizGenius — {fn}").encode(),
                f"quiz_{fn.replace('.pdf','')}.html","text/html",use_container_width=True)

        with st.expander("📋 Copy all questions as text"):
            st.code("\n\n".join(
                f"Q{i+1}: {q['question']}\n"+"".join(f"  {o}\n" for o in q['options'])+
                f"  Answer: {q['correct']}" for i,q in enumerate(qs)),language=None)

        if fm == "Bookmarked":
            dqs = [(i,q) for i,q in enumerate(qs) if i in bms]
            if not dqs: st.info("No bookmarks yet. Star questions while studying.")
        elif fm == "Wrong Answers":
            wa = {q["question"] for q in st.session_state.wrong_answers}
            dqs = [(i,q) for i,q in enumerate(qs) if q["question"] in wa]
            if not dqs: st.info("No wrong answers tracked yet — take a test first.")
        else:
            dqs = list(enumerate(qs))

        for idx,q in dqs:
            is_bm = idx in bms
            st.markdown(f"""<div class="qc">
              <div class="qc-head">
                <span class="qc-num">Q{idx+1}</span>
                <span class="badge b-or">{S(q.get('type','MCQ'))}</span>
              </div>
              <div class="qc-q">{S(q['question'])}</div>""", unsafe_allow_html=True)
            seen = set()
            for opt in q["options"]:
                raw = opt.strip()
                if not raw: continue
                lt = raw[0]
                if lt not in "ABCD" or lt in seen: continue
                seen.add(lt)
                txt = clean_opt(raw)
                if lt == q["correct"]:
                    st.markdown(f"""<div class="oc oc-ok">
                      <div class="ol-ok">{S(lt)}</div>
                      <div class="ot-ok">{S(txt)}</div>
                      <span style="margin-left:auto;color:#22c55e;">✓</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="oc oc-no">
                      <div class="ol-no">{S(lt)}</div>
                      <div class="ot-no">{S(txt)}</div>
                    </div>""", unsafe_allow_html=True)
            st.markdown('<div class="qc-foot">', unsafe_allow_html=True)
            bc,cc = st.columns([1,4])
            with bc:
                if st.button("⭐ Saved" if is_bm else "☆ Bookmark",
                              key=f"bm_{idx}_{st.session_state.quiz_key}",
                              type="primary" if is_bm else "secondary",
                              use_container_width=True):
                    if idx in st.session_state.bookmarks: st.session_state.bookmarks.remove(idx)
                    else: st.session_state.bookmarks.append(idx)
                    persist(); st.rerun()
            with cc:
                with st.expander(f"Source context — Q{idx+1}"):
                    st.markdown(f"""<div style="background:#f9fafb;border-left:3px solid #e84c1e;
                      padding:.75rem 1rem;border-radius:7px;font-size:.825rem;
                      color:#374151;line-height:1.7;">{S(q['context'])}</div>""", unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

    # ── FLASHCARD ──
    elif cp == "Flashcard":
        def get_fc_qs():
            qs2 = st.session_state.questions; f2 = st.session_state.fc_filter
            if f2=="Bookmarked": return [(i,q) for i,q in enumerate(qs2) if i in st.session_state.bookmarks]
            if f2=="Mistakes":
                wa2={q["question"] for q in st.session_state.wrong_answers}
                return [(i,q) for i,q in enumerate(qs2) if q["question"] in wa2]
            return list(enumerate(qs2))

        fc_qs=get_fc_qs(); total=len(fc_qs)
        st.markdown('<div style="max-width:680px;margin:0 auto;">', unsafe_allow_html=True)
        st.markdown("""<div style="text-align:center;margin-bottom:1.5rem;">
          <div style="font-size:1.5rem;font-weight:800;color:#111;margin-bottom:.25rem;">Flashcards</div>
          <div style="font-size:.875rem;color:#6b7280;">Click the card to reveal the answer.</div>
        </div>""", unsafe_allow_html=True)
        f1,f2,f3 = st.columns(3)
        for fi,flt in zip([f1,f2,f3],["All","Bookmarked","Mistakes"]):
            with fi:
                if st.button(flt,key=f"fcf_{flt}",
                              type="primary" if st.session_state.fc_filter==flt else "secondary",
                              use_container_width=True):
                    st.session_state.fc_filter=flt; st.session_state.fc_idx=0; st.rerun()
        st.markdown("<div style='height:.875rem'></div>", unsafe_allow_html=True)
        if not fc_qs:
            st.info("No cards match this filter." if st.session_state.fc_filter!="All"
                    else "No questions generated yet.")
        else:
            idx=min(st.session_state.fc_idx,total-1); oi,q=fc_qs[idx]
            is_bm=oi in st.session_state.bookmarks
            ans=clean_opt(next((o for o in q["options"] if o.startswith(q["correct"])),q["correct"]))
            render_flashcard(q["question"],ans,idx+1,total,is_bm)
            st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
            n1,n2,n3,n4 = st.columns(4)
            with n1:
                if st.button("← Prev", key="fp", type="secondary", use_container_width=True):
                    st.session_state.fc_idx=(idx-1)%total; st.rerun()
            with n2:
                if st.button("🔀 Random", key="fr", type="secondary", use_container_width=True):
                    import random; st.session_state.fc_idx=random.randint(0,total-1); st.rerun()
            with n3:
                if st.button("Next →", key="fn", type="secondary", use_container_width=True):
                    st.session_state.fc_idx=(idx+1)%total; st.rerun()
            with n4:
                if st.button("⭐ Saved" if is_bm else "☆ Save",
                              key=f"fc_bm_{idx}",type="primary" if is_bm else "secondary",
                              use_container_width=True):
                    if oi in st.session_state.bookmarks: st.session_state.bookmarks.remove(oi)
                    else: st.session_state.bookmarks.append(oi)
                    persist(); st.rerun()
            st.markdown(f"""<div style="text-align:center;margin-top:.875rem;
              font-size:.82rem;font-weight:600;color:#9ca3af;">
              {idx+1} / {total} · {st.session_state.fc_filter}</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── TEST ──
    elif cp == "Test":
        if not st.session_state.selected_difficulty:
            st.markdown("""<div style="background:#eff6ff;border:1px solid #bfdbfe;
              border-radius:10px;padding:.875rem 1.375rem;margin-bottom:1.25rem;
              font-size:.875rem;color:#1e40af;">
              📝 Test questions are generated fresh via Groq — separate from your study set.
            </div>""", unsafe_allow_html=True)
            tm1,tm2 = st.columns([2,2])
            with tm1:
                timed = st.checkbox("⏱ Enable per-question timer", value=st.session_state.timed_mode)
                st.session_state.timed_mode = timed
            if timed:
                with tm2:
                    secs = st.slider("Seconds",15,120,st.session_state.per_q_time,5,label_visibility="collapsed")
                    st.session_state.per_q_time = secs
                    st.caption(f"⏱ {secs}s per question")
            st.markdown("""<div style="text-align:center;margin:1.5rem 0 1rem;">
              <div style="font-size:1.25rem;font-weight:800;color:#111;margin-bottom:.25rem;">
                Select Your Challenge Level</div>
              <div style="font-size:.875rem;color:#6b7280;">Fresh questions are generated for each level.</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("""<div class="tdg">
              <div class="tdc"><div class="tdc-ico e">🌱</div>
                <div class="tdc-n">Beginner</div>
                <div class="tdc-h">Foundational concepts and direct recall. 5 questions.</div></div>
              <div class="tdc feat"><span class="tdc-pop">POPULAR</span>
                <div class="tdc-ico m">🎓</div>
                <div class="tdc-n">Standard</div>
                <div class="tdc-h">Applied comprehension scenarios. 7 questions.</div></div>
              <div class="tdc"><div class="tdc-ico h">🔥</div>
                <div class="tdc-n">Expert</div>
                <div class="tdc-h">Complex synthesis and analysis. 10 questions.</div></div>
            </div>""", unsafe_allow_html=True)
            d1,d2,d3 = st.columns(3)
            with d1:
                if st.button("🌱 Start Easy",   key="easy", use_container_width=True, type="secondary"):
                    st.session_state.selected_difficulty="Easy"; st.session_state.test_started_at=int(time.time()*1000); st.rerun()
            with d2:
                if st.button("🎓 Start Medium", key="med",  use_container_width=True, type="primary"):
                    st.session_state.selected_difficulty="Medium"; st.session_state.test_started_at=int(time.time()*1000); st.rerun()
            with d3:
                if st.button("🔥 Start Hard",   key="hard", use_container_width=True, type="secondary"):
                    st.session_state.selected_difficulty="Hard"; st.session_state.test_started_at=int(time.time()*1000); st.rerun()

        elif not st.session_state.has_test_generated:
            diff=st.session_state.selected_difficulty
            n={"Easy":5,"Medium":7,"Hard":10}[diff]
            if not get_key():
                st.error("Groq API key required.")
                if st.button("← Back"): st.session_state.selected_difficulty=None; st.rerun()
            else:
                st.markdown(f"""<div style="background:#fff;border:1.5px solid #e5e7eb;border-radius:14px;
                  padding:2.5rem;text-align:center;box-shadow:0 4px 24px rgba(0,0,0,.08);">
                  <div style="width:56px;height:56px;border-radius:14px;background:#e84c1e;
                    display:flex;align-items:center;justify-content:center;
                    box-shadow:0 6px 20px rgba(232,76,30,.3);margin:0 auto 1rem;">
                    <span style="font-size:1.5rem;font-weight:900;color:#fff;
                      font-family:'Inter',system-ui,sans-serif;line-height:1;">Q</span></div>
                  <div style="font-size:1.125rem;font-weight:800;color:#111;margin-bottom:.375rem;">
                    Generating {diff} Test</div>
                  <div style="font-size:.875rem;color:#6b7280;">Creating {n} fresh questions via Groq Llama 3.1…</div>
                </div>""", unsafe_allow_html=True)
                prog=st.progress(0,text=f"Generating {diff} test…")
                st.session_state.test_questions=[]; st.session_state.user_answers={}; st.session_state.test_submitted=False
                chunks=st.session_state.chunks; qt=st.session_state.q_type; temp=[]
                for i in range(n):
                    prog.progress(int((i+1)*100/n),text=f"Question {i+1}/{n}…")
                    rel=keyword_search(f"{diff} level concepts",chunks,k=3)
                    ctx=("\n".join(rel[:2]) if rel else
                         (chunks[i%len(chunks)] if chunks else st.session_state.pdf_text[:1500]))
                    try: temp.append(llm_gen(ctx,qt,diff))
                    except Exception as e:
                        temp.append({"question":f"[Error: {str(e)[:80]}]",
                            "options":["A) --","B) --","C) --","D) --"],
                            "correct":"A","context":"","type":qt,"difficulty":diff})
                prog.progress(100,text="Ready!")
                st.session_state.test_questions=temp; st.session_state.has_test_generated=True
                prog.empty(); st.rerun()

        elif not st.session_state.test_submitted:
            diff=st.session_state.selected_difficulty
            tq=len(st.session_state.test_questions); ans_cnt=len(st.session_state.user_answers)
            pct_d=int(ans_cnt/tq*100) if tq else 0
            started=st.session_state.test_started_at or int(time.time()*1000)
            st.markdown(f"""<div class="tpb">
              <div style="display:flex;flex-direction:column;gap:4px;flex:1;">
                <div style="display:flex;justify-content:space-between;">
                  <span style="font-size:.72rem;font-weight:700;color:#e84c1e;">{ans_cnt}/{tq} answered</span>
                  <span style="font-size:.72rem;font-weight:700;color:#9ca3af;">{pct_d}%</span></div>
                <div class="tpb-bar"><div class="tpb-fill" style="width:{pct_d}%"></div></div>
              </div>
              <div class="tpb-timer">⏱ <span id="tf">00:00</span></div>
            </div>
            <script>(function(){{var s={started};setInterval(function(){{
              var e=document.getElementById("tf");
              if(e){{var t=Math.floor((Date.now()-s)/1000);
                e.innerText=(Math.floor(t/60)<10?"0":"")+Math.floor(t/60)+":"+(t%60<10?"0":"")+t%60;}}}},500);}})();
            </script>""", unsafe_allow_html=True)

            for i,q in enumerate(st.session_state.test_questions):
                d=q.get("difficulty",diff)
                dcol={"Easy":"#22c55e","Medium":"#e84c1e","Hard":"#ef4444"}.get(d,"#e84c1e")
                st.markdown(f"""<div class="tc">
                  <div class="tc-top">
                    <span class="tc-num">Q{i+1}</span>
                    <span class="badge" style="background:{dcol}18;color:{dcol};
                      border:1px solid {dcol}44;">{S(d)}</span>
                  </div>
                  <div class="tc-q">{S(q['question'])}</div>
                </div>""", unsafe_allow_html=True)
                copts=[]; seen=set()
                for opt in q["options"]:
                    raw=opt.strip()
                    if not raw: continue
                    lt=raw[0]
                    if lt not in "ABCD" or lt in seen: continue
                    seen.add(lt); copts.append(f"{lt}) {clean_opt(raw)}")
                a = st.radio(f"q{i}", copts, index=None,
                              key=f"tq_{i}_{st.session_state.quiz_key}",
                              label_visibility="collapsed")
                if a: st.session_state.user_answers[i] = a[0]
                st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

            s1,s2,s3 = st.columns([1,2,1])
            with s2:
                if ans_cnt == tq:
                    if st.button("✅ Submit Test", use_container_width=True, type="primary"):
                        corr=sum(1 for i2,q2 in enumerate(st.session_state.test_questions)
                                  if st.session_state.user_answers.get(i2)==q2["correct"])
                        save_score(diff,corr,tq,st.session_state.pdf_filename or "Unknown")
                        st.session_state.test_submitted=True; st.rerun()
                else:
                    st.warning(f"Answer all {tq} questions first. ({ans_cnt}/{tq} done)")

        else:
            diff=st.session_state.selected_difficulty
            corr=sum(1 for i,q in enumerate(st.session_state.test_questions)
                      if st.session_state.user_answers.get(i)==q["correct"])
            total=len(st.session_state.test_questions)
            pct=corr/total*100 if total else 0; wrong=total-corr
            render_result(pct,corr,total)
            if wrong>0:
                st.markdown(f"""<div style="background:#fffbeb;border:1px solid #fde68a;
                  border-radius:10px;padding:.75rem 1.125rem;font-size:.875rem;
                  color:#92400e;margin:.75rem 0;">
                  ⚠️ {wrong} wrong answer(s) saved — revisit via Study → Wrong Answers.</div>""",
                    unsafe_allow_html=True)
            st.markdown("""<div style="background:#fff;border:1.5px solid #e5e7eb;
              border-radius:14px;overflow:hidden;margin-top:.75rem;">
              <div style="padding:1.25rem;border-bottom:1px solid #f3f4f6;">
                <div style="font-size:.9rem;font-weight:700;color:#111;">📋 Detailed Review</div></div>
              <div style="padding:.875rem 1.25rem;">""", unsafe_allow_html=True)
            for i,q in enumerate(st.session_state.test_questions):
                ua=st.session_state.user_answers.get(i); ok=ua==q["correct"]
                yt=ua or "--"; ct2=q["correct"]
                for opt in q["options"]:
                    if opt and len(opt)>1:
                        if opt[0]==ua: yt=clean_opt(opt)
                        if opt[0]==q["correct"]: ct2=clean_opt(opt)
                wrong_txt=(f'<br>Correct: <span style="color:#22c55e;font-weight:700;">'
                            f'{S(ct2)}</span>') if not ok else ""
                lc="#22c55e" if ok else "#ef4444"
                st.markdown(f"""<div class="rv {'rv-c' if ok else 'rv-w'}">
                  <div style="font-size:.9rem;flex-shrink:0;color:{lc};">{'✓' if ok else '✗'}</div>
                  <div style="flex:1;">
                    <div class="rv-q">Q{i+1}: {S(q['question'])}</div>
                    <div class="rv-a">Your answer:
                      <span style="color:{lc};font-weight:600;">{S(str(yt))}</span>{wrong_txt}</div>
                  </div>
                  <span class="badge {'b-gr' if ok else 'b-re'}">{'Correct' if ok else 'Wrong'}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div></div>', unsafe_allow_html=True)
            r1,r2,r3,r4 = st.columns(4)
            with r1:
                if st.button("🔄 Retake", use_container_width=True, type="secondary"):
                    st.session_state.user_answers={}; st.session_state.test_submitted=False
                    st.session_state.quiz_key+=1; st.rerun()
            with r2:
                if st.button("🎯 New Level", use_container_width=True, type="primary"):
                    st.session_state.selected_difficulty=None
                    st.session_state.has_test_generated=False
                    st.session_state.user_answers={}; st.session_state.test_submitted=False; st.rerun()
            with r3:
                if st.button("🎴 Flashcards", use_container_width=True, type="secondary"): go("Flashcard")
            with r4:
                if st.button("📊 Dashboard",  use_container_width=True, type="secondary"): go("Dashboard")

    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════
elif cp == "Dashboard":
    sh=st.session_state.score_history; qs=st.session_state.questions
    bms=st.session_state.bookmarks; was=st.session_state.wrong_answers
    tt=len(sh); avg=round(sum(s["pct"] for s in sh)/max(tt,1),1) if sh else 0
    best=max((s["pct"] for s in sh),default=0)
    un=st.session_state.current_user or "User"
    greet=f"Welcome back, {un.capitalize()}!" if un!="__guest__" else "Welcome, Guest!"

    st.markdown(f"""<div style="background:#fff;border-bottom:1px solid #e5e7eb;
      padding:1.5rem 1.5rem 1.25rem;">
      <div style="max-width:900px;margin:0 auto;">
        <div style="font-size:.65rem;font-weight:700;text-transform:uppercase;
          letter-spacing:.1em;color:#9ca3af;margin-bottom:.2rem;">Your Progress › Dashboard</div>
        <div style="font-size:1.75rem;font-weight:800;color:#111;letter-spacing:-.03em;">Learning Dashboard</div>
        <div style="font-size:.875rem;color:#6b7280;margin-top:.25rem;">
          {S(greet)} Track your study patterns and scores.</div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="pw" style="padding-top:.25rem;">', unsafe_allow_html=True)

    st.markdown(f"""<div class="sg">
      <div class="sc"><span class="sc-ico">📊</span><div class="sc-val">{tt}</div>
        <div class="sc-lbl">Tests Taken</div></div>
      <div class="sc"><span class="sc-ico">🎯</span><div class="sc-val">{avg}%</div>
        <div class="sc-lbl">Avg Score</div></div>
      <div class="sc"><span class="sc-ico">🏆</span><div class="sc-val">{best}%</div>
        <div class="sc-lbl">Best Score</div></div>
      <div class="sc"><span class="sc-ico">⭐</span><div class="sc-val">{len(bms)}</div>
        <div class="sc-lbl">Bookmarks</div></div>
    </div>""", unsafe_allow_html=True)

    d1,d2 = st.columns(2)
    with d1:
        st.markdown(f"""<div class="sc" style="margin-bottom:.875rem;">
          <span class="sc-ico">📝</span><div class="sc-val">{len(qs)}</div>
          <div class="sc-lbl">Questions Generated</div></div>""", unsafe_allow_html=True)
    with d2:
        st.markdown(f"""<div class="sc" style="margin-bottom:.875rem;">
          <span class="sc-ico">❌</span><div class="sc-val">{len(was)}</div>
          <div class="sc-lbl">Mistakes Tracked</div></div>""", unsafe_allow_html=True)

    if len(sh)>=2:
        st.markdown('<div class="ds-t">📈 Score Trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="ch-w">', unsafe_allow_html=True)
        render_score_chart(sh)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="ds-t">🗂️ Score History</div>', unsafe_allow_html=True)
    if not sh:
        st.markdown("""<div style="background:#fff;border:1.5px solid #e5e7eb;border-radius:14px;
          padding:3rem;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.06);">
          <div style="font-size:2.5rem;margin-bottom:.75rem;">🧪</div>
          <div style="font-size:.95rem;font-weight:700;color:#111;">No tests taken yet</div>
          <div style="font-size:.82rem;color:#9ca3af;margin-top:.375rem;">Complete a test to see your history here.</div>
        </div>""", unsafe_allow_html=True)
        _,bc,_ = st.columns([1,1,1])
        with bc:
            if st.button("🎯 Take a Test Now", type="primary", use_container_width=True): go("Test")
    else:
        for e in reversed(sh):
            pc=e["pct"]; bcol="#22c55e" if pc>=80 else "#e84c1e" if pc>=60 else "#ef4444"
            dcls=e["diff"]
            dbg=(("#f0fdf4","#16a34a") if dcls=="Easy" else
                 ("#fff7ed","#d97706") if dcls=="Medium" else ("#fef2f2","#dc2626"))
            st.markdown(f"""<div class="shr">
              <span class="shr-d" style="background:{dbg[0]};color:{dbg[1]};">{S(e['diff'])}</span>
              <div class="shr-s">{e['score']}/{e['total']}</div>
              <div class="shr-m">
                <div class="shr-pdf">{S(e['pdf'])}</div>
                <div class="shr-dt">{e['date']}</div>
              </div>
              <div class="shr-pb"><div class="shr-pf" style="width:{pc}%;background:{bcol};"></div></div>
              <span class="shr-pct" style="color:{bcol};">{pc}%</span>
            </div>""", unsafe_allow_html=True)
        _,cl,_ = st.columns([1,1,1])
        with cl:
            if st.button("🗑 Clear History", type="secondary", use_container_width=True):
                st.session_state.score_history=[]; persist(); st.rerun()

    if was:
        st.markdown(f'<div class="ds-t">❌ Wrong Answers ({len(was)})</div>', unsafe_allow_html=True)
        for wq in was[:5]:
            ct3=clean_opt(next((o for o in wq["options"] if o.startswith(wq["correct"])),wq["correct"]))
            st.markdown(f"""<div class="rv rv-w">
              <div style="color:#ef4444;font-size:.9rem;flex-shrink:0;">✗</div>
              <div style="flex:1;">
                <div class="rv-q">{S(wq['question'])}</div>
                <div class="rv-a">Correct: <span style="color:#22c55e;font-weight:600;">{S(ct3)}</span></div>
              </div></div>""", unsafe_allow_html=True)
        if len(was)>5: st.caption(f"+ {len(was)-5} more. See all in Study → Wrong Answers.")
        w1,w2=st.columns(2)
        with w1:
            if st.button("📖 Review Wrong Answers",  type="secondary", use_container_width=True): go("Study")
        with w2:
            if st.button("🎴 Flashcard Mistakes", type="secondary", use_container_width=True):
                st.session_state.fc_filter="Mistakes"; go("Flashcard")

    if bms:
        st.markdown(f'<div class="ds-t">⭐ Bookmarks ({len(bms)})</div>', unsafe_allow_html=True)
        for bi in bms[:3]:
            if bi<len(qs):
                st.markdown(f"""<div class="rv" style="border-left:3px solid #e84c1e;background:#fff7ed;">
                  <div style="flex-shrink:0;">⭐</div>
                  <div class="rv-q">{S(qs[bi]['question'])}</div>
                </div>""", unsafe_allow_html=True)
        if len(bms)>3: st.caption(f"+ {len(bms)-3} more bookmarks.")
        if st.button("📖 Study Bookmarks", type="secondary"): go("Study")

    if un=="__guest__":
        st.markdown("""<div style="background:#fffbeb;border:1.5px solid #fde68a;border-radius:12px;
          padding:1rem 1.375rem;font-size:.875rem;color:#92400e;margin-top:1.25rem;">
          💡 <strong>Guest mode</strong> — data resets on page refresh. Create an account to keep data.</div>""",
            unsafe_allow_html=True)
        _,bc3,_ = st.columns([1,1,1])
        with bc3:
            if st.button("Create Account", type="primary", use_container_width=True):
                for k in list(st.session_state.keys()): del st.session_state[k]
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# ABOUT
# ════════════════════════════════════════════════════════════════
elif cp == "About":
    st.markdown("""<div style="background:#fff;border-bottom:1px solid #e5e7eb;
      padding:1.5rem 1.5rem 1.25rem;">
      <div style="max-width:960px;margin:0 auto;">
        <div style="font-size:.65rem;font-weight:700;text-transform:uppercase;
          letter-spacing:.1em;color:#9ca3af;margin-bottom:.2rem;">QuizGenius AI › About</div>
        <div style="font-size:1.75rem;font-weight:800;color:#111;letter-spacing:-.03em;">About QuizGenius AI</div>
        <div style="font-size:.875rem;color:#6b7280;margin-top:.25rem;">
          Revolutionising learning through adaptive AI-powered assessments.</div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="aw" style="padding-top:.25rem;">', unsafe_allow_html=True)
    cs = "background:#fff;border:1.5px solid #e5e7eb;border-radius:14px;padding:1.875rem;margin-bottom:1.125rem;box-shadow:0 1px 4px rgba(0,0,0,.06);"
    ct2 = "font-size:.95rem;font-weight:700;color:#111;margin-bottom:.625rem;"
    cp2 = "font-size:.875rem;color:#374151;line-height:1.85;"
    a1,a2 = st.columns(2)
    with a1:
        st.markdown(f"""<div style="{cs}">
          <div style="{ct2}">🧠 Our Mission</div>
          <p style="{cp2}">QuizGenius AI empowers students and professionals to study smarter.
          Groq-powered Llama 3.1 generates adaptive quizzes with real-time progress tracking.
          Fast, free, and deployed on Render.</p></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div style="{cs}">
          <div style="{ct2}">🚀 Technology Stack</div>
          <p style="{cp2}">
            <strong style="color:#e84c1e;">Llama 3.1</strong> — via Groq API (llama-3.1-8b-instant)<br>
            <strong style="color:#e84c1e;">LangChain</strong> — document splitting<br>
            <strong style="color:#e84c1e;">Tesseract OCR</strong> — scanned PDFs<br>
            <strong style="color:#e84c1e;">Streamlit</strong> — web framework<br>
            <strong style="color:#e84c1e;">Render</strong> — cloud deployment</p></div>""",
            unsafe_allow_html=True)
    with a2:
        st.markdown(f"""<div style="{cs}">
          <div style="{ct2}">✨ Key Features</div>
          <p style="{cp2}">
            🎴 Flashcard mode with 3D flip<br>⭐ Bookmark questions<br>
            🎯 Auto difficulty detection<br>❌ Wrong answer tracker<br>
            ⏱ Per-question timed mode<br>📊 Score history + chart<br>
            🗂️ Topic / chapter filter<br>🔀 MCQ / True-False / Fill-in-Blank<br>
            👁 Live preview before generation<br>📤 Export as formatted HTML<br>
            ⚡ Groq-powered speed</p></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div style="{cs}">
          <div style="{ct2}">📞 Connect</div>
          <a href="https://www.linkedin.com/in/vishwas-patel-ba91a2288/" target="_blank"
            style="display:flex;align-items:center;gap:.75rem;padding:.625rem .875rem;
            background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:10px;
            text-decoration:none;margin-bottom:.5rem;">
            <div style="width:34px;height:34px;border-radius:8px;background:#0077b5;
              display:flex;align-items:center;justify-content:center;flex-shrink:0;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="white">
                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037
                  -1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046
                  c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286z
                  M5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063
                  1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065z
                  M7.119 20.452H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729
                  v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271
                  V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg></div>
            <div><div style="font-size:.8rem;font-weight:700;color:#1e40af;">LinkedIn</div>
              <div style="font-size:.68rem;color:#9ca3af;">Vishwas Patel</div></div>
            <div style="margin-left:auto;color:#9ca3af;">↗</div></a>
          <a href="mailto:patelvishwas702@gmail.com"
            style="display:flex;align-items:center;gap:.75rem;padding:.625rem .875rem;
            background:#fef2f2;border:1.5px solid #fecaca;border-radius:10px;text-decoration:none;">
            <div style="width:34px;height:34px;border-radius:8px;background:#ea4335;
              display:flex;align-items:center;justify-content:center;flex-shrink:0;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="white">
                <path d="M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73
                  L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 0 1 0 19.366V5.457
                  c0-2.023 2.309-3.178 3.927-1.964L5.455 4.64 12 9.548l6.545-4.91
                  1.528-1.145C21.69 2.28 24 3.434 24 5.457z"/></svg></div>
            <div><div style="font-size:.8rem;font-weight:700;color:#dc2626;">Email</div>
              <div style="font-size:.68rem;color:#9ca3af;">patelvishwas702@gmail.com</div></div>
            <div style="margin-left:auto;color:#9ca3af;">↗</div></a>
          <div style="padding-top:.75rem;border-top:1px solid #f3f4f6;margin-top:.625rem;">
            <p style="font-size:.75rem;color:#9ca3af;margin:0;">
              Design &amp; Dev by <strong style="color:#111;">Vishwas Patel</strong></p>
          </div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
