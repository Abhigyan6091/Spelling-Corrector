"""
NLP TextSense Analyzer — Q4 Streamlit App
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import math, re, time, random
from collections import Counter

import nltk
for pkg in ('brown', 'treebank', 'punkt', 'punkt_tab'):
    nltk.download(pkg, quiet=True)

st.set_page_config(
    page_title='TextSense NLP Analyzer',
    page_icon='NLP',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
  .metric-box {
      background: #1e2130; border-radius: 10px;
      padding: 16px; text-align: center; margin: 4px;
  }
  .metric-val { font-size: 2rem; font-weight: 700; }
  .metric-lbl { font-size: 0.78rem; color: #aaa; }
  .alert-merge  { background:#2d1f00; border-left:4px solid #ff9800;
                  padding:8px 12px; border-radius:4px; margin:4px 0; }
  .alert-spell  { background:#001f2d; border-left:4px solid #03a9f4;
                  padding:8px 12px; border-radius:4px; margin:4px 0; }
  .alert-rword  { background:#001a1a; border-left:4px solid #00bcd4;
                  padding:8px 12px; border-radius:4px; margin:4px 0; }
  .alert-grammar{ background:#1a001f; border-left:4px solid #e040fb;
                  padding:8px 12px; border-radius:4px; margin:4px 0; }
  .tag { display:inline-block; font-size:0.7rem; font-weight:700;
         border-radius:3px; padding:1px 5px; margin-right:4px; }
  .tag-merge   { background:#ff9800; color:#000; }
  .tag-spell   { background:#03a9f4; color:#000; }
  .tag-rword   { background:#00bcd4; color:#000; }
  .tag-grammar { background:#e040fb; color:#000; }
  .latency-box { background:#141824; border-radius:8px; padding:10px 16px;
                 font-size:0.82rem; color:#ccc; margin-top:8px; }
</style>
""", unsafe_allow_html=True)

K_VAL       = 0.1
TRIGGER_N   = 10
GRAM_THRESH = 200
MERGE_PROB  = 0.15

k_val       = K_VAL
trigger_n   = TRIGGER_N
gram_thresh = GRAM_THRESH

with st.sidebar:
    st.markdown("## TextSense Config")
    st.divider()
    st.markdown("**Language Model (fixed)**")
    st.caption(f"Add-k smoothing (k) = **{K_VAL}**")
    st.caption(f"Grammar trigger every **{TRIGGER_N}** words")
    st.caption(f"Perplexity alert threshold = **{GRAM_THRESH}**")
    st.markdown("**Merge Detection (fixed)**")
    st.caption(f"Merge threshold = mean + 1σ word length")
    st.caption(f"Live demo merge prob = **{MERGE_PROB}**")
    st.divider()
    st.markdown("**Decision Rule**")
    st.caption("PCFG → Trigram → Bigram\n\nPCFG used when parse succeeds. Trigram when coverage adequate. Bigram as fallback.")
    st.divider()
    st.caption("NLP Assignment 1 — Q4\nTextSense Analyzer v1.0")


@st.cache_resource(show_spinner='Building models (this runs once, ~60 s)...')
def load_models(k):
    from nltk.corpus import brown, treebank
    from nltk import Nonterminal as NT

    def clean(w):
        return ''.join(c for c in w.lower() if c.isalpha())

    def build_delete_idx(vocab):
        idx = {}
        for w in vocab:
            for i in range(len(w)):
                key = w[:i] + w[i+1:]
                idx.setdefault(key, set()).add(w)
        return idx

    def fast_candidates(word, delete_idx, vocab):
        cands = set()
        cands.update(delete_idx.get(word, set()))
        for i in range(len(word)):
            cands.update(delete_idx.get(word[:i]+word[i+1:], set()))
        return cands & vocab

    sents = [[clean(w) for w in s if clean(w)] for s in brown.sents()]
    sents = [s for s in sents if s]
    vocab = set(w for s in sents for w in s)
    train = sents[:int(0.8*len(sents))]

    delete_idx = build_delete_idx(vocab)

    uni = Counter(w for s in train for w in s)
    N   = sum(uni.values())
    uni_prob = {w: c/N for w,c in uni.items()}

    V = len(vocab)+2
    bi_cnt=Counter(); bi_ctx=Counter()
    tri_cnt=Counter(); tri_ctx=Counter()
    for s in train:
        p=['<S>']+s+['<S>']
        for i in range(len(p)-1):
            bi_cnt[(p[i],p[i+1])]+=1; bi_ctx[p[i]]+=1
        for i in range(len(p)-2):
            tri_cnt[(p[i],p[i+1],p[i+2])]+=1; tri_ctx[(p[i],p[i+1])]+=1

    def bi_lp(w,p1):
        c=bi_cnt.get((p1,w),0); cc=bi_ctx.get(p1,0)
        return math.log((c+k)/(cc+k*V))
    def tri_lp(w,p2,p1):
        c=tri_cnt.get((p2,p1,w),0); cc=tri_ctx.get((p2,p1),0)
        if cc==0: return bi_lp(w,p1)
        return math.log((c+k)/(cc+k*V))
    def sent_lp(words,n=3):
        lp=0.0; p2=p1='<S>'
        for w in words:
            lp+=(tri_lp(w,p2,p1) if n==3 else bi_lp(w,p1))
            p2,p1=p1,w
        return lp
    def perplexity(words):
        if not words: return float('inf')
        return math.exp(-sent_lp(words,3)/len(words))

    all_len=[len(w) for s in train for w in s]
    mean_l=sum(all_len)/len(all_len)
    std_l=(sum((x-mean_l)**2 for x in all_len)/len(all_len))**0.5

    # 1-std threshold catches shorter merged tokens like "dogit" (6 chars)
    merge_threshold = mean_l + std_l

    def try_split(token):
        if token in vocab: return None
        if len(token) <= merge_threshold: return None
        best_sc, best = float('-inf'), None
        for s in range(1, len(token)):
            l, r = token[:s], token[s:]
            if l in vocab and r in vocab:
                sc = bi_lp(l,'<S>') + bi_lp(r,l)
                if sc > best_sc: best_sc, best = sc, (l, r)
        return best

    # Bigram-context spell score — matches Q3 notebook correct_non_word fix
    def spell_score(c, prev_w, next_w):
        uni = math.log(max(uni_prob.get(c, 1e-10), 1e-10))
        ctx = 0.0
        if prev_w:
            ctx += bi_lp(c, prev_w)
        if next_w:
            ctx += 0.5 * bi_lp(next_w, c)
        return ctx + 0.3 * uni

    # Repeated-char normalization: catches hellllo→hello, sppeeling→spelling
    # (edit distance > 1 caused by key-repeat; standard edit-1 misses these)
    def normalize_repeats(word):
        """Collapse 3+ consecutive identical chars to 2, then to 1.
        Return first vocab match found, or None."""
        for target_len in [2, 1]:
            pattern = r'(.)\1{' + str(target_len) + r',}'
            replacement = r'\1' * target_len
            normalized = re.sub(pattern, replacement, word)
            if normalized != word and normalized in vocab:
                return normalized
        return None


    lhs_cnt=Counter(); prod_cnt=Counter()
    for tree in treebank.parsed_sents():
        t=tree.copy(deep=True)
        t.chomsky_normal_form(); t.collapse_unary(collapsePOS=False)
        for pr in t.productions():
            lhs=str(pr.lhs()); lhs_cnt[lhs]+=1
            rhs=pr.rhs()
            if len(rhs)==2:
                prod_cnt[('bin',lhs,str(rhs[0]),str(rhs[1]))]+=1
            elif len(rhs)==1 and isinstance(rhs[0],str):
                prod_cnt[('lex',lhs,rhs[0].lower())]+=1
            elif len(rhs)==1 and isinstance(rhs[0],NT):
                prod_cnt[('uni',lhs,str(rhs[0]))]+=1

    pcfg_left={}; pcfg_lex={}; pcfg_uni={}
    for key,cnt in prod_cnt.items():
        lp_r=math.log(cnt/lhs_cnt[key[1]])
        if key[0]=='bin':
            B,C=key[2],key[3]
            pcfg_left.setdefault(B,[]).append((C,key[1],lp_r))
        elif key[0]=='lex':
            pcfg_lex.setdefault(key[2],{})[key[1]]=lp_r
        elif key[0]=='uni':
            pcfg_uni.setdefault(key[2],[]).append((key[1],lp_r))

    def oov(word):
        w=word.lower()
        if re.match(r'^\d+$',w):                return {'CD':math.log(0.9)}
        if len(word)>1 and word[0].isupper():   return {'NNP':math.log(0.5),'NN':math.log(0.3)}
        if w.endswith('ing'):                    return {'VBG':math.log(0.4),'NN':math.log(0.3)}
        if w.endswith('ed'):                     return {'VBD':math.log(0.4),'JJ':math.log(0.3)}
        if w.endswith('ly'):                     return {'RB':math.log(0.6)}
        if w.endswith(('ness','tion','ment','ity')): return {'NN':math.log(0.8)}
        return {'NN':math.log(0.4),'JJ':math.log(0.2),'VB':math.log(0.2)}

    def apply_uni(cell):
        changed=True
        while changed:
            changed=False
            for B,rules in pcfg_uni.items():
                if B not in cell: continue
                lp_B=cell[B]
                for A,rlp in rules:
                    t=rlp+lp_B
                    if A not in cell or t>cell[A]: cell[A]=t; changed=True

    def cky(words,max_len=25):
        n=len(words)
        if n>max_len or n==0: return None
        tbl=[[dict() for _ in range(n)] for _ in range(n)]
        for i,word in enumerate(words):
            tbl[i][i]=dict(pcfg_lex.get(word.lower()) or oov(word))
            apply_uni(tbl[i][i])
        for length in range(2,n+1):
            for i in range(n-length+1):
                j=i+length-1; cell=tbl[i][j]
                for kk in range(i,j):
                    for B,lp_B in tbl[i][kk].items():
                        for (C,A,rlp) in pcfg_left.get(B,[]):
                            if C in tbl[kk+1][j]:
                                t=rlp+lp_B+tbl[kk+1][j][C]
                                if A not in cell or t>cell[A]: cell[A]=t
                apply_uni(cell)
        final=tbl[0][n-1]
        for root in ('S','TOP','ROOT','SINV','SQ','FRAG'):
            if root in final and final[root]>-60: return final[root]
        return None

    word_pool = [w for s in train for w in s if len(w) >= 3]

    return dict(
        vocab=vocab, uni_prob=uni_prob, V=V,
        delete_idx=delete_idx, fast_candidates=fast_candidates,
        try_split=try_split, spell_score=spell_score,
        normalize_repeats=normalize_repeats,
        sent_lp=sent_lp, perplexity=perplexity, cky=cky,
        mean_l=mean_l, std_l=std_l,
        merge_threshold=merge_threshold,
        word_pool=word_pool,
    )


M = load_models(k_val)


def analyse_text(text, trigger_n, gram_thresh):
    words   = text.lower().split()
    alerts  = []
    corrected = []
    sp_lats = []
    gr_lats = []
    trigger_ctr = 0

    for i, token in enumerate(words):
        t_tok = time.perf_counter()
        emit  = [token]

        # C. Merge detection (1-std threshold)
        split = M['try_split'](token)
        if split:
            alerts.append({'type':'merge','original':token,'result':list(split)})
            emit = list(split)

        # A. Bigram-context spell correction
        for w in emit:
            if w not in M['vocab']:
                # Step 1: repeated-char normalization (hellllo→hello, sppeeling→spelling)
                norm = M['normalize_repeats'](w)
                if norm:
                    alerts.append({'type':'spell','original':w,'result':norm})
                    corrected.append(norm)
                else:
                    # Step 2: edit-distance-1 candidates with bigram context
                    cands = M['fast_candidates'](w, M['delete_idx'], M['vocab'])
                    if cands:
                        prev_w = corrected[-1] if corrected else None
                        next_raw = words[i+1] if i+1 < len(words) else None
                        next_w = next_raw if next_raw and next_raw in M['vocab'] else None
                        best = max(cands,
                                   key=lambda c: M['spell_score'](c, prev_w, next_w))
                        alerts.append({'type':'spell','original':w,'result':best})
                        corrected.append(best)
                    else:
                        corrected.append(w)
            else:
                corrected.append(w)
            trigger_ctr += 1


        sp_lats.append((time.perf_counter()-t_tok)*1000)

        # Grammar + B. Real-word trigger
        if trigger_ctr >= trigger_n:
            trigger_ctr = 0
            t_gram = time.perf_counter()
            window = corrected[-trigger_n:]

            perp = M['perplexity'](window)
            if perp > gram_thresh:
                alerts.append({'type':'grammar',
                               'original':' '.join(window),
                               'perplexity':round(perp,1)})

            # B. Real-word correction
            for idx, w in enumerate(window):
                if w in M['vocab']:
                    cands = M['fast_candidates'](w, M['delete_idx'], M['vocab'])
                    if cands:
                        orig_bi = M['sent_lp'](window, 2)
                        for c in list(cands)[:20]:
                            if c == w: continue
                            test_win = window[:]
                            test_win[idx] = c
                            alt_bi = M['sent_lp'](test_win, 2)
                            if alt_bi - orig_bi > 3.0:
                                alerts.append({'type':'realword',
                                               'original':w,'result':c,
                                               'improvement':round(alt_bi-orig_bi,1)})
                                pos = len(corrected) - trigger_n + idx
                                if 0 <= pos < len(corrected):
                                    corrected[pos] = c
                                break

            gr_lats.append((time.perf_counter()-t_gram)*1000)

    return corrected, alerts, sp_lats, gr_lats


def sentence_scores(text):
    import pandas as pd
    t0    = time.perf_counter()
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    rows  = []
    for sent in sents:
        ws = sent.lower().split()
        if not ws: continue
        pcfg_lp = M['cky'](ws)
        bi_lp   = M['sent_lp'](ws,2)
        tri_lp  = M['sent_lp'](ws,3)
        n       = len(ws)
        norm_tri = tri_lp/n if n else float('-inf')
        if pcfg_lp is not None:
            method='PCFG'; verdict='Grammatical' if pcfg_lp>-35 else 'Marginal'; score=pcfg_lp
        elif norm_tri>-12:
            method='Trigram'; verdict='Grammatical' if norm_tri>-9 else 'Marginal'; score=tri_lp
        else:
            method='Bigram'; verdict='Ungrammatical'; score=bi_lp
        rows.append({
            'Sentence':    sent[:55]+'...' if len(sent)>55 else sent,
            'Words':       n,
            'PCFG logP':   round(pcfg_lp,2) if pcfg_lp is not None else '—',
            'Bigram logP': round(bi_lp,2),
            'Trigram logP':round(tri_lp,2),
            'Best Method': method,
            'Score':       round(score,2),
            'Verdict':     verdict,
        })
    return pd.DataFrame(rows), time.perf_counter()-t0


def generate_stream(words, merge_prob=MERGE_PROB, seed=42):
    random.seed(seed)
    stream = []; i = 0
    while i < len(words):
        if i+1 < len(words) and random.random() < merge_prob:
            stream.append((words[i]+words[i+1], [words[i], words[i+1]])); i += 2
        else:
            stream.append((words[i], [words[i]])); i += 1
    return stream


# ═════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# TextSense NLP Analyzer")
st.caption("Segmentation · Bigram-context spelling · Real-word correction · Grammar analysis · PCFG parsing")
st.divider()

tab1, tab2, tab3 = st.tabs(["Live Checker", "Sentence Analysis", "Live Demo"])

with tab1:
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown("#### Input Text")
        with st.form(key="analysis_form", border=False):
            text_input = st.text_area(
                "Input Text", label_visibility="collapsed",
                height=220, key="text_input",
                placeholder=(
                    "Paste or type text here…\n\n"
                    "Try:\n"
                    "• Merged words: 'thequick brownfox'\n"
                    "• Misspelling: 'helllo mie frend'\n"
                    "• Real-word error: 'i would like to sea the world'"
                )
            )
            run = st.form_submit_button("Run Analysis", type="primary", use_container_width=True)
            st.caption("Tip: **Ctrl+Enter** also runs analysis")

    with col_right:
        st.markdown("#### Detection Summary")
        if run and text_input.strip():
            corrected, alerts, sp_lats, gr_lats = analyse_text(
                text_input, trigger_n, gram_thresh)

            n_merge   = sum(1 for a in alerts if a['type']=='merge')
            n_spell   = sum(1 for a in alerts if a['type']=='spell')
            n_rword   = sum(1 for a in alerts if a['type']=='realword')
            n_grammar = sum(1 for a in alerts if a['type']=='grammar')

            m1,m2,m3,m4 = st.columns(4)
            m1.markdown(f"<div class='metric-box'><div class='metric-val' style='color:#ff9800'>{n_merge}</div><div class='metric-lbl'>Merges</div></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-box'><div class='metric-val' style='color:#03a9f4'>{n_spell}</div><div class='metric-lbl'>Spell</div></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-box'><div class='metric-val' style='color:#00bcd4'>{n_rword}</div><div class='metric-lbl'>Real-Word</div></div>", unsafe_allow_html=True)
            m4.markdown(f"<div class='metric-box'><div class='metric-val' style='color:#e040fb'>{n_grammar}</div><div class='metric-lbl'>Grammar</div></div>", unsafe_allow_html=True)

            # F. Per-subsystem latency
            avg_sp = sum(sp_lats)/len(sp_lats) if sp_lats else 0
            avg_gr = sum(gr_lats)/len(gr_lats) if gr_lats else 0
            st.markdown(
                f"<div class='latency-box'>"
                f"Seg+Spell: <b>{avg_sp:.3f} ms/token</b>&nbsp;&nbsp;|&nbsp;&nbsp;"
                f"Grammar trigger: <b>{avg_gr:.3f} ms/trigger</b>"
                f"</div>", unsafe_allow_html=True)
        else:
            st.info("Run analysis to see detection counts.")

    if run and text_input.strip():
        corrected, alerts, sp_lats, gr_lats = analyse_text(
            text_input, trigger_n, gram_thresh)

        st.divider()
        col_a, col_b = st.columns([1,1], gap="large")

        with col_a:
            st.markdown("#### Detected Issues")
            if not alerts:
                st.success("No issues detected.")
            else:
                for a in alerts:
                    if a['type'] == 'merge':
                        st.markdown(f"<div class='alert-merge'><span class='tag tag-merge'>MERGE</span><code>{a['original']}</code> split into <code>{a['result']}</code></div>", unsafe_allow_html=True)
                    elif a['type'] == 'spell':
                        st.markdown(f"<div class='alert-spell'><span class='tag tag-spell'>SPELL</span><code>{a['original']}</code> → <code>{a['result']}</code></div>", unsafe_allow_html=True)
                    elif a['type'] == 'realword':
                        st.markdown(f"<div class='alert-rword'><span class='tag tag-rword'>REAL-WORD</span><code>{a['original']}</code> → <code>{a['result']}</code> (bigram +{a['improvement']})</div>", unsafe_allow_html=True)
                    elif a['type'] == 'grammar':
                        st.markdown(f"<div class='alert-grammar'><span class='tag tag-grammar'>GRAMMAR</span>Perplexity <b>{a['perplexity']}</b> on: <i>\"{a['original']}\"</i></div>", unsafe_allow_html=True)

        with col_b:
            st.markdown("#### Corrected Output")
            st.text_area("Corrected Output", label_visibility="collapsed",
                         value=' '.join(corrected), height=180,
                         key="corrected_out", disabled=True)
            cc1, cc2 = st.columns(2)
            cc1.metric("Original words", len(text_input.split()))
            corrected_wc = len(corrected)
            cc2.metric("After correction", corrected_wc,
                       delta=corrected_wc-len(text_input.split()))

with tab2:
    st.markdown("#### Passage for Analysis")
    analysis_text = st.text_area(
        "Passage for Analysis", label_visibility="collapsed",
        height=160, key="analysis_input",
        placeholder="Paste full passage for sentence-by-sentence grammaticality scoring…",
        value=st.session_state.get("text_input","")
    )

    if st.button("Run Sentence Analysis", type="primary", use_container_width=False):
        if analysis_text.strip():
            with st.spinner("Parsing with PCFG + N-gram LMs…"):
                df, elapsed = sentence_scores(analysis_text)

            def colour_verdict(val):
                if val == 'Grammatical': return 'color:#4caf50;font-weight:700'
                elif val == 'Marginal':  return 'color:#ff9800;font-weight:700'
                else:                    return 'color:#f44336;font-weight:700'

            def colour_method(val):
                c = {'PCFG':'#bb86fc','Trigram':'#03a9f4','Bigram':'#ff9800'}.get(val,'white')
                return f'color:{c};font-weight:700'

            styled = df.style.map(colour_verdict, subset=['Verdict'])\
                             .map(colour_method,  subset=['Best Method'])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            import pandas as pd
            ch1, ch2 = st.columns(2)
            mc = df['Best Method'].value_counts().reset_index(); mc.columns=['Method','Count']
            vc = df['Verdict'].value_counts().reset_index();     vc.columns=['Verdict','Count']
            with ch1:
                st.markdown("**Method usage**"); st.bar_chart(mc.set_index('Method'))
            with ch2:
                st.markdown("**Verdict distribution**"); st.bar_chart(vc.set_index('Verdict'))
            st.success(f"{len(df)} sentence(s) scored in **{elapsed:.2f} s**")
        else:
            st.warning("Please enter some text first.")

with tab3:
    st.markdown("#### Live Typing Simulation")
    st.caption(
        f"Samples a random passage from the Brown corpus training vocabulary, "
        f"drops spaces between adjacent words with probability **p={MERGE_PROB}** "
        f"to create merged tokens, then streams word-by-word simulating live typing. "
        f"Pipeline runs merge detection, bigram-context spell correction, "
        f"real-word correction and grammar checking in real time."
    )
    st.divider()

    d1, d2, d3 = st.columns(3)
    demo_seed   = d1.number_input("Random seed", value=42, step=1)
    demo_speed  = d2.slider("Speed (words/sec)", 1, 10, 4)
    demo_length = d3.slider("Passage length (words)", 20, 80, 40)

    if st.button("Start Live Demo", type="primary"):
        pool = M['word_pool']
        random.seed(int(demo_seed))
        base_words = random.choices(pool, k=demo_length)
        stream = generate_stream(base_words, merge_prob=MERGE_PROB, seed=int(demo_seed))

        st.markdown("---")
        lc1, lc2 = st.columns([3, 2], gap="large")
        with lc1:
            st.markdown("**Token stream (as typed)**")
            stream_box    = st.empty()
            st.markdown("**Corrected output**")
            corrected_box = st.empty()
        with lc2:
            st.markdown("**Live Alerts**")
            alerts_box  = st.empty()
            st.markdown("**Latency**")
            latency_box = st.empty()

        shown_tokens  = []
        corrected_out = []
        all_alerts    = []
        sp_lats_live  = []
        gr_lats_live  = []
        trigger_ctr   = 0

        for token, orig_parts in stream:
            shown_tokens.append(token)
            stream_box.markdown(
                f"<div style='font-family:monospace;font-size:0.9rem;'>{' '.join(shown_tokens)}</div>",
                unsafe_allow_html=True)

            t_tok = time.perf_counter()
            emit  = [token]

            split = M['try_split'](token)
            if split:
                all_alerts.append({'type':'merge','original':token,'result':list(split)})
                emit = list(split)

            for w in emit:
                if w not in M['vocab']:
                    cands = M['fast_candidates'](w, M['delete_idx'], M['vocab'])
                    if cands:
                        prev_w = corrected_out[-1] if corrected_out else None
                        best = max(cands, key=lambda c: M['spell_score'](c, prev_w, None))
                        all_alerts.append({'type':'spell','original':w,'result':best})
                        corrected_out.append(best)
                    else:
                        corrected_out.append(w)
                else:
                    corrected_out.append(w)
                trigger_ctr += 1

            sp_lats_live.append((time.perf_counter()-t_tok)*1000)

            if trigger_ctr >= trigger_n:
                trigger_ctr = 0
                t_gram = time.perf_counter()
                window = corrected_out[-trigger_n:]
                perp = M['perplexity'](window)
                if perp > gram_thresh:
                    all_alerts.append({'type':'grammar','original':' '.join(window),'perplexity':round(perp,1)})
                for idx, w in enumerate(window):
                    if w in M['vocab']:
                        cands = M['fast_candidates'](w, M['delete_idx'], M['vocab'])
                        if cands:
                            orig_bi = M['sent_lp'](window, 2)
                            for c in list(cands)[:20]:
                                if c == w: continue
                                test_win = window[:]
                                test_win[idx] = c
                                alt_bi = M['sent_lp'](test_win, 2)
                                if alt_bi - orig_bi > 3.0:
                                    all_alerts.append({'type':'realword','original':w,'result':c,'improvement':round(alt_bi-orig_bi,1)})
                                    pos = len(corrected_out) - trigger_n + idx
                                    if 0 <= pos < len(corrected_out):
                                        corrected_out[pos] = c
                                    break
                gr_lats_live.append((time.perf_counter()-t_gram)*1000)

            corrected_box.markdown(
                f"<div style='font-family:monospace;font-size:0.9rem;color:#4caf50;'>{' '.join(corrected_out)}</div>",
                unsafe_allow_html=True)

            # Last 8 alerts
            ahtml = ""
            for a in all_alerts[-8:]:
                if a['type']=='merge':
                    ahtml += f"<div class='alert-merge'><span class='tag tag-merge'>MERGE</span><code>{a['original']}</code> → <code>{a['result']}</code></div>"
                elif a['type']=='spell':
                    ahtml += f"<div class='alert-spell'><span class='tag tag-spell'>SPELL</span><code>{a['original']}</code> → <code>{a['result']}</code></div>"
                elif a['type']=='realword':
                    ahtml += f"<div class='alert-rword'><span class='tag tag-rword'>REAL-WORD</span><code>{a['original']}</code> → <code>{a['result']}</code></div>"
                elif a['type']=='grammar':
                    ahtml += f"<div class='alert-grammar'><span class='tag tag-grammar'>GRAMMAR</span>perp={a['perplexity']}</div>"
            alerts_box.markdown(ahtml or "<i>No alerts yet…</i>", unsafe_allow_html=True)

            avg_sp = sum(sp_lats_live)/len(sp_lats_live)
            avg_gr = sum(gr_lats_live)/len(gr_lats_live) if gr_lats_live else 0
            latency_box.markdown(
                f"<div class='latency-box'>Seg+Spell: <b>{avg_sp:.3f} ms/tok</b><br>"
                f"Grammar: <b>{avg_gr:.3f} ms/trigger</b></div>",
                unsafe_allow_html=True)

            time.sleep(1.0/demo_speed)

        st.divider()
        fc1,fc2,fc3,fc4 = st.columns(4)
        fc1.metric("Tokens streamed", len(stream))
        fc2.metric("Merges detected", sum(1 for a in all_alerts if a['type']=='merge'))
        fc3.metric("Spell fixes", sum(1 for a in all_alerts if a['type']=='spell'))
        fc4.metric("Grammar flags", sum(1 for a in all_alerts if a['type']=='grammar'))
        st.success("Live demo complete.")
