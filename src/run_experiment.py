#!/usr/bin/env python
"""End-to-end activation readout and causal steering experiment."""
import argparse, gc, json, os, random, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
try:
    # Torch 2.14 enables a Triton eager override that requires a system C compiler;
    # the stock CUDA bmm is fully adequate for this experiment.
    from torch._native.registry import deregister_op_overrides
    deregister_op_overrides(disable_op_symbols="bmm")
except Exception:
    pass
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT / "data", ROOT / "results"
MODEL_I = "Qwen/Qwen2.5-3B-Instruct"
MODEL_B = "Qwen/Qwen2.5-3B"
SEED = 481516

def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def load_hc3(n_train=240, n_test=60):
    DATA.mkdir(exist_ok=True)
    cache = DATA / "hc3_pairs.jsonl"
    if cache.exists():
        pairs = [json.loads(x) for x in cache.read_text().splitlines()]
    else:
        raw = DATA / "hc3_all.jsonl"
        if not raw.exists():
            u="https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
            rr=requests.get(u,timeout=180); rr.raise_for_status(); raw.write_bytes(rr.content)
        ds = [json.loads(line) for line in raw.read_text().splitlines()]
        pairs = []
        for i, x in enumerate(ds):
            hs, ais = x.get("human_answers") or [], x.get("chatgpt_answers") or []
            if hs and ais and len(hs[0]) > 80 and len(ais[0]) > 80:
                pairs.append({"id": i, "question": x["question"], "human": hs[0], "ai": ais[0],
                              "domain": x.get("source", "unknown")})
        random.Random(SEED).shuffle(pairs)
        cache.write_text("\n".join(json.dumps(x) for x in pairs))
    random.Random(SEED).shuffle(pairs)
    return pairs[:n_train], pairs[n_train:n_train+n_test]

def load_lm(name):
    tok = AutoTokenizer.from_pretrained(name, cache_dir=str(DATA / "hf"))
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
        device_map="cuda", cache_dir=str(DATA / "hf"), low_cpu_mem_usage=True)
    model.eval()
    return tok, model

@torch.inference_mode()
def activations(model, tok, texts, batch=8, maxlen=256):
    out = []
    for k in range(0, len(texts), batch):
        z = tok(texts[k:k+batch], padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
        z = {a:b.cuda() for a,b in z.items()}
        y = model(**z, output_hidden_states=True, use_cache=False).hidden_states
        last = z["attention_mask"].sum(1)-1
        per_layer = [h[torch.arange(h.shape[0], device=h.device), last].float().cpu() for h in y]
        out.append(torch.stack(per_layer, 1))
    return torch.cat(out).numpy()

def get_features(text):
    words = re.findall(r"[A-Za-z']+", text)
    n = max(1, len(words)); sents=max(1,len(re.findall(r"[.!?]",text)))
    first = {"i","me","my","we","us","our"}; informal={"you","your","yeah","ok","okay","really","just","like"}
    return [np.log1p(n), n/sents, sum(w.lower() in first for w in words)/n,
            sum(w.lower() in informal for w in words)/n, sum(len(w) for w in words)/n]

def readout_experiment(model_name, train, test, tag):
    tok, model = load_lm(model_name)
    tr_text = [p[k] for p in train for k in ("human","ai")]
    te_text = [p[k] for p in test for k in ("human","ai")]
    ytr=np.tile([0,1],len(train)); yte=np.tile([0,1],len(test))
    Atr=activations(model,tok,tr_text); Ate=activations(model,tok,te_text)
    directions=[]; rows=[]
    for l in range(Atr.shape[1]):
        d=Atr[ytr==1,l].mean(0)-Atr[ytr==0,l].mean(0); d=d/(np.linalg.norm(d)+1e-9)
        directions.append(d)
        a_tr=Atr[:,l]@d; a_te=Ate[:,l]@d
        auc=roc_auc_score(yte,a_te)
        rows.append({"model":tag,"layer":l,"test_auc":auc,"train_auc":roc_auc_score(ytr,a_tr),
                     "projection_sd":float(np.std(a_tr))})
    # Layer selection uses training data only; test AUROC is untouched.
    df=pd.DataFrame(rows); best=int(df.loc[df.train_auc.idxmax(),"layer"])
    # Honest confound adjustment on held-out examples (descriptive): label ~ projection + observable controls.
    proj=Ate[:,best]@directions[best]
    Xctrl=np.array([get_features(t) for t in te_text])
    domains=np.repeat([p["domain"] for p in test],2).reshape(-1,1)
    oh=OneHotEncoder(sparse_output=False,handle_unknown="ignore").fit_transform(domains)
    X=np.column_stack([proj,Xctrl,oh]); lr=LogisticRegression(max_iter=2000).fit(X,yte)
    ctrl_lr=LogisticRegression(max_iter=2000).fit(np.column_stack([Xctrl,oh]),yte)
    conf={"best_layer":best,"auc_projection":roc_auc_score(yte,proj),
          "auc_controls":roc_auc_score(yte,ctrl_lr.predict_proba(np.column_stack([Xctrl,oh]))[:,1]),
          "auc_combined":roc_auc_score(yte,lr.predict_proba(X)[:,1]),
          "partial_projection_coef":float(lr.coef_[0,0])}
    np.savez_compressed(RESULTS/f"{tag}_activations.npz", directions=np.stack(directions),
                        train=Atr,test=Ate,ytrain=ytr,ytest=yte)
    df.to_csv(RESULTS/f"{tag}_layer_auc.csv",index=False)
    (RESULTS/f"{tag}_confounds.json").write_text(json.dumps(conf,indent=2))
    del model; gc.collect(); torch.cuda.empty_cache()
    return conf

@torch.inference_mode()
def generate_one(model,tok,prompt,layer,direction,scale,seed):
    messages=[{"role":"user","content":prompt}]
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=tok(text,return_tensors="pt").to("cuda")
    handle=None
    if scale != 0:
        vec=torch.tensor(direction*scale,device="cuda",dtype=torch.bfloat16)
        def hook(_m,_i,o):
            h=o[0] if isinstance(o,tuple) else o
            h=h.clone(); h[:,-1,:]+=vec
            return (h,)+o[1:] if isinstance(o,tuple) else h
        handle=model.model.layers[layer-1].register_forward_hook(hook) # hidden_states index l follows l blocks
    torch.manual_seed(seed)
    out=model.generate(**inputs,max_new_tokens=120,do_sample=True,temperature=.75,top_p=.9,
                       pad_token_id=tok.eos_token_id)
    if handle: handle.remove()
    return tok.decode(out[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()

def steering_experiment(test, conf):
    tag="instruct"; npz=np.load(RESULTS/f"{tag}_activations.npz")
    layer=conf["best_layer"]; direction=npz["directions"][layer]
    sd=float(pd.read_csv(RESULTS/f"{tag}_layer_auc.csv").query("layer==@layer").projection_sd.iloc[0])
    rng=np.random.default_rng(SEED); rand=rng.normal(size=direction.shape); rand/=np.linalg.norm(rand)
    tok,model=load_lm(MODEL_I)
    conds=[("minus2",direction,-2), ("minus1",direction,-1),("baseline",direction,0),
           ("plus1",direction,1),("plus2",direction,2),("random_minus2",rand,-2),("random_plus2",rand,2)]
    rows=[]
    for i,p in enumerate(test[:36]):
        prompt=p["question"]+"\n\nGive a helpful, accurate answer."
        for name,d,a in conds:
            txt=generate_one(model,tok,prompt,layer,d,a*sd,SEED+i)
            rows.append({"prompt_id":p["id"],"question":p["question"],"condition":name,"alpha":a,
                         "text":txt,"words":len(txt.split())})
        # prompt-only comparator
        hp=p["question"]+"\n\nAnswer naturally like a human writing online, without sounding like an AI assistant."
        txt=generate_one(model,tok,hp,layer,direction,0,SEED+i)
        rows.append({"prompt_id":p["id"],"question":p["question"],"condition":"human_prompt","alpha":0,
                     "text":txt,"words":len(txt.split())})
    pd.DataFrame(rows).to_json(RESULTS/"generations.jsonl",orient="records",lines=True)
    del model; gc.collect(); torch.cuda.empty_cache()

def detector_scores():
    df=pd.read_json(RESULTS/"generations.jsonl",lines=True)
    name="openai-community/roberta-base-openai-detector"
    tok=AutoTokenizer.from_pretrained(name,cache_dir=str(DATA/"hf"))
    model=AutoModelForSequenceClassification.from_pretrained(name,cache_dir=str(DATA/"hf")).cuda().eval()
    scores=[]
    with torch.inference_mode():
        for k in range(0,len(df),16):
            z=tok(df.text.iloc[k:k+16].tolist(),padding=True,truncation=True,max_length=512,return_tensors="pt").to("cuda")
            pr=model(**z).logits.softmax(-1).cpu().numpy()
            # detector config is LABEL_0 fake, LABEL_1 real
            scores.extend(pr[:,0].tolist())
    df["detector_ai"]=scores
    # A second, independently trained text-side detector. Determine label polarity
    # from a small calibration slice because the repository uses generic labels.
    name2="Hello-SimpleAI/chatgpt-detector-roberta"
    tok2=AutoTokenizer.from_pretrained(name2,cache_dir=str(DATA/"hf"))
    mod2=AutoModelForSequenceClassification.from_pretrained(name2,cache_dir=str(DATA/"hf")).cuda().eval()
    _,cal=load_hc3(n_train=300,n_test=20); ct=[p[k] for p in cal for k in ("human","ai")]
    def predict(m,tok,texts,batch=16):
        ans=[]
        with torch.inference_mode():
            for k in range(0,len(texts),batch):
                z=tok(texts[k:k+batch],padding=True,truncation=True,max_length=512,return_tensors="pt").to("cuda")
                ans.extend(m(**z).logits.softmax(-1).cpu().numpy())
        return np.array(ans)
    cp=predict(mod2,tok2,ct); cy=np.tile([0,1],len(cal)); ai_class=int(np.argmax([cp[cy==1,j].mean()-cp[cy==0,j].mean() for j in range(cp.shape[1])]))
    df["detector_hc3_ai"]=predict(mod2,tok2,df.text.tolist())[:,ai_class]
    del mod2; gc.collect(); torch.cuda.empty_cache()
    # External language-model perplexity as a fluency safeguard.
    p_name="distilbert/distilgpt2"; pt=AutoTokenizer.from_pretrained(p_name,cache_dir=str(DATA/"hf")); pt.pad_token=pt.eos_token
    pm=AutoModelForCausalLM.from_pretrained(p_name,cache_dir=str(DATA/"hf")).cuda().eval(); ppls=[]
    with torch.inference_mode():
        for text in df.text:
            z=pt(text,return_tensors="pt",truncation=True,max_length=512).to("cuda")
            loss=pm(**z,labels=z.input_ids).loss; ppls.append(float(torch.exp(loss.clamp(max=10)).cpu()))
    df["perplexity"]=ppls
    # Off-the-shelf MS MARCO cross-encoder relevance (question, answer).
    r_name="cross-encoder/ms-marco-MiniLM-L6-v2"; rt=AutoTokenizer.from_pretrained(r_name,cache_dir=str(DATA/"hf"))
    rm=AutoModelForSequenceClassification.from_pretrained(r_name,cache_dir=str(DATA/"hf")).cuda().eval(); rel=[]
    with torch.inference_mode():
        for k in range(0,len(df),32):
            qs=df.question.iloc[k:k+32].tolist(); ts=df.text.iloc[k:k+32].tolist()
            z=rt(qs,ts,padding=True,truncation=True,max_length=512,return_tensors="pt").to("cuda")
            rel.extend(rm(**z).logits.squeeze(-1).float().cpu().tolist())
    df["relevance"]=rel
    df["repetition"]=[1-len(set(re.findall(r"\w+",t.lower())))/max(1,len(re.findall(r"\w+",t.lower()))) for t in df.text]
    df.to_json(RESULTS/"scored_generations.jsonl",orient="records",lines=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",default="all")
    a=parser.parse_args(); seed_all(); RESULTS.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
    train,test=load_hc3();
    if a.stage in ("all","readout"):
        ci=readout_experiment(MODEL_I,train,test,"instruct")
        cb=readout_experiment(MODEL_B,train,test,"base")
        di=np.load(RESULTS/"instruct_activations.npz")["directions"]
        db=np.load(RESULTS/"base_activations.npz")["directions"]
        sims=np.sum(di*db,axis=1); pd.DataFrame({"layer":range(len(sims)),"cosine":sims}).to_csv(RESULTS/"base_instruct_cosine.csv",index=False)
    else: ci=json.loads((RESULTS/"instruct_confounds.json").read_text())
    if a.stage in ("all","steer"): steering_experiment(test,ci)
    if a.stage in ("all","score"): detector_scores()

if __name__=="__main__": main()
