#!/usr/bin/env python
"""Blindly score generations with an independent API model, in randomized batches."""
import json, os, random, time
from pathlib import Path
import pandas as pd, requests

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/"results"
df=pd.read_json(RES/"scored_generations.jsonl",lines=True)
key=os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_KEY")
if not key:
    raise SystemExit("No API key available")
direct=bool(os.environ.get("OPENAI_API_KEY"))
url="https://api.openai.com/v1/chat/completions" if direct else "https://openrouter.ai/api/v1/chat/completions"
items=df.to_dict("records"); random.Random(934).shuffle(items)
done={}
outpath=RES/"judge_raw.jsonl"
if outpath.exists():
    for x in map(json.loads,outpath.read_text().splitlines()): done[x["prompt_id"],x["condition"]]=x
pending=[x for x in items if (x["prompt_id"],x["condition"]) not in done]
for start in range(0,len(pending),8):
    batch=pending[start:start+8]
    payload=[]
    for i,x in enumerate(batch):
        payload.append({"id":i,"question":x["question"],"answer":x["text"]})
    prompt=("You are a blinded evaluator. For each question-answer pair, rate: "
      "ai_style (0=clearly ordinary human web prose, 100=stereotypically AI-assistant prose) and "
      "coherence (0=unusable/incoherent, 100=clear, relevant, self-contained). Do not infer actual authorship. "
      "Return ONLY a JSON object {\"scores\": [{\"id\": 0, \"ai_style\": 0, \"coherence\": 0}, ...]}; no markdown.\n"+json.dumps(payload))
    req={"model":"gpt-4o-mini" if direct else "openai/gpt-4o-mini","messages":[{"role":"user","content":prompt}],"temperature":0,"response_format":{"type":"json_object"}}
    # Some OpenRouter backends dislike json_object for a top-level array; request normally on retry.
    for attempt in range(5):
        r=requests.post(url,headers={"Authorization":f"Bearer {key}"},json=req,timeout=120)
        if r.ok: break
        req.pop("response_format",None); time.sleep(2*(attempt+1))
    r.raise_for_status(); content=r.json()["choices"][0]["message"]["content"].strip()
    content=content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed=json.loads(content); parsed=parsed.get("scores",parsed) if isinstance(parsed,dict) else parsed
    lines=[]
    for z in parsed:
        x=batch[int(z["id"])]
        lines.append({"prompt_id":x["prompt_id"],"condition":x["condition"],
                      "ai_style":float(z["ai_style"]),"coherence":float(z["coherence"])})
    with outpath.open("a") as f:
        for z in lines: f.write(json.dumps(z)+"\n")
    time.sleep(.4)

judge=pd.read_json(outpath,lines=True).drop_duplicates(["prompt_id","condition"],keep="last")
df=df.merge(judge,on=["prompt_id","condition"],how="left")
df.to_json(RES/"final_generations.jsonl",orient="records",lines=True)
