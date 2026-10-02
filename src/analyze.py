#!/usr/bin/env python
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"results"; F=ROOT/"paper_draft"/"figures"
F.mkdir(parents=True,exist_ok=True)
df=pd.read_json(R/("final_generations.jsonl" if (R/"final_generations.jsonl").exists() else "scored_generations.jsonl"),lines=True)
metrics=[x for x in ["detector_ai","detector_hc3_ai","ai_style","coherence","perplexity","relevance","words","repetition"] if x in df]
summary=df.groupby("condition")[metrics].agg(["mean","std","count"]); summary.to_csv(R/"condition_summary.csv")

rng=np.random.default_rng(711)
comparisons=[]
base=df[df.condition=="baseline"].set_index("prompt_id")
for cond in [c for c in df.condition.unique() if c!="baseline"]:
    other=df[df.condition==cond].set_index("prompt_id")
    common=base.index.intersection(other.index)
    for m in metrics:
        d=(other.loc[common,m]-base.loc[common,m]).dropna().values
        boots=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(5000)])
        try: p=wilcoxon(d).pvalue if np.any(d!=0) else 1.0
        except ValueError: p=np.nan
        comparisons.append({"condition":cond,"metric":m,"mean_delta":d.mean(),"ci_low":np.quantile(boots,.025),
                            "ci_high":np.quantile(boots,.975),"wilcoxon_p":p,"n":len(d)})
pd.DataFrame(comparisons).to_csv(R/"paired_comparisons.csv",index=False)

# Difference-in-differences against the equal-norm random control.
wide=df.pivot(index="prompt_id",columns="condition",values=metrics)
specific=[]
for cond,rand in [("minus2","random_minus2"),("plus2","random_plus2")]:
    for m in metrics:
        d=(wide[m][cond]-wide[m]["baseline"])-(wide[m][rand]-wide[m]["baseline"])
        d=d.dropna().values
        boots=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(5000)])
        try: p=wilcoxon(d).pvalue if np.any(d!=0) else 1.0
        except ValueError: p=np.nan
        specific.append({"condition":cond,"metric":m,"target_minus_random":d.mean(),
                         "ci_low":np.quantile(boots,.025),"ci_high":np.quantile(boots,.975),
                         "wilcoxon_p":p,"n":len(d)})
pd.DataFrame(specific).to_csv(R/"specificity_comparisons.csv",index=False)

fig,ax=plt.subplots(1,2,figsize=(9,3.3))
for tag,color in [("instruct","#3569a8"),("base","#d05a47")]:
    z=pd.read_csv(R/f"{tag}_layer_auc.csv"); ax[0].plot(z.layer,z.test_auc,label=tag,color=color)
ax[0].axhline(.5,color="k",ls="--",lw=.8); ax[0].set(xlabel="Residual-stream layer",ylabel="Held-out AUROC",ylim=(.45,1.01)); ax[0].legend(frameon=False)
order=["minus2","minus1","baseline","plus1","plus2","random_minus2","random_plus2","human_prompt"]
g=df.groupby("condition").detector_ai.agg(["mean","sem"]).reindex(order)
ax[1].bar(range(len(g)),g["mean"],yerr=1.96*g["sem"],color=["#3569a8"]*5+["#999999"]*2+["#5a9f68"])
ax[1].set_xticks(range(len(g)),["−2","−1","0","+1","+2","rand−","rand+","prompt"],rotation=35)
ax[1].set(xlabel="Intervention (AI direction SD)",ylabel="External detector P(machine)",ylim=(0,1))
fig.tight_layout(); fig.savefig(F/"main_results.pdf",bbox_inches="tight"); fig.savefig(F/"main_results.png",dpi=180,bbox_inches="tight")

meta={}
for tag in ["instruct","base"]: meta[tag]=json.loads((R/f"{tag}_confounds.json").read_text())
cos=pd.read_csv(R/"base_instruct_cosine.csv"); meta["cosine_at_instruct_best"]=float(cos.loc[cos.layer==meta["instruct"]["best_layer"],"cosine"].iloc[0])
if "ai_style" in df: meta["detector_judge_spearman"]=[float(x) for x in spearmanr(df.detector_ai,df.ai_style,nan_policy="omit")]
(R/"analysis_summary.json").write_text(json.dumps(meta,indent=2))
print(summary); print(json.dumps(meta,indent=2))
